from typing import Literal, Iterator, IO, Any, cast
from collections import defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile

import pickle
import boto3
import torch
from torchcodec.decoders import VideoDecoder
from torchvision.tv_tensors import Image, BoundingBoxes, BoundingBoxFormat, \
    wrap
from torchvision.transforms.v2 import Compose, Transform, ClampBoundingBoxes, \
    ToDtype, ConvertBoundingBoxFormat
from torchvision.transforms.v2.functional import pad, crop, resize, \
    convert_bounding_box_format


Bucket = Literal["scalable-training-dataset"]
VIDEO_PREFIX = Path("Simone_28_full/videos/")
PICKLE_PREFIX = Path("gemini_fine_tuning/32k/people_masks/")


FrameId = int
FrameIds = list[int]
TrackId = int
Box = tuple[float, float, float, float]
Boxes = list[Box]
FramesBoxes = dict[FrameId, dict[TrackId, tuple[Box, float, int]]]
TracksBoxes = dict[TrackId, tuple[FrameIds, BoundingBoxes]]


ALPHA = 0.15


def download_from_s3(s3_client, bucket: Bucket, key: Path) -> IO:
    return s3_client.get_object(Bucket=bucket, Key=str(key))['Body']


def transpose_frames_to_tracks(frames: FramesBoxes, start: int, height: int,
                               width: int) -> TracksBoxes:
    tracks_ = defaultdict(list)
    for frameid, frame in frames.items():
        for trackid, (box, _, _) in frame.items():
            tracks_[trackid].append((frameid - start, box))
    tracks = {}
    for trackid in tracks_:
        frame_ids_, boxes_ = zip(*tracks_[trackid])
        frame_ids = cast(FrameIds, list(frame_ids_))
        boxes = BoundingBoxes(
            boxes_, format='XYXY', canvas_size=(height, width))
        tracks[trackid] = (frame_ids, boxes)
    return tracks


def smooth_boxes(boxes: torch.Tensor, format: BoundingBoxFormat, alpha: float
                 ) -> torch.Tensor:
    smooth = torch.zeros_like(boxes)
    smooth[0] = boxes[0]
    for i in range(1, len(boxes)):
        smooth[i] = alpha * boxes[i] + (1 - alpha) * smooth[i-1]

    smooth = convert_bounding_box_format(
        smooth, format, BoundingBoxFormat.XYXY)
    # make sure smoothed box contains the original box
    x1y1 = torch.minimum(boxes[:, :2], smooth[:, :2])
    x2y2 = torch.maximum(boxes[:, 2:], smooth[:, 2:])
    return torch.cat([x1y1, x2y2], dim=-1)


class Smooth(Transform):

    def __init__(self, alpha: float):
        super(Smooth, self).__init__()
        self.alpha = alpha

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if isinstance(inpt, BoundingBoxes):
            return wrap(smooth_boxes(inpt, inpt.format, self.alpha), like=inpt)
        return inpt


class Margin(Transform):

    def __init__(self, alpha: float):
        super(Margin, self).__init__()
        self.alpha = alpha

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if isinstance(inpt, BoundingBoxes):
            format = inpt.format
            boxes = convert_bounding_box_format(
                inpt.data, format, BoundingBoxFormat.CXCYWH)
            boxes[:, 2:] = boxes[:, 2:] * self.alpha
            boxes = convert_bounding_box_format(
                boxes, BoundingBoxFormat.CXCYWH, format)
            return wrap(boxes, like=inpt)
        return inpt


def square_pad(image: torch.Tensor) -> torch.Tensor:
    height, width = image.shape[-2:]
    max_size = max(height, width)
    min_size = min(height, width)
    padding_value = (max_size - min_size) // 2

    padding = [0, 0, 0, 0]
    left, top, right, bottom = 0, 1, 2, 3
    if height < width:
        a, b = top, bottom
    else:
        a, b = left, right

    padding[a] = padding_value
    padding[b] = max_size - (padding[a] + min_size)

    assert padding[a] + min_size + padding[b] == max_size

    return pad(image, padding=padding, fill=127)


def boxes_crop(images: torch.Tensor, boxes: torch.Tensor,
               format: BoundingBoxFormat, max_size: int) -> torch.Tensor:
    def box_crop(image: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        left, top, width, height = box[0], box[1], box[2], box[3]
        image = crop(image, top, left, height, width)
        image = resize(image, size=None, max_size=max_size)
        image = square_pad(image)
        return image
    boxes = convert_bounding_box_format(boxes, format, BoundingBoxFormat.XYWH)
    return torch.stack([box_crop(image, box)
                        for image, box in zip(images, boxes)])


class BoxCrop(torch.nn.Module):

    def __init__(self, max_size: int):
        super(BoxCrop, self).__init__()
        self.max_size = max_size

    def forward(self, images: Image, boxes: BoundingBoxes) -> Image:
        return Image(boxes_crop(images.data, boxes.data, boxes.format,
                                max_size=self.max_size))


def load_video_tracks(video_id: str, max_size=256
                      ) -> Iterator[tuple[TrackId, torch.Tensor]]:
    s3_client = boto3.client('s3')
    video_key = VIDEO_PREFIX / video_id
    pkl_key = (PICKLE_PREFIX / video_id).with_suffix(".pkl")
    frames_boxes = pickle.load(
        download_from_s3(s3_client, 'scalable-training-dataset', pkl_key))
    start, end = min(frames_boxes), max(frames_boxes)

    # waiting for torchcodec 0.3 for directly reading from boto3 StreamingBody
    data = download_from_s3(s3_client, 'scalable-training-dataset', video_key).read()
    # it is not possible to read bytes directly with torchcodec 0.2
    with NamedTemporaryFile(suffix='.mp4', buffering=0) as f:
        f.write(data)
        decoder = VideoDecoder(f.name, num_ffmpeg_threads=1)
        height, width = decoder.metadata.height, decoder.metadata.width
        video_frames = decoder.get_frames_in_range(start, end+1)

    tracks_boxes = transpose_frames_to_tracks(
        frames_boxes, start, height, width)

    transforms = Compose([
        Smooth(ALPHA),
        # Margin(1.5),
        ClampBoundingBoxes(),
        ToDtype(dtype={BoundingBoxes: torch.int32, "others": None}),
        ConvertBoundingBoxFormat('XYWH'),
        BoxCrop(max_size),
    ])
    for track_id, (frame_ids, boxes) in tracks_boxes.items():
        track_frames = Image(video_frames.data[frame_ids])
        track_frames = transforms(track_frames, boxes)
        yield track_id, track_frames
