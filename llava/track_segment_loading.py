from typing import Literal, Iterator, IO, Any, cast
from collections import defaultdict
from pathlib import Path
import io
from io import RawIOBase

import pickle
import boto3
import torch
from torchcodec import FrameBatch
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


class S3File(io.RawIOBase):
    def __init__(self, s3_object):
        self.s3_object = s3_object
        self.position = 0
        self.cache = {}

    def __repr__(self):
        return "<%s s3_object=%r>" % (type(self).__name__, self.s3_object)

    @property
    def size(self):
        return self.s3_object.content_length

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.position = offset
        elif whence == io.SEEK_CUR:
            self.position += offset
        elif whence == io.SEEK_END:
            self.position = self.size + offset
        else:
            raise ValueError("invalid whence (%r, should be %d, %d, %d)" % (
                whence, io.SEEK_SET, io.SEEK_CUR, io.SEEK_END
            ))

        return self.position

    def seekable(self):
        return True

    def read(self, size=-1):
        # boto3 does not support reading with start at the end of the file
        if self.size == self.position:
            return b''

        if size == -1:
            # Read to the end of the file
            range_header = "bytes=%d-" % self.position
            self.seek(offset=0, whence=io.SEEK_END)
        else:
            new_position = self.position + size

            # If we're going to read beyond the end of the object, return
            # the entire object.
            if new_position >= self.size:
                return self.read()

            range_header = "bytes=%d-%d" % (self.position, new_position - 1)
            self.seek(offset=size, whence=io.SEEK_CUR)

        return self.get(range_header)

    def readable(self):
        return True

    def get(self, range_header: str) -> bytes:
        if range_header in self.cache:
            return self.cache[range_header]
        res = self.s3_object.get(Range=range_header)
        self.cache[range_header] = data = res['Body'].read()
        return data


def download_from_s3(s3, bucket: Bucket, key: Path, seekable: bool = False
                     ) -> S3File:
    obj = s3.Object(bucket_name=bucket, key=str(key))
    if seekable:
        return S3File(obj)
    return obj.get()['Body']


def transpose_frames_to_tracks(frames: FramesBoxes, start: int, height: int,
                               width: int) -> TracksBoxes:
    tracks_ = defaultdict(list)
    for frameid, frame in sorted(frames.items()):
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


def get_track_segment_boxes(frames: FramesBoxes, track_id: int, height: int,
                            width: int, frame_ids: FrameIds
                            ) -> BoundingBoxes:
    boxes = []
    for frame_id in frame_ids:
        frame = frames[frame_id]
        boxes.append(frame[track_id][0])
    return BoundingBoxes(boxes, format='XYXY', canvas_size=(height, width))


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


ALPHA = 0.15
MAX_SIZE = 256
transforms = Compose([
    Smooth(ALPHA),
    ClampBoundingBoxes(),
    ToDtype(dtype={BoundingBoxes: torch.int32, "others": None}),
    ConvertBoundingBoxFormat('XYWH'),
    BoxCrop(MAX_SIZE),
])


def load_video_track_segment(
    video_id: str, track_id: int, timespan: tuple[float, float],
    num_frames: int
) -> FrameBatch:
    s3_client = boto3.resource('s3')
    video_key = VIDEO_PREFIX / video_id
    pkl_key = (PICKLE_PREFIX / video_id).with_suffix(".pkl")
    frames_boxes = pickle.load(
        download_from_s3(s3_client, 'scalable-training-dataset', pkl_key))

    data = download_from_s3(s3_client, 'scalable-training-dataset', video_key,
                            seekable=True)
    decoder = VideoDecoder(data)

    start = int(round(timespan[0] * decoder.metadata.average_fps))
    end = int(round(timespan[1] * decoder.metadata.average_fps))
    frame_ids = torch.linspace(
        start, end - 1, num_frames, dtype=torch.int32).tolist()
    height, width = decoder.metadata.height, decoder.metadata.width
    boxes = get_track_segment_boxes(frames_boxes, track_id, height, width,
                                    frame_ids)

    video_frames = decoder.get_frames_at(frame_ids)

    assert len(boxes) == len(video_frames), (len(boxes), len(video_frames))
    track_frames = transforms(Image(video_frames.data), boxes)

    return FrameBatch(track_frames, video_frames.pts_seconds,
                      video_frames.duration_seconds)
