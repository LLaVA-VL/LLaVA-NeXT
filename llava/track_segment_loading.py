from typing import Literal, Iterator, IO
from collections import defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile

import pickle
import boto3
import torch
from torchcodec import FrameBatch
from torchcodec.decoders import VideoDecoder  # type: ignore
from torchvision.transforms.v2 import Resize, CenterCrop, Compose  # type: ignore
from torchvision.transforms.v2.functional import crop


Bucket = Literal["scalable-training-dataset"]
VIDEO_PREFIX = Path("Simone_28_full/videos/")
PICKLE_PREFIX = Path("gemini_fine_tuning/32k/people_masks/")


FrameId = int
TrackId = int
Box = tuple[float, float, float, float]
BoxI = tuple[int, int, int, int]
Frame = dict[TrackId, tuple[Box, float, int]]
Frames = dict[FrameId, Frame]
Track = list[tuple[FrameId, Box]]
TrackI = list[tuple[FrameId, BoxI]]
Tracks = dict[TrackId, Track]


ALPHA = 0.15


def download_from_s3(s3_client, bucket: Bucket, key: Path) -> IO:
    return s3_client.get_object(Bucket=bucket, Key=str(key))['Body']


def union(b1: Box, b2: Box) -> Box:
    return (
        min(b1[0], b2[0]),
        min(b1[1], b2[1]),
        max(b1[2], b2[2]),
        max(b1[3], b2[3]),
    )


def inter(b1: Box, b2: Box) -> Box:
    return (
        max(b1[0], b2[0]),
        max(b1[1], b2[1]),
        min(b1[2], b2[2]),
        min(b1[3], b2[3]),
    )


def as_int(box: Box) -> BoxI:
    return (
        int(box[0]),
        int(box[1]),
        int(box[2]),
        int(box[3]),
    )


def smooth_boxes(width: float, height: float, boxes: list[Box]
                 ) -> Iterator[BoxI]:
    smooth = boxes[0]
    yield as_int(smooth)
    for box in boxes[1:]:
        smooth = (
            ALPHA * box[0] + (1 - ALPHA) * smooth[0],
            ALPHA * box[1] + (1 - ALPHA) * smooth[1],
            ALPHA * box[2] + (1 - ALPHA) * smooth[2],
            ALPHA * box[3] + (1 - ALPHA) * smooth[3],
        )
        clamped = inter(union(smooth, box), (0, 0, width, height))
        yield as_int(clamped)


def convert_frames_to_tracks(frames: Frames) -> Tracks:
    tracks = defaultdict(list)
    for frameid, frame in frames.items():
        for trackid, (box, _, _) in frame.items():
            tracks[trackid].append((frameid, box))
    return tracks


def smooth_tracks(height: int, width: int, tracks: Tracks
                  ) -> Iterator[tuple[TrackId, list[FrameId], list[BoxI]]]:
    for trackid, track in tracks.items():
        frame_ids = []
        boxes = []
        for frame_id, box in track:
            frame_ids.append(frame_id)
            boxes.append(box)
        yield trackid, frame_ids, list(smooth_boxes(height, width, boxes))


def crop_transforms(frames: FrameBatch, frame_id: int, start: int, box: BoxI,
                    transforms: torch.nn.Module) -> torch.Tensor:
    t = frames[frame_id - start].data
    t = t[:, box[1]:box[3], box[0]:box[2]]
    return transforms(t)


def load_video_tracks(video_id: str, max_size=256
                      ) -> Iterator[tuple[TrackId, torch.Tensor]]:
    s3_client = boto3.client('s3')
    video_key = VIDEO_PREFIX / video_id
    pkl_key = (PICKLE_PREFIX / video_id).with_suffix(".pkl")
    frames = pickle.load(
        download_from_s3(s3_client, 'scalable-training-dataset', pkl_key))
    start, end = min(frames), max(frames)

    # waiting for torchcodec 0.3 for directly reading from boto3 StreamingBody
    data = download_from_s3(s3_client, 'scalable-training-dataset', video_key).read()
    # it is not possible to read bytes directly with torchcodec 0.2
    with NamedTemporaryFile(suffix='.mp4', buffering=0) as f:
        f.write(data)
        decoder = VideoDecoder(f.name)
        video_frames = decoder.get_frames_in_range(start, end+1)
    tracks = convert_frames_to_tracks(frames)
    transforms = Compose([
        Resize(size=None, max_size=max_size),
        CenterCrop(max_size),
    ])
    for track_id, frame_ids, boxes in smooth_tracks(
            decoder.metadata.width, decoder.metadata.height, tracks):
        t = torch.stack([
            crop_transforms(video_frames, frame_id, start, box, transforms)
            for frame_id, box in zip(frame_ids, boxes)
        ])
        yield track_id, t
