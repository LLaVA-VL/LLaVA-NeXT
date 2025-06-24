from typing import Literal, Any, cast
from collections import defaultdict
from pathlib import Path
import io
import time
from math import ceil

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
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
        self._size = None
        try:
            self._size = self.s3_object.content_length
        except ClientError as e:
            if e.response.get('Error', {}).get('Code') == '404':
                self._size = 0
            else:
                self._size = 0
        except Exception:
            self._size = 0

    def __repr__(self):
        return "<%s s3_object=%r>" % (type(self).__name__, self.s3_object)

    @property
    def size(self):
        if self._size is None:
            try:
                self._size = self.s3_object.content_length
            except ClientError as e:
                if e.response.get('Error', {}).get('Code') == '404':
                    self._size = 0
                else:
                    self._size = 0
            except Exception:
                self._size = 0
        return self._size if self._size is not None else 0

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
                return self.read() # This will recursively call read with size=-1

            range_header = "bytes=%d-%d" % (self.position, new_position - 1)
            self.seek(offset=size, whence=io.SEEK_CUR)
        return self.get(range_header)

    def readable(self):
        return True

    def get(self, range_header: str) -> bytes:
        if range_header in self.cache:
            return self.cache[range_header]
        try:
            res = self.s3_object.get(Range=range_header)
            data = res['Body'].read()
            self.cache[range_header] = data
            return data
        except Exception:
            raise


def download_from_s3(s3, bucket: Bucket, key: Path, seekable: bool = False
                     ) -> S3File:
    try:
        # Check if object exists by trying to get its metadata
        try:
            s3.head_object(Bucket=bucket, Key=str(key))
        except ClientError as e:
            if e.response.get('Error', {}).get('Code') == '404':
                # Return a dummy S3File that will return empty data
                class DummyS3Object:
                    def __init__(self, key):
                        self.key = key
                        self.content_length = 0
                    
                    def get(self, **kwargs):
                        return {'Body': io.BytesIO(b'')}
                
                dummy_obj = DummyS3Object(str(key))
                return S3File(dummy_obj)
            else:
                raise
        
        # Create a wrapper object that mimics the boto3 resource interface
        class S3ClientWrapper:
            def __init__(self, client, bucket, key):
                self.client = client
                self.bucket = bucket
                self.key = key
                self._content_length = None
            
            @property
            def content_length(self):
                if self._content_length is None:
                    try:
                        response = self.client.head_object(Bucket=self.bucket, Key=self.key)
                        self._content_length = response['ContentLength']
                    except:
                        self._content_length = 0
                return self._content_length
            
            def get(self, **kwargs):
                return self.client.get_object(Bucket=self.bucket, Key=self.key, **kwargs)
        
        obj = S3ClientWrapper(s3, bucket, str(key))
        
        if seekable:
            return S3File(obj)
        body = obj.get()['Body']
        return body
    except Exception:
        raise


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


def get_track_boxes(frames: FramesBoxes, track_id: TrackId, width: int,
                    height: int) -> tuple[FrameIds, BoundingBoxes]:
    boxes = []
    frame_ids = []
    for frame_id, frame in frames.items():
        if track_id in frame:
            frame_ids.append(frame_id)
            boxes.append(frame[track_id][0])
    return frame_ids, BoundingBoxes(boxes, format='XYXY',
                                    canvas_size=(height, width))


def get_boxes_at(boxes: BoundingBoxes, all_frame_ids: FrameIds,
                 frame_ids: FrameIds) -> BoundingBoxes:
    # the track is not guaranted to have box in all consecutive frames
    # get the box from the closest frame id
    diff_matrix = torch.tensor(all_frame_ids).reshape([-1, 1]) \
        - torch.tensor(frame_ids)
    min_indices = abs(diff_matrix).argmin(0)
    return wrap(boxes.data[min_indices], like=boxes)


def get_track_segment_boxes(frames: FramesBoxes, track_id: int, height: int,
                            width: int, frame_ids: FrameIds
                            ) -> BoundingBoxes:
    # the track is not guaranted to have box in all consecutive frames
    # get the box from the closest frame id
    all_frame_ids = [frame_id for frame_id, frame in frames.items()
                     if track_id in frame]
    diff_matrix = torch.tensor(all_frame_ids).reshape([-1, 1]) \
        - torch.tensor(frame_ids)
    min_indices = abs(diff_matrix).argmin(0).tolist()
    frame_ids = [all_frame_ids[i] for i in min_indices]

    boxes = [frames[frame_id][track_id][0] for frame_id in frame_ids]
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
        # minimum size for resizing
        min_size = ceil(max(width, height) / max_size)
        if width >= min_size and height >= min_size:
            # can't crop with 0 width or height
            image = crop(image, top, left, height, width)
            # output size should be > 0 for height or width
            image = resize(image, size=None, max_size=max_size)
            image = square_pad(image)
        else:
            image = torch.full([image.shape[0], max_size, max_size],
                               fill_value=127, dtype=image.dtype)
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
    ClampBoundingBoxes(),
    ToDtype(dtype={BoundingBoxes: torch.int32, "others": None}),
    ConvertBoundingBoxFormat('XYWH'),
    BoxCrop(MAX_SIZE),
])


def load_video_track_segment(
    video_id: str, track_id: int, timespan: tuple[float, float],
    num_frames: int
) -> FrameBatch:
    """Load a video track segment for the given `track_id` and `timespan`,
    resized to `num_frames`."""
    from llava.utils import rank0_print
    import os
    
    start, end = timespan
    bucket = "scalable-training-dataset"
    
    # Initialize S3 with proper region and retry on credentials issues
    max_s3_retries = 3
    s3 = None
    for retry in range(max_s3_retries):
        try:
            s3 = boto3.client('s3', region_name='eu-west-1')
            # Test credentials by making a simple call
            s3.head_object(Bucket=bucket, Key="test")  # This will fail but test credentials
            break
        except ClientError as e:
            if e.response.get('Error', {}).get('Code') == '404':
                # 404 is expected for test key, credentials are working
                break
            elif 'credentials' in str(e).lower() and retry < max_s3_retries - 1:
                rank0_print(f"S3 credentials issue (attempt {retry+1}), retrying...")
                time.sleep(1)
                continue
            else:
                rank0_print(f"S3 initialization failed: {e}")
                raise
        except Exception as e:
            if 'credentials' in str(e).lower() and retry < max_s3_retries - 1:
                rank0_print(f"S3 initialization failed (attempt {retry+1}): {e}, retrying...")
                time.sleep(1)
                continue
            else:
                rank0_print(f"S3 initialization failed: {e}")
                raise
    
    if s3 is None:
        raise RuntimeError("Failed to initialize S3 client after retries")

    try:
        video_file = download_from_s3(s3, bucket, VIDEO_PREFIX / video_id, seekable=True)
        # Check if the downloaded file has reasonable size
        if hasattr(video_file, 'size'):
            file_size = video_file.size
            if file_size < 1000:  # Less than 1KB is likely corrupted
                rank0_print(f"Video file too small ({file_size} bytes): {video_id}")
                raise ValueError(f"Video file {video_id} is too small ({file_size} bytes)")
            elif file_size > 0 and hash(video_id) % 50 == 0:  # Log ~2% of downloads
                rank0_print(f"Downloaded video {video_id}: {file_size} bytes")
    except Exception as e:
        rank0_print(f"S3 download failed {video_id}: {e}")
        # Re-raise the exception instead of creating dummy data
        raise


    try:
        decoder = VideoDecoder(video_file, device='cpu')
        
        # Handle timespan validation by catching the error and clipping
        original_start, original_end = start, end
        decoded_frames = None
        
        # Try original timespan first
        try:
            decoded_frames = decoder.get_frames_played_in_range(start, end)
        except Exception as timespan_error:
            error_str = str(timespan_error)
            
            # Only handle timespan validation errors, not other errors
            if "Invalid" in error_str and ("start seconds" in error_str or "stop seconds" in error_str):
                rank0_print(f"Timespan validation failed for {video_id} [{start}, {end}]: {error_str}")
                
                # Try to extract valid range from error message and clip
                clipped_start, clipped_end = None, None
                
                # Handle different error formats
                if "Invalid start seconds" in error_str and "must be greater than or equal to" in error_str:
                    # Format: "Invalid start seconds: 0.23589285714285715. It must be greater than or equal to 1.41 and less than or equal to 14.610011."
                    try:
                        if " and less than or equal to " in error_str:
                            parts = error_str.split("greater than or equal to ")[1].split(" and less than or equal to ")
                            min_start = float(parts[0])
                            max_end = float(parts[1].rstrip("."))
                            clipped_start = max(min_start + 0.001, min(start, max_end - 0.1))
                            clipped_end = min(max_end - 0.001, max(clipped_start + 0.1, end))
                    except:
                        pass
                elif "Invalid stop seconds" in error_str and "must be less than or equal to" in error_str:
                    # Format: "Invalid stop seconds: 16.0. It must be less than or equal to 14.610011."
                    try:
                        parts = error_str.split("must be less than or equal to ")[1]
                        max_end = float(parts.rstrip("."))
                        clipped_start = max(0.001, start)
                        clipped_end = min(max_end - 0.001, max(clipped_start + 0.1, end))
                    except:
                        pass
                
                if clipped_start is not None and clipped_end is not None:
                    try:
                        rank0_print(f"Clipping timespan for {video_id}: [{start:.3f}, {end:.3f}] -> [{clipped_start:.3f}, {clipped_end:.3f}]")
                        decoded_frames = decoder.get_frames_played_in_range(clipped_start, clipped_end)
                    except Exception as clip_error:
                        rank0_print(f"Failed to clip timespan for {video_id}: {clip_error}")
                        # Don't re-raise timespan_error, let it fall through to raise the original error
                        pass
                
                # If clipping failed or couldn't parse error, re-raise original
                if decoded_frames is None:
                    raise timespan_error
            else:
                # Not a timespan validation error, re-raise
                raise timespan_error

        if decoded_frames is None or decoded_frames.data is None or len(decoded_frames.data) == 0:
            rank0_print(f"No frames decoded: {video_id} timespan {start}-{end}")
            raise ValueError(f"No frames decoded for {video_id} in timespan {start}-{end}")

        # Sample num_frames from the decoded frames
        total_frames = len(decoded_frames.data)
        # Only log occasionally to reduce spam
        if hash(video_id) % 100 == 0:  # Log ~1% of videos
            rank0_print(f"Decoded {total_frames} frames for {video_id}, pts range: [{decoded_frames.pts_seconds[0]:.2f}, {decoded_frames.pts_seconds[-1]:.2f}]")
        
        if total_frames <= num_frames:
            frames_data = decoded_frames.data
            frames_pts = decoded_frames.pts_seconds
            frames_duration = decoded_frames.duration_seconds
        else:
            # Sample evenly from the available frames
            interval = (total_frames - 1) / (num_frames - 1) if num_frames > 1 else 0
            indices = [round(i * interval) for i in range(num_frames)]
            
            frames_data = []
            frames_pts = []
            frames_duration = []
            for frame_idx in indices:
                try:
                    frames_data.append(decoded_frames.data[frame_idx])
                    frames_pts.append(decoded_frames.pts_seconds[frame_idx])
                    frames_duration.append(decoded_frames.duration_seconds[frame_idx])
                except (IndexError, Exception):
                    pass
            
            if frames_data:
                frames_data = torch.stack(frames_data)
                frames_pts = torch.tensor(frames_pts)
                frames_duration = torch.tensor(frames_duration)
            else:
                rank0_print(f"No frames sampled: {video_id}")
                raise ValueError(f"No frames could be sampled for {video_id}")

        frames = FrameBatch(frames_data, frames_pts, frames_duration)

    except Exception as e:
        rank0_print(f"Video loading exception {video_id}: {e}")
        # Re-raise the exception instead of creating dummy data
        raise

    return frames
