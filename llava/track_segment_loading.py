from typing import Literal, Any, cast
from collections import defaultdict
from pathlib import Path
import io
from math import ceil
import traceback

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
        from llava.utils import rank0_print
        rank0_print(f"DEBUG_LOG: S3File.__init__ called for s3_object: {s3_object.key if hasattr(s3_object, 'key') else 'N/A'}")
        self.s3_object = s3_object
        self.position = 0
        self.cache = {}
        self._size = None
        try:
            rank0_print(f"DEBUG_LOG: S3File.__init__ - Attempting to get content_length for {s3_object.key if hasattr(s3_object, 'key') else 'N/A'}")
            self._size = self.s3_object.content_length
            rank0_print(f"DEBUG_LOG: S3File.__init__ - Got content_length: {self._size} for {s3_object.key if hasattr(s3_object, 'key') else 'N/A'}")
        except ClientError as e:
            if e.response.get('Error', {}).get('Code') == '404':
                rank0_print(f"DEBUG_LOG: S3File.__init__ - S3 404 Not Found for {s3_object.key if hasattr(s3_object, 'key') else 'N/A'}. Setting size to 0.")
                self._size = 0
            else:
                rank0_print(f"DEBUG_LOG: ClientError in S3File.__init__ getting content_length for {s3_object.key if hasattr(s3_object, 'key') else 'N/A'}: {e}. Traceback: {traceback.format_exc()}")
                self._size = 0
        except Exception as e:
            rank0_print(f"DEBUG_LOG: ERROR in S3File.__init__ getting content_length for {s3_object.key if hasattr(s3_object, 'key') else 'N/A'}: {e}. Traceback: {traceback.format_exc()}")
            self._size = 0

    def __repr__(self):
        return "<%s s3_object=%r>" % (type(self).__name__, self.s3_object)

    @property
    def size(self):
        from llava.utils import rank0_print
        if self._size is None:
            try:
                rank0_print(f"DEBUG_LOG: S3File.size - Attempting to re-fetch content_length for {self.s3_object.key if hasattr(self.s3_object, 'key') else 'N/A'}")
                self._size = self.s3_object.content_length
                rank0_print(f"DEBUG_LOG: S3File.size - Re-fetched content_length: {self._size} for {self.s3_object.key if hasattr(self.s3_object, 'key') else 'N/A'}")
            except ClientError as e:
                if e.response.get('Error', {}).get('Code') == '404':
                    rank0_print(f"DEBUG_LOG: S3File.size - S3 404 Not Found (re-fetch) for {self.s3_object.key if hasattr(self.s3_object, 'key') else 'N/A'}. Setting size to 0.")
                    self._size = 0
                else:
                    rank0_print(f"DEBUG_LOG: ClientError in S3File.size re-fetching content_length for {self.s3_object.key if hasattr(self.s3_object, 'key') else 'N/A'}: {e}. Traceback: {traceback.format_exc()}")
                    self._size = 0
            except Exception as e:
                rank0_print(f"DEBUG_LOG: ERROR in S3File.size re-fetching content_length for {self.s3_object.key if hasattr(self.s3_object, 'key') else 'N/A'}: {e}. Traceback: {traceback.format_exc()}")
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
        from llava.utils import rank0_print
        rank0_print(f"DEBUG_LOG: S3File.read called for {self.s3_object.key if hasattr(self.s3_object, 'key') else 'N/A'}. Position: {self.position}, Read size: {size}")
        # boto3 does not support reading with start at the end of the file
        if self.size == self.position:
            rank0_print(f"DEBUG_LOG: S3File.read - At EOF for {self.s3_object.key if hasattr(self.s3_object, 'key') else 'N/A'}. Returning empty bytes.")
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
                rank0_print(f"DEBUG_LOG: S3File.read - Read request exceeds EOF for {self.s3_object.key if hasattr(self.s3_object, 'key') else 'N/A'}. Reading till end.")
                return self.read() # This will recursively call read with size=-1

            range_header = "bytes=%d-%d" % (self.position, new_position - 1)
            self.seek(offset=size, whence=io.SEEK_CUR)
        rank0_print(f"DEBUG_LOG: S3File.read - Range header for S3 GET: {range_header} for {self.s3_object.key if hasattr(self.s3_object, 'key') else 'N/A'}")
        return self.get(range_header)

    def readable(self):
        return True

    def get(self, range_header: str) -> bytes:
        from llava.utils import rank0_print
        rank0_print(f"DEBUG_LOG: S3File.get called for {self.s3_object.key if hasattr(self.s3_object, 'key') else 'N/A'} with range: {range_header}")
        if range_header in self.cache:
            rank0_print(f"DEBUG_LOG: S3File.get - Cache hit for range {range_header} in {self.s3_object.key if hasattr(self.s3_object, 'key') else 'N/A'}")
            return self.cache[range_header]
        try:
            rank0_print(f"DEBUG_LOG: S3File.get - S3 GET Object for {self.s3_object.key if hasattr(self.s3_object, 'key') else 'N/A'} with range: {range_header}")
            res = self.s3_object.get(Range=range_header)
            data = res['Body'].read()
            self.cache[range_header] = data
            rank0_print(f"DEBUG_LOG: S3File.get - S3 GET successful for {self.s3_object.key if hasattr(self.s3_object, 'key') else 'N/A'} with range: {range_header}. Data length: {len(data)}")
            return data
        except Exception as e:
            rank0_print(f"DEBUG_LOG: ERROR in S3File.get for {self.s3_object.key if hasattr(self.s3_object, 'key') else 'N/A'} with range {range_header}: {e}. Traceback: {traceback.format_exc()}")
            raise


def download_from_s3(s3, bucket: Bucket, key: Path, seekable: bool = False
                     ) -> S3File:
    from llava.utils import rank0_print
    rank0_print(f"DEBUG_LOG: download_from_s3 called. Bucket: {bucket}, Key: {key}, Seekable: {seekable}")
    try:
        obj = s3.Object(bucket_name=bucket, key=str(key))
        rank0_print(f"DEBUG_LOG: download_from_s3 - s3.Object() created for Key: {key}")
        
        # Check if object exists by trying to get its metadata
        try:
            obj.load()  # This will raise ClientError if object doesn't exist
            rank0_print(f"DEBUG_LOG: download_from_s3 - Object exists for Key: {key}")
        except ClientError as e:
            if e.response.get('Error', {}).get('Code') == '404':
                rank0_print(f"DEBUG_LOG: download_from_s3 - S3 404 Not Found for Key: {key}. Creating dummy file.")
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
        
        if seekable:
            rank0_print(f"DEBUG_LOG: download_from_s3 - Returning S3File for Key: {key}")
            return S3File(obj)
        rank0_print(f"DEBUG_LOG: download_from_s3 - Attempting obj.get()['Body'] for Key: {key}")
        body = obj.get()['Body']
        rank0_print(f"DEBUG_LOG: download_from_s3 - Successfully got Body for Key: {key}. Returning Body.")
        return body
    except Exception as e:
        rank0_print(f"DEBUG_LOG: ERROR in download_from_s3 for Bucket: {bucket}, Key: {key}: {e}. Traceback: {traceback.format_exc()}")
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
    rank0_print(f"DEBUG_LOG: load_video_track_segment entered for video_id: {video_id}, track_id: {track_id}, timespan: {timespan}, num_frames: {num_frames}")

    start, end = timespan
    bucket = "scalable-training-dataset"
    # Configure S3 client with timeouts
    s3_config = Config(
        connect_timeout=10,  # seconds
        read_timeout=30      # seconds
    )
    s3 = boto3.resource('s3', config=s3_config)
    rank0_print("DEBUG_LOG: load_video_track_segment - boto3.resource('s3') initialized with timeouts.")

    video_file = download_from_s3(
        s3, bucket, VIDEO_PREFIX / f'{video_id}.mp4', seekable=True)
    rank0_print("DEBUG_LOG: load_video_track_segment - Downloaded video file from S3.")

    # Check if video file is empty or missing
    if hasattr(video_file, 'size') and video_file.size == 0:
        rank0_print(f"DEBUG_LOG: load_video_track_segment - Video file is empty for {video_id}. Creating dummy FrameBatch.")
        dummy_data = torch.zeros((num_frames if num_frames > 0 else 1, 3, 224, 224), dtype=torch.uint8)
        dummy_pts = torch.zeros(num_frames if num_frames > 0 else 1, dtype=torch.float64)
        dummy_duration = torch.full((num_frames if num_frames > 0 else 1,), 0.033, dtype=torch.float64)  # ~30fps
        return FrameBatch(dummy_data, dummy_pts, dummy_duration)

    # Use torchcodec VideoDecoder API correctly
    try:
        rank0_print(f"DEBUG_LOG: load_video_track_segment - Before VideoDecoder initialization. Start: {start}, End: {end}")
        decoder = VideoDecoder(video_file, device='cpu')
        rank0_print("DEBUG_LOG: load_video_track_segment - VideoDecoder initialized.")
        
        # Use get_frames_played_in_range to get frames in the time range
        rank0_print(f"DEBUG_LOG: load_video_track_segment - Before get_frames_played_in_range({start}, {end})")
        decoded_frames = decoder.get_frames_played_in_range(start, end)
        rank0_print(f"DEBUG_LOG: load_video_track_segment - After get_frames_played_in_range. Number of decoded frames: {len(decoded_frames.data) if decoded_frames and decoded_frames.data is not None else 'None'}")

        if decoded_frames is None or decoded_frames.data is None or len(decoded_frames.data) == 0:
            rank0_print(f"DEBUG_LOG: ERROR in load_video_track_segment - No frames decoded for {video_id} between {start} and {end}. Creating dummy FrameBatch.")
            # Create a dummy FrameBatch to avoid crashing downstream, though this indicates a problem.
            # A single black frame of a common size.
            dummy_data = torch.zeros((1, 3, 224, 224), dtype=torch.uint8)
            dummy_pts = torch.tensor([0.0], dtype=torch.float64)
            dummy_duration = torch.tensor([0.033], dtype=torch.float64)  # ~30fps
            return FrameBatch(dummy_data, dummy_pts, dummy_duration)

        # Sample num_frames from the decoded frames
        total_frames = len(decoded_frames.data)
        if total_frames <= num_frames:
            # Use all available frames if we have fewer than requested
            frames_data = decoded_frames.data
            frames_pts = decoded_frames.pts_seconds
            frames_duration = decoded_frames.duration_seconds
        else:
            # Sample evenly from the available frames
            interval = (total_frames - 1) / (num_frames - 1) if num_frames > 1 else 0
            indices = [round(i * interval) for i in range(num_frames)]
            rank0_print(f"DEBUG_LOG: load_video_track_segment - Frame indices for sampling: {indices}")
            
            frames_data = []
            frames_pts = []
            frames_duration = []
            for i, frame_idx in enumerate(indices):
                try:
                    rank0_print(f"DEBUG_LOG: load_video_track_segment - Accessing decoded_frames at index {frame_idx} (sample {i})")
                    current_frame_data = decoded_frames.data[frame_idx]
                    current_frame_pts = decoded_frames.pts_seconds[frame_idx]
                    current_frame_duration = decoded_frames.duration_seconds[frame_idx]
                    frames_data.append(current_frame_data)
                    frames_pts.append(current_frame_pts)
                    frames_duration.append(current_frame_duration)
                    rank0_print(f"DEBUG_LOG: load_video_track_segment - Successfully accessed frame {frame_idx}. Shape: {current_frame_data.shape}, PTS: {current_frame_pts}")
                except IndexError:
                    rank0_print(f"DEBUG_LOG: WARNING in load_video_track_segment - IndexError when accessing frame {frame_idx} (sample {i}) for video {video_id}. Total decoded: {total_frames}. Skipping this frame sample.")
                    pass
                except Exception as e:
                    rank0_print(f"DEBUG_LOG: ERROR in load_video_track_segment - Exception when accessing frame {frame_idx} (sample {i}) for video {video_id}: {e}. Skipping this frame sample.")
                    pass
            
            if frames_data:
                frames_data = torch.stack(frames_data)
                frames_pts = torch.tensor(frames_pts)
                frames_duration = torch.tensor(frames_duration)
            else:
                rank0_print(f"DEBUG_LOG: ERROR in load_video_track_segment - No frames could be sampled for {video_id} (all frames in indices failed or indices out of bound). Creating dummy FrameBatch.")
                dummy_data = torch.zeros((1, 3, 224, 224), dtype=torch.uint8)
                dummy_pts = torch.tensor([0.0], dtype=torch.float64)
                dummy_duration = torch.tensor([0.033], dtype=torch.float64)  # ~30fps
                return FrameBatch(dummy_data, dummy_pts, dummy_duration)

        frames = FrameBatch(frames_data, frames_pts, frames_duration)
        rank0_print(f"DEBUG_LOG: load_video_track_segment - FrameBatch created. Num frames: {len(frames)}, Shape of data: {frames.data.shape if frames else 'N/A'}")

    except RuntimeError as re:
        rank0_print(f"DEBUG_LOG: RuntimeError in load_video_track_segment for video {video_id} (likely pyav/ffmpeg issue): {re}")
        # Create a dummy FrameBatch
        dummy_data = torch.zeros((num_frames if num_frames > 0 else 1, 3, 224, 224), dtype=torch.uint8) # try to match num_frames
        dummy_pts = torch.zeros(num_frames if num_frames > 0 else 1, dtype=torch.float64)
        dummy_duration = torch.full((num_frames if num_frames > 0 else 1,), 0.033, dtype=torch.float64)  # ~30fps
        frames = FrameBatch(dummy_data, dummy_pts, dummy_duration)
        rank0_print("DEBUG_LOG: load_video_track_segment - Created dummy FrameBatch due to RuntimeError.")
    except Exception as e:
        rank0_print(f"DEBUG_LOG: UNEXPECTED_ERROR in load_video_track_segment for video {video_id}: {e}. Traceback: {traceback.format_exc()}")
        # Create a dummy FrameBatch
        dummy_data = torch.zeros((num_frames if num_frames > 0 else 1, 3, 224, 224), dtype=torch.uint8)
        dummy_pts = torch.zeros(num_frames if num_frames > 0 else 1, dtype=torch.float64)
        dummy_duration = torch.full((num_frames if num_frames > 0 else 1,), 0.033, dtype=torch.float64)  # ~30fps
        frames = FrameBatch(dummy_data, dummy_pts, dummy_duration)
        rank0_print("DEBUG_LOG: load_video_track_segment - Created dummy FrameBatch due to UNEXPECTED_ERROR.")

    rank0_print(f"DEBUG_LOG: load_video_track_segment finished for video_id: {video_id}. Returning FrameBatch with {len(frames) if frames else 0} frames.")
    return frames
