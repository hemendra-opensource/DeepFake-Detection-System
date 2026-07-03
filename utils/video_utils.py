"""
utils/video_utils.py
====================
Video processing helpers.

Provides:
- Frame extraction from video files
- Frame sampling at a configurable rate
- Video metadata (duration, fps, resolution)
- Video validation (format, readability)
- Frame iterator for memory-efficient processing
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np

from utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

ImageArray = np.ndarray


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class VideoMetadata:
    """Metadata extracted from a video file."""

    path: str
    fps: float
    total_frames: int
    duration_seconds: float
    width: int
    height: int
    codec: str
    is_valid: bool


@dataclass
class FrameResult:
    """A single extracted frame with its metadata."""

    frame_index: int          # Zero-based original frame index
    sample_index: int         # Zero-based sample index (after sub-sampling)
    timestamp_ms: float       # Millisecond position in video
    image: ImageArray         # RGB image array


# ── Metadata ──────────────────────────────────────────────────────────────────

def get_video_metadata(path: str | Path) -> VideoMetadata:
    """
    Extract metadata from a video file without decoding all frames.

    Args:
        path: Path to the video file.

    Returns:
        :class:`VideoMetadata` dataclass instance.
    """
    p = str(path)
    cap = cv2.VideoCapture(p)

    is_valid = cap.isOpened()
    if not is_valid:
        logger.warning("Cannot open video: %s", p)
        return VideoMetadata(
            path=p, fps=0.0, total_frames=0, duration_seconds=0.0,
            width=0, height=0, codec="", is_valid=False,
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0.0

    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)).strip()

    cap.release()

    return VideoMetadata(
        path=p,
        fps=fps,
        total_frames=total_frames,
        duration_seconds=duration,
        width=width,
        height=height,
        codec=codec,
        is_valid=True,
    )


# ── Frame extraction ──────────────────────────────────────────────────────────

def extract_frames(
    path: str | Path,
    sample_rate: int = 5,
    max_frames: int = 200,
    start_frame: int = 0,
) -> List[FrameResult]:
    """
    Extract frames from a video at a given sample rate.

    Args:
        path:        Path to the video file.
        sample_rate: Extract every Nth frame (e.g., 5 → every 5th frame).
        max_frames:  Maximum number of frames to return.
        start_frame: Frame index to begin reading from.

    Returns:
        List of :class:`FrameResult` objects in temporal order.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        logger.error("Cannot open video for frame extraction: %s", path)
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames: List[FrameResult] = []
    frame_idx = 0
    sample_idx = 0

    # Seek to start frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame

    try:
        while len(frames) < max_frames:
            ret, bgr_frame = cap.read()
            if not ret:
                break

            if (frame_idx - start_frame) % sample_rate == 0:
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                timestamp_ms = (frame_idx / fps) * 1000
                frames.append(
                    FrameResult(
                        frame_index=frame_idx,
                        sample_index=sample_idx,
                        timestamp_ms=timestamp_ms,
                        image=rgb_frame,
                    )
                )
                sample_idx += 1

            frame_idx += 1
    except Exception as exc:
        logger.error("Error extracting frames from %s: %s", path, exc)
    finally:
        cap.release()

    logger.info(
        "Extracted %d frames from %s (sample_rate=%d, max=%d)",
        len(frames), path, sample_rate, max_frames,
    )
    return frames


def frame_iterator(
    path: str | Path,
    sample_rate: int = 5,
    max_frames: int = 200,
) -> Generator[FrameResult, None, None]:
    """
    Memory-efficient generator that yields one frame at a time.

    Prefer over :func:`extract_frames` when processing long videos
    where loading all frames at once would exhaust memory.

    Args:
        path:        Video file path.
        sample_rate: Extract every Nth frame.
        max_frames:  Maximum total frames to yield.

    Yields:
        :class:`FrameResult` instances.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = 0
    sample_idx = 0
    yielded = 0

    try:
        while yielded < max_frames:
            ret, bgr_frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                timestamp_ms = (frame_idx / fps) * 1000
                yield FrameResult(
                    frame_index=frame_idx,
                    sample_index=sample_idx,
                    timestamp_ms=timestamp_ms,
                    image=rgb_frame,
                )
                sample_idx += 1
                yielded += 1

            frame_idx += 1
    finally:
        cap.release()


# ── Validation ────────────────────────────────────────────────────────────────

def is_valid_video(path: str | Path) -> bool:
    """
    Check whether a video file can be opened and has at least one readable frame.

    Args:
        path: Path to the video file.

    Returns:
        ``True`` if the video is readable.
    """
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return False
        ret, _ = cap.read()
        cap.release()
        return ret
    except Exception as exc:
        logger.debug("Video validation error for %s: %s", path, exc)
        return False


# ── Thumbnail ─────────────────────────────────────────────────────────────────

def get_video_thumbnail(
    path: str | Path,
    size: Tuple[int, int] = (320, 180),
) -> Optional[ImageArray]:
    """
    Extract a single thumbnail from the middle of a video.

    Args:
        path: Video file path.
        size: ``(width, height)`` of the returned thumbnail.

    Returns:
        RGB thumbnail array, or ``None`` if extraction fails.
    """
    meta = get_video_metadata(path)
    if not meta.is_valid or meta.total_frames == 0:
        return None

    mid_frame = meta.total_frames // 2
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
