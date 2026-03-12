"""Video assembly - encodes animation frames into MP4 using FFmpeg."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import logging
from pathlib import Path
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _find_ffmpeg() -> str:
    """Find the FFmpeg binary, preferring imageio_ffmpeg's bundled version."""
    # Try imageio_ffmpeg first (bundled with moviepy)
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe:
            return exe
    except ImportError:
        pass

    # Fall back to system FFmpeg
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    raise RuntimeError(
        "FFmpeg not found. Install with: brew install ffmpeg (macOS) "
        "or pip install imageio-ffmpeg"
    )


def frames_to_video(
    frames: list[Image.Image],
    output_path: str | Path,
    fps: int = 24,
    codec: str = "libx264",
    bitrate: str = "5M",
) -> Path:
    """Encode a list of PIL Images into an MP4 video using FFmpeg.

    Args:
        frames: List of PIL Images (all same size, RGB).
        output_path: Path for the output MP4 file.
        fps: Frames per second.
        codec: FFmpeg video codec.
        bitrate: Target bitrate.

    Returns:
        Path to the generated video file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not frames:
        raise ValueError("No frames to encode")

    ffmpeg_exe = _find_ffmpeg()
    logger.info(f"Using FFmpeg: {ffmpeg_exe}")

    w, h = frames[0].size

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Write frames as numbered PNGs
        logger.info(f"Writing {len(frames)} frames to temp directory...")
        for i, frame in enumerate(tqdm(frames, desc="Writing frames", unit="frame")):
            frame_path = tmp_path / f"frame_{i:06d}.png"
            frame.save(frame_path)

        # Encode with FFmpeg
        input_pattern = str(tmp_path / "frame_%06d.png")
        cmd = [
            ffmpeg_exe,
            "-y",  # Overwrite output
            "-framerate", str(fps),
            "-i", input_pattern,
            "-c:v", codec,
            "-b:v", bitrate,
            "-pix_fmt", "yuv420p",  # Compatibility
            "-vf", f"scale={w}:{h}",
            "-movflags", "+faststart",  # Web-friendly
            str(output_path),
        ]

        logger.info(f"Encoding video: {w}x{h} @ {fps}fps, codec={codec}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            raise RuntimeError(f"FFmpeg encoding failed: {result.stderr[:500]}")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Video saved: {output_path} ({file_size_mb:.1f} MB)")
    return output_path
