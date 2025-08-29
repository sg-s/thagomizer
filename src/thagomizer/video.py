"""module to work with video files using ffmpeg"""

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from beartype import beartype

from thagomizer.utils import hash_file

FFMPEG_LOC = shutil.which("ffmpeg")
FFPROBE_LOC = shutil.which("ffprobe")

VIDEO_FORMATS = ["mkv", "avi", "mp4", "mov", "flv", "webm"]

if not FFMPEG_LOC:
    raise FileNotFoundError("Cannot find the ffmpeg binary.")

if not FFPROBE_LOC:
    raise FileNotFoundError("Cannot find the ffprobe binary.")


@beartype
def get_english_subtitle_stream(input_file: str | Path) -> str | None:
    """Uses ffprobe to check if an English subtitle stream exists.

    Args:
        input_file (str): Path to the video file.

    Returns:
        str: Stream index of the English subtitle, or None if not found.

    Raises:
        RuntimeError: If ffprobe fails.
    """

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Progress file {input_file} does not exist.")

    cmd_ffprobe = [
        FFPROBE_LOC,
        "-v",
        "error",
        "-show_streams",
        "-select_streams",
        "s",
        "-of",
        "json",
        input_file,
    ]

    try:
        result = subprocess.run(
            cmd_ffprobe,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        streams = json.loads(result.stdout).get("streams", [])

        # Choose the English subtitle stream with the maximum NUMBER_OF_FRAMES.
        # Some files contain multiple English subtitle tracks (e.g., forced and full).
        # We prefer the one with the most frames as it generally contains the most content.
        best_stream_index: Optional[int] = None
        best_num_frames: int = -1

        for stream in streams:
            tags = stream.get("tags", {})
            if tags.get("language") != "eng":
                continue

            num_frames = int(tags.get("NUMBER_OF_FRAMES", -1))

            if num_frames > best_num_frames:
                best_num_frames = num_frames
                best_stream_index = int(stream.get("index"))

        if best_stream_index is not None:
            return str(best_stream_index)

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed: {e.stderr.strip()}") from None

    return None  # No English subtitles found


@beartype
def extract_english_subtitles(
    input_file: str | Path,
    output_file: Optional[str] = None,
) -> None | Path:
    """Extracts English subtitles from a video and saves them in WebVTT format.

    Args:
        input_file (str): Path to the input video file.
        output_file (str, optional): Path to save the extracted subtitles. If not provided,
                                     appends ".vtt" to the input filename.

    Raises:
        RuntimeError: If ffmpeg fails to extract subtitles.
    """

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Progress file {input_file} does not exist.")

    if isinstance(input_file, str):
        input_file = Path(input_file)

    # Check for English subtitles using ffprobe
    subtitle_stream = get_english_subtitle_stream(input_file)
    if subtitle_stream is None:
        print("No English subtitles found in the input file.")
        return None

    # Default to .vtt output file if not provided
    if output_file is None:
        output_file = input_file.with_suffix(".vtt")

    if output_file.exists():
        return output_file

    # Extract subtitles using ffmpeg in WebVTT format
    cmd_ffmpeg = [
        FFMPEG_LOC,
        "-i",
        input_file,
        "-map",
        f"0:{subtitle_stream}",
        "-c:s",
        "webvtt",
        str(output_file),
    ]

    try:
        subprocess.run(
            cmd_ffmpeg,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    except subprocess.CalledProcessError as e:
        print("FFmpeg error:", e.stderr.strip())
        raise RuntimeError(
            f"ffmpeg failed to extract English subtitles: {e.stderr.strip()}"
        ) from None

    # Verify the output file exists and is not empty
    if not os.path.exists(output_file):
        raise FileNotFoundError(f"Subtitle file {output_file} was not created.")

    if os.path.getsize(output_file) == 0:
        raise RuntimeError(f"Subtitle file {output_file} is empty.")

    print(f"English subtitles saved to: {output_file}")
    return output_file


@beartype
def extract_sample(
    input_file: str,
    output_file: Optional[str] = None,
    *,
    sample_length: int = 30,
) -> str:
    """Extracts a sample from the middle of a video file.

    Args:
        input_file (str): Path to the input video file.
        output_file (str): Path to save the extracted sample.

    Raises:

        RuntimeError: If ffmpeg fails to process the video.
    """

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Progress file {input_file} does not exist.")

    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}-sample{ext}"

    # Get the video duration using ffprobe
    cmd_probe = [
        FFPROBE_LOC,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        input_file,
    ]

    try:
        duration = float(subprocess.check_output(cmd_probe, text=True).strip())
    except subprocess.CalledProcessError:
        raise RuntimeError("Failed to retrieve video duration using ffprobe.") from None

    # Calculate the midpoint start time
    start_time = max(
        0, (duration / 2) - sample_length / 2
    )  # Ensure non-negative start time

    # Extract the 30-second clip
    cmd_ffmpeg = [
        FFMPEG_LOC,
        "-y",
        "-i",
        input_file,
        "-ss",
        str(start_time),
        "-t",
        str(sample_length),
        "-c",
        "copy",
        output_file,
    ]

    try:
        subprocess.run(
            cmd_ffmpeg, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError:
        raise RuntimeError("ffmpeg failed to extract the video sample.") from None

    print(f"Sample saved to: {output_file}")
    return output_file


@beartype
def hash_video_file(input_file: str) -> str:
    """fast hash of a video file using ffprobe"""

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Progress file {input_file} does not exist.")

    command = [
        FFPROBE_LOC,
        "-i",
        input_file,
        "-show_entries",
        "format=duration",
        "-v",
        "quiet",
        "-of",
        "csv=p=0",
    ]

    try:
        output = subprocess.check_output(
            command,
            stderr=subprocess.STDOUT,
            text=True,
        )
        duration = float(output.strip())

    except subprocess.CalledProcessError as e:
        print(f"Error: {e.output}")

    command = [
        FFMPEG_LOC,
        "-ss",
        str(int(duration / 2)),
        "-i",
        input_file,
        "-frames:v",
        "1",  # Extract 1 frame
        "-q:v",
        "0",
        "middle_frame.jpg",  # Output file
    ]

    # Run the command
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.output}")

    frame_hash = hash_file("middle_frame.jpg")
    os.remove("middle_frame.jpg")

    # Define the command as a list of arguments
    command = [
        FFPROBE_LOC,
        "-hide_banner",
        input_file,
    ]

    # Run the command
    try:
        ffprobe_output = subprocess.check_output(
            command, stderr=subprocess.STDOUT, text=True
        )

    except subprocess.CalledProcessError as e:
        print(f"Error: {e.output}")

    text = ffprobe_output + frame_hash
    text_bytes = text.encode("utf-8")

    hash_object = hashlib.sha256(text_bytes)

    return hash_object.hexdigest()


@beartype
def get_audio_channels(input_file: str | Path) -> int:
    """
    Uses ffprobe to determine the number of audio channels in a video file.

    Args:
        input_file (str): Path to the input video file.

    Returns:
        int: Number of audio channels.
    """

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist.")

    command = [
        FFPROBE_LOC,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=channels",
        "-of",
        "csv=p=0",
        input_file,
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return (
            int(result.stdout.strip()) if result.stdout.strip().isdigit() else 2
        )  # Default to stereo if unknown
    except Exception as e:
        print(f"Error probing audio channels: {e}")
        return 2  # Fallback to stereo


@beartype
def get_video_bitrate(input_file: str | Path) -> int:
    """
    Gets the video bitrate from a video file using ffprobe.

    Args:
        input_file (str | Path): Path to the input video file.

    Returns:
        int: Video bitrate in kbps.

    Raises:
        FileNotFoundError: If the input file doesn't exist.
        RuntimeError: If ffprobe fails or bitrate cannot be determined.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist.")

    cmd_ffprobe = [
        FFPROBE_LOC,
        "-v",
        "error",
        "-show_entries",
        "format=bit_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        input_file,
    ]

    try:
        result = subprocess.run(
            cmd_ffprobe,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        bitrate_str = result.stdout.strip()
        if bitrate_str and bitrate_str.isdigit():
            # Convert from bps to kbps
            return int(int(bitrate_str) / 1000)
        else:
            # Fallback: try to get bitrate from video stream
            cmd_ffprobe_stream = [
                FFPROBE_LOC,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=bit_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                input_file,
            ]

            result_stream = subprocess.run(
                cmd_ffprobe_stream,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            bitrate_str = result_stream.stdout.strip()
            if bitrate_str and bitrate_str.isdigit():
                return int(int(bitrate_str) / 1000)
            else:
                # Final fallback: estimate from file size and duration
                return _estimate_bitrate_from_file_size(input_file)

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed: {e.stderr.strip()}") from None


@beartype
def _estimate_bitrate_from_file_size(input_file: str | Path) -> int:
    """
    Estimates video bitrate from file size and duration as a fallback.

    Args:
        input_file (str | Path): Path to the input video file.

    Returns:
        int: Estimated video bitrate in kbps.

    Raises:
        FileNotFoundError: If the input file doesn't exist.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist.")

    # Get duration
    cmd_duration = [
        FFPROBE_LOC,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        input_file,
    ]

    try:
        result = subprocess.run(
            cmd_duration,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        duration = float(result.stdout.strip())
        file_size = Path(input_file).stat().st_size

        # Estimate bitrate: (file_size * 8) / (duration * 1000)
        # Convert to kbps
        estimated_bitrate = int((file_size * 8) / (duration * 1000))

        # Cap at reasonable maximum (50 Mbps)
        return min(estimated_bitrate, 50000)

    except (subprocess.CalledProcessError, ValueError):
        # Final fallback: return a reasonable default
        return 10000  # 10 Mbps


@beartype
def transcode_for_streaming(
    input_file: str | Path,
    output_file: Optional[str | Path] = None,
    *,
    quality_factor: float = 1.0,
) -> Path:
    """
    Transcodes a video file using AV1 (libsvtav1) and Opus audio in WebM format,
    targeting approximately the same bitrate as the input file for quality preservation.
    Uses two-pass encoding for optimal quality and bitrate control.

    Args:
        input_file (str | Path): Path to the input video file.
        output_file (Optional[str | Path]): Path to the output file. If None, appends '.webm' to input filename.
        quality_factor (float): Multiplier for target bitrate. 1.0 = same size, 0.8 = 80% size, etc.

    Raises:
        RuntimeError: If transcoding fails.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist.")

    input_path = Path(input_file)
    if output_file is None:
        output_file = input_path.with_suffix(".webm")

    if output_file.exists():
        return output_file

    temp_file = input_path.with_suffix(".tmp.webm")
    progress_log = input_path.with_suffix(".webm.progress.log")

    # Get input bitrate and calculate target bitrate
    input_bitrate = get_video_bitrate(input_file)
    target_bitrate = int(input_bitrate * quality_factor)

    # For AV1, we'll use CRF mode but target the desired bitrate
    # Start with a conservative CRF value and adjust if needed
    crf = 18  # High quality (lower values = higher quality)

    audio_channels = get_audio_channels(input_file)
    audio_filter = "channelmap=channel_layout=5.1" if audio_channels >= 6 else "anull"

    print(
        f"Transcoding: {input_file} -> {temp_file}"
        f"\nInput bitrate: {input_bitrate}k, Target: {target_bitrate}k"
        f"\nUsing CRF {crf} for quality control"
        f"\nAudio Channels: {audio_channels}"
    )

    # First pass: analyze video for optimal encoding
    first_pass_cmd = [
        FFMPEG_LOC,
        "-y",  # Overwrite output files
        "-i",
        input_file,
        "-c:v",
        "libsvtav1",
        "-crf",
        str(crf),
        "-pass",
        "1",
        "-an",  # No audio in first pass
        "-f",
        "null",
        "/dev/null" if os.name != "nt" else "NUL",  # Cross-platform null device
    ]

    # Second pass: encode with optimal settings
    second_pass_cmd = [
        FFMPEG_LOC,
        "-y",  # Overwrite output files
        "-progress",
        progress_log,
        "-i",
        input_file,
        "-map",
        "0",
        "-c:v",
        "libsvtav1",
        "-crf",
        str(crf),
        "-pass",
        "2",
        "-c:a",
        "libopus",
        "-b:a",
        "192k",
        "-af",
        audio_filter,
        "-c:s",
        "webvtt",
        "-f",
        "webm",
        temp_file,
    ]

    try:
        print("Starting first pass (analysis)...")
        subprocess.run(
            first_pass_cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        print("Starting second pass (encoding)...")
        subprocess.run(
            second_pass_cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Clean up pass files
        pass_files = [
            input_path.with_suffix(".webm.ffindex"),
            input_path.with_suffix(".webm-0.log"),
        ]
        for pass_file in pass_files:
            if pass_file.exists():
                pass_file.unlink()

        shutil.move(temp_file, output_file)
        print(f"Successfully created: {output_file}")

    except subprocess.CalledProcessError as e:
        print(f"Transcoding failed for {input_file}:\n{e.stderr}")
        Path(temp_file).unlink(missing_ok=True)

        # Clean up pass files on failure
        pass_files = [
            input_path.with_suffix(".webm.ffindex"),
            input_path.with_suffix(".webm-0.log"),
        ]
        for pass_file in pass_files:
            if pass_file.exists():
                pass_file.unlink()

        raise RuntimeError(f"FFmpeg transcoding failed: {e.stderr}") from e

    return output_file


@beartype
def find_video_files(dir_name: str | Path) -> list:
    """find all video files in a given directory"""

    if not os.path.exists(dir_name):
        raise FileNotFoundError(f"Directory {dir_name} does not exist.")

    if isinstance(dir_name, str):
        dir_name = Path(dir_name)

    return [str(file) for ext in VIDEO_FORMATS for file in dir_name.glob(f"*.{ext}")]


@beartype
def get_duration_seconds(input_file: str | Path) -> float:
    """get the duration of a video file in seconds

    Args:
        input_file (str): Path to the input video file."""

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist.")

    cmd = [
        FFPROBE_LOC,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        input_file,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


@beartype
def parse_transcode_progress(
    progress_file: str | Path,
    *,
    total_num_frames: Optional[int] = None,
):
    """
    parse the progress file to get the transcoding progress

    Args:
        progress_file (str): Path to the progress file
        total_duration (float): Total duration of the video in seconds. This can be obtained via get_duration_seconds
    """

    if not os.path.exists(progress_file):
        raise FileNotFoundError(f"Progress file {progress_file} does not exist.")

    if total_num_frames is None:
        progress_file = Path(progress_file)  # Ensure it's a Path object

        # Remove the last three extensions (.webm.progress.log) and append .tmp.webm
        # progress_file has format: filename.webm.progress.log
        # We want: filename.tmp.webm
        base_name = progress_file.stem  # removes .log
        base_name = Path(base_name).stem  # removes .progress
        base_name = Path(base_name).stem  # removes .webm
        input_file = progress_file.parent / f"{base_name}.tmp.webm"
        total_num_frames = get_frame_count(input_file)

    if total_num_frames is None:
        return 0, None

    kv_pattern = re.compile(r"^(\w+)=([\S]+)$")

    speed = None
    progress = None
    complete = False

    # Read lines once
    with open(progress_file, "r") as f:
        lines = f.readlines()

    # Reverse the lines so we see the bottom of the file first
    for line in reversed(lines):
        line = line.strip()
        match = kv_pattern.match(line)
        if not match:
            continue

        key, value = match.groups()

        if key == "progress":
            if value == "end":
                complete = True
                progress = 100

        elif key == "frame":
            progress = int(100 * int(value) / total_num_frames)
        elif key == "speed":
            speed = value
        else:
            continue

        if speed is not None and progress is not None:
            if complete:
                return 100, speed
            else:
                return progress, speed

    # fallback
    return 0, None


@beartype
def get_frame_count(input_file: str | Path) -> int | None:
    """count the number of frames in a video file"""

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Video file {input_file} does not exist.")

    # Run ffprobe and capture its stdout/stderr
    result = subprocess.run(
        [
            FFPROBE_LOC,
            input_file,
        ],
        capture_output=True,
        text=True,
    )

    # Search stdout for lines containing "NUMBER_OF_FRAMES"
    found_lines = [
        line for line in result.stderr.splitlines() if "NUMBER_OF_FRAMES" in line
    ]

    if found_lines:
        for line in found_lines:
            line = line.replace("NUMBER_OF_FRAMES:", "")
            line = line.strip()
            return int(line)


@beartype
def extract_keyframe(input_file: str | Path) -> str:
    """
    Extracts a keyframe from approximately the middle third of the video and saves it as an image.

    The output image is saved in the same directory as the input video and is named
    <original_video_basename>.keyframe.jpg.

    :param input_file: Path to the input video file.
    """

    input_file = Path(input_file)

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist.")

    duration = get_duration_seconds(input_file)
    seek_time = duration / 2.0

    # Build the output image path.
    output_image_path = input_file.parent / f"{input_file.stem}.keyframe.jpg"

    # Construct the FFmpeg command.
    # -ss before the input does a fast seek.
    # -skip_frame nokey makes FFmpeg only decode keyframes.
    # -vsync 0 avoids frame duplication.
    # -frames:v 1 ensures only one frame is output.
    command = [
        "ffmpeg",
        "-ss",
        str(seek_time),
        "-skip_frame",
        "nokey",
        "-i",
        str(input_file),
        "-vsync",
        "0",
        "-frames:v",
        "1",
        str(output_image_path),
    ]

    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"Thumbnail saved at {output_image_path}")

    return str(output_image_path)
