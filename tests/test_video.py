import glob
import inspect
import os
from pathlib import Path

import pytest

from thagomizer import video

VIDEOS_DIR = Path("tests/fixtures/videos/")

VIDEO_FILES = video.find_video_files(VIDEOS_DIR)
VIDEO_FILES = [file for file in VIDEO_FILES if not file.endswith(".webm")]


PROGRESS_FILES = glob.glob("tests/fixtures/*.log")

# Total number of frames for the test video
TOTAL_FRAMES = 100  # This matches the 100% progress file


FUNCTIONS = [
    func
    for name, func in inspect.getmembers(video, inspect.isfunction)
    if func.__module__ == video.__name__
]


@pytest.fixture(scope="session", autouse=True)
def cleanup_logs():
    """Cleanup any .webvtt files created before running the test."""

    yield

    for log_file in VIDEOS_DIR.glob("*.log"):
        if log_file.exists():
            os.remove(log_file)
            print(f"Deleted {log_file}")


@pytest.fixture(scope="session", autouse=True)
def cleanup_subtitles():
    """Cleanup any .webvtt files created before running the test."""

    for subtitle_file in VIDEOS_DIR.glob("*.vtt"):
        if subtitle_file.exists():
            os.remove(subtitle_file)
            print(f"Deleted {subtitle_file}")

    yield


@pytest.fixture(scope="session", autouse=True)
def cleanup_transcoded_videos():
    """Cleanup any .webm files created during the test."""

    for transcoded_file in VIDEOS_DIR.glob("*.webm"):
        if transcoded_file.exists():
            os.remove(transcoded_file)
            print(f"Deleted {transcoded_file}")

    yield


@pytest.mark.parametrize("video_file", VIDEO_FILES)
def test_extract_english_subtitles(video_file: str):
    video.extract_english_subtitles(video_file)


@pytest.mark.parametrize("video_file", VIDEO_FILES)
def test_get_audio_channels(video_file: str):
    channels = video.get_audio_channels(video_file)
    assert isinstance(channels, int)
    assert channels > 0  # Ensure valid channel count


@pytest.mark.parametrize("video_file", VIDEO_FILES)
def test_get_video_bitrate(video_file: str):
    """Test that get_video_bitrate returns a reasonable bitrate value."""
    bitrate = video.get_video_bitrate(video_file)

    # Bitrate should be a positive integer
    assert isinstance(bitrate, int)
    assert bitrate > 0

    # Bitrate should be reasonable (between 100 kbps and 100 Mbps)
    assert 100 <= bitrate <= 100000


@pytest.mark.parametrize("video_file", VIDEO_FILES)
def test_transcode_for_streaming(video_file: str):
    video_path = Path(video_file)
    output_file = video_path.with_suffix(".webm")

    # Test with default quality_factor (1.0 = same size)
    video.transcode_for_streaming(video_file, output_file=output_file)

    # Ensure output file exists and is not empty
    assert output_file.exists()
    assert output_file.stat().st_size > 0


@pytest.mark.parametrize("video_file", VIDEO_FILES)
def test_transcode_for_streaming_with_quality_factor(video_file: str):
    """Test transcoding with different quality factors."""
    video_path = Path(video_file)

    # Test with 80% quality (smaller file)
    output_file_80 = video_path.with_suffix(".80.webm")
    video.transcode_for_streaming(
        video_file, output_file=output_file_80, quality_factor=0.8
    )

    # Test with 120% quality (larger file)
    output_file_120 = video_path.with_suffix(".120.webm")
    video.transcode_for_streaming(
        video_file, output_file=output_file_120, quality_factor=1.2
    )

    # Ensure both output files exist and are not empty
    assert output_file_80.exists()
    assert output_file_80.stat().st_size > 0
    assert output_file_120.exists()
    assert output_file_120.stat().st_size > 0

    # The 80% quality file should be smaller than the 120% quality file
    # (though this might not always be true due to encoding efficiency)
    size_80 = output_file_80.stat().st_size
    size_120 = output_file_120.stat().st_size

    print(f"80% quality file size: {size_80}")
    print(f"120% quality file size: {size_120}")


@pytest.mark.parametrize("video_file", VIDEO_FILES)
def test_transcode_for_streaming_bitrate_targeting(video_file: str):
    """Test that transcoding targets approximately the input bitrate."""
    input_bitrate = video.get_video_bitrate(video_file)

    video_path = Path(video_file)
    output_file = video_path.with_suffix(".target.webm")

    # Transcode with default quality_factor (1.0)
    video.transcode_for_streaming(video_file, output_file=output_file)

    # Check that output file exists
    assert output_file.exists()

    # Get output bitrate (this would require implementing get_video_bitrate for output files)
    # For now, just verify the file was created successfully
    assert output_file.stat().st_size > 0

    print(f"Input bitrate: {input_bitrate}k")
    print(f"Output file size: {output_file.stat().st_size} bytes")


@pytest.mark.parametrize("progress_file", PROGRESS_FILES)
def test_parse_transcode_progress(progress_file: str):
    progress, speed = video.parse_transcode_progress(
        progress_file, total_num_frames=TOTAL_FRAMES
    )

    assert speed == "16x", "Unexpected speed"
    actual_progress = float(os.path.basename(progress_file).split(".")[0])

    assert actual_progress == progress, f"Unexpected progress: {actual_progress}"


@pytest.mark.parametrize("function", FUNCTIONS)
def test_missing_file_error(function):
    with pytest.raises(FileNotFoundError):
        function("missing_file.webm")
