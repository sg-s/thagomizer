import os
from pathlib import Path

import pytest
from thagomizer import video

VIDEOS_DIR = Path("tests/fixtures/videos/")

VIDEO_FILES = video.find_video_files(VIDEOS_DIR)
VIDEO_FILES = [file for file in VIDEO_FILES if not file.endswith(".webm")]


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
def test_transcode_for_streaming(video_file: str):
    video_path = Path(video_file)
    output_file = video_path.with_suffix(".webm")

    video.transcode_for_streaming(video_file, output_file=output_file)

    # Ensure output file exists and is not empty
    assert output_file.exists()
    assert output_file.stat().st_size > 0
