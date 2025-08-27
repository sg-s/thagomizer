"""Tests for English subtitle stream selection logic."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from thagomizer.video import get_english_subtitle_stream


def _ffprobe_result(streams: list[dict]) -> SimpleNamespace:
    """Helper to fake subprocess.run result with JSON stdout."""

    return SimpleNamespace(
        stdout=json.dumps({"streams": streams}), stderr="", returncode=0
    )


def test_selects_english_stream_with_max_frames(monkeypatch, tmp_path: Path):
    """Ensure the function picks the English stream with the highest NUMBER_OF_FRAMES."""

    fake_file = tmp_path / "video.mkv"
    fake_file.write_bytes(b"dummy")

    streams = [
        {
            "index": 2,
            "tags": {
                "language": "eng",
                "title": "English [Forced]",
                "NUMBER_OF_FRAMES": "2",
            },
        },
        {
            "index": 3,
            "codec_name": "subrip",
            "tags": {
                "language": "eng",
                "title": "English",
                "NUMBER_OF_FRAMES": "529",
            },
        },
        {
            "index": 7,
            "tags": {"language": "spa", "NUMBER_OF_FRAMES": "700"},
        },
    ]

    def fake_run(cmd, check, stdout, stderr, text):  # noqa: ARG001
        return _ffprobe_result(streams)

    monkeypatch.setattr("subprocess.run", fake_run)

    selected = get_english_subtitle_stream(str(fake_file))
    assert selected == "3"
