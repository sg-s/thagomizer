"""module to test functions in utils"""

from pathlib import Path

from thagomizer.utils import hash_file, sanitize


def test_hash_file():
    path = Path(__file__).parent.parent / "src" / "thagomizer" / "utils.py"
    assert hash_file(path) == "419cf5b9"


def test_sanitize():
    txt = "Büoo wow"
    assert "buoo--wow" == sanitize(txt)
