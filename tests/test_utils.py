"""module to test functions in utils"""

from pathlib import Path

from thagomizer.utils import hash_file


def test_hash_file():
    path = Path(__file__).parent.parent / "src" / "thagomizer" / "utils.py"
    assert hash_file(path) == "34173abd"
