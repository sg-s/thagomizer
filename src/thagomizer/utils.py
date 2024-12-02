"""general utilities"""

from pathlib import Path

from beartype import beartype


@beartype
def hash_file(file_name: str | Path) -> str:
    """hash a file, reading in chunks using xxhash

    Args:
        file_name (str): name of the file

    Returns:
        str: hash of file
    """

    import xxhash

    BUF_SIZE = 65536

    x = xxhash.xxh32()

    with open(file_name, "rb") as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            x.update(data)

    return x.hexdigest()
