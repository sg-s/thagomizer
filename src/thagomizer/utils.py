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


@beartype
def sanitize(text: str) -> str:
    """
    sanitizes a text by doing the following:
    - Converts non-ASCII characters to the nearest ASCII equivalent.
    - Replaces spaces with `--`.
    - Removes characters that are not alphanumeric, `--`, or `_`.
    - converts to lower case

    Args:
        text (str): Input text to convert.

    Returns:
        str: sanitized text text.
    """

    import re
    import unicodedata

    normalized_text = unicodedata.normalize("NFD", text)
    ascii_text = (
        "".join(char for char in normalized_text if unicodedata.category(char) != "Mn")
        .encode("ascii", "ignore")
        .decode("ascii")
    )

    # Replace spaces with `--`
    ascii_text = ascii_text.replace(" ", "--")

    # Remove characters that are not alphanumeric, `--`, or `_`
    sanitized_text = re.sub(r"[^a-zA-Z0-9\-_]", "", ascii_text)

    return sanitized_text.lower()


def format_bytes(size_bytes: int) -> str:
    """converts bytes into MB, GB, etc. for a human-readable format

    from here:

    https://stackoverflow.com/questions/5194057/better-way-to-convert-file-sizes-in-python
    """

    import math

    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])
