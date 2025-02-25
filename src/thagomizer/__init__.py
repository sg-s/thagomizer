"""thagomizer: useful tools in python."""

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # for Python < 3.8
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("thagomizer")
except PackageNotFoundError:
    __version__ = "0.2.0"
