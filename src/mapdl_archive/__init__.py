"""MAPDL archive reader."""

from mapdl_archive import examples
from mapdl_archive.archive import (
    Archive,
    save_as_archive,
    write_cmblock,
    write_nblock,
)

# setuptools-scm writes _version.py at build time; fall back to the installed
# metadata, which is what a source checkout without a build has.
try:
    from mapdl_archive._version import version as __version__
except ImportError:  # pragma: no cover
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("mapdl-archive")
    except PackageNotFoundError:
        __version__ = "unknown"


__all__ = ["Archive", "save_as_archive", "write_cmblock", "write_nblock", "examples", "__version__"]
