"""MAPDL archive reader."""
from mapdl_archive import examples  # noqa: F401
from mapdl_archive._version import __version__  # noqa: F401
from mapdl_archive.archive import (  # noqa: F401
    Archive,
    save_as_archive,
    write_cmblock,
    write_nblock,
)
