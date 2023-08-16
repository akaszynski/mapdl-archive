"""Installation file for mapdl_archive."""
from io import open as io_open
import os

import numpy as np
from setuptools import Extension, setup

DEBUG = False
if DEBUG:
    extra_compile_args = ["-O0", "-g"]
else:
    extra_compile_args = ["/O2", "/w"] if os.name == "nt" else ["-O3", "-w"]


# Get version from version info
__version__ = None
this_file = os.path.dirname(__file__)
version_file = os.path.join(this_file, "mapdl_archive", "_version.py")
with io_open(version_file, mode="r") as fd:
    exec(fd.read())


setup(
    name="mapdl-archive",
    packages=["mapdl_archive", "mapdl_archive.examples"],
    version=__version__,
    description="Pythonic interface to MAPDL archive files.",
    long_description=open("README.rst").read(),
    long_description_content_type="text/x-rst",
    author="Alex Kaszynski",
    author_email="akascap@gmail.com",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    url="https://github.com/akaszynski/mapdl-archive",
    # Build cython modules
    include_dirs=[np.get_include()],
    ext_modules=[
        Extension(
            "mapdl_archive._relaxmidside",
            ["mapdl_archive/cython/_relaxmidside.pyx"],
            extra_compile_args=extra_compile_args,
            language="c",
        ),
        Extension(
            "mapdl_archive._archive",
            [
                "mapdl_archive/cython/_archive.pyx",
                "mapdl_archive/cython/archive.c",
            ],
            extra_compile_args=extra_compile_args,
            language="c",
        ),
        Extension(
            "mapdl_archive._reader",
            [
                "mapdl_archive/cython/_reader.pyx",
                "mapdl_archive/cython/reader.c",
                "mapdl_archive/cython/vtk_support.c",
            ],
            extra_compile_args=extra_compile_args,
            language="c",
        ),
    ],
    python_requires=">=3.8",
    keywords="vtk MAPDL ANSYS cdb",
    package_data={
        "mapdl_archive.examples": [
            "TetBeam.cdb",
            "HexBeam.cdb",
            "sector.cdb",
        ]
    },
    install_requires=["pyvista>=0.41.1"],
)
