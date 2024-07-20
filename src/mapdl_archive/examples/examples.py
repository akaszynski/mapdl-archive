"""Mapdl-archive example files."""

import os

# get location of this folder and the example files
dir_path = os.path.dirname(os.path.realpath(__file__))
hexarchivefile = os.path.join(dir_path, "HexBeam.cdb")
tetarchivefile = os.path.join(dir_path, "TetBeam.cdb")
sector_archive_file = os.path.join(dir_path, "sector.cdb")
academic_rotor_archive_file = os.path.join(dir_path, "academic_rotor.rst")
