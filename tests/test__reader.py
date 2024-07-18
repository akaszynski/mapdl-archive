"""Test the C++ _reader wrapper."""

from _pytest._py.path import LocalPath

import numpy as np
import numpy.typing as npt
import pytest

from mapdl_archive import _reader

ETBLOCK_STR = """ETBLOCK,        1,        1
(2i9,19a9)
        1      181        0        0        2        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0
       -1
"""

EBLOCK_STR = """EBLOCK,19,SOLID,      4644,         4
(19i8)
       1       4       1       1       0       0       0       0      20       0    2170    1071    1148     668     635    4986    6006     845     638
    1142    1147     667    1070    6022    6021     844    4971    4985    5991     851     637
       1       4       1       1       0       0       0       0      20       0    4488    5692    5649    5697    5697   13153   13148   13154   13154
    5609    5630    5697    5629   13157   13156   13154   13155   13676   13674   13677   13677
       1       4       1       1       0       0       0       0      20       0    4643     941     939     919     921   13984   13984   13984   13984
     940     934     920     935   13984   13984   13984   13984   14040   14003   14371   14038
       1       4       1       1       0       0       0       0      20       0    4644   13983     921     919     919   13984   13984   13984   13984
   14000     920   13984   13998   13984   13984   13984   13984   14004   14038   14371   14371
      -1
"""


EBLOCK_STR_NOT_SOLID = """EBLOCK,19,XYZ,      2170,         1
(19i8)
       1       4       1       1       0       0       0       0      20       0    2170    1071    1148     668     635    4986    6006     845     638
    1142    1147     667    1070    6022    6021     844    4971    4985    5991     851     637
"""

# full integration for SOLID185
KEYOPT_STR = """KEYOPT, 1, 1, 0
KEYOPT, 1, 2, 0"""


def test_read_et(tmpdir: LocalPath) -> None:
    et_line = "ET, 4, 186\n"

    filename = str(tmpdir.join("tmp.cdb"))  # type: ignore
    with open(filename, "w") as fid:
        fid.write(et_line)

    archive = _reader.Archive(filename)
    archive.read_et_line()
    ekey = archive.get_element_types()
    assert np.array_equal([[4, 186]], ekey)


def test_read_etblock(tmpdir: LocalPath) -> None:
    filename = str(tmpdir.join("tmp.cdb"))  # type: ignore
    with open(filename, "w") as fid:
        fid.write(ETBLOCK_STR)

    archive = _reader.Archive(filename)
    archive.read_etblock()
    ekey = archive.get_element_types()
    assert np.array_equal([[1, 181]], ekey)


def test_read_eblock_not_solid(tmpdir: LocalPath) -> None:
    filename = str(tmpdir.join("tmp.cdb"))  # type: ignore
    with open(filename, "w") as fid:
        fid.write(EBLOCK_STR_NOT_SOLID)

    archive = _reader.Archive(filename)
    archive.read_eblock()
    assert archive.n_elem == 0


def test_read_eblock(tmpdir: LocalPath) -> None:
    filename = str(tmpdir.join("tmp.cdb"))  # type: ignore
    with open(filename, "w") as fid:
        fid.write(EBLOCK_STR)

    archive = _reader.Archive(filename)
    archive.read_eblock()
    assert archive.n_elem == 4

    # extract numbers from elem array
    elem_txt = EBLOCK_STR[EBLOCK_STR.find(")") + 1 : -3].strip()
    elem_expected = np.array(elem_txt.split(), dtype=int)

    # not reading number of nodes (field 9) and field 10 is unused
    elem_expected = np.delete(elem_expected, list(range(8, 120, 31)) + list(range(9, 120, 31)))
    # insert 0 right after field 11
    elem_expected = np.insert(elem_expected, range(9, 120, 29), [0, 0, 0, 0])
    assert np.allclose(archive.elem, elem_expected)
    assert np.allclose(archive.elem_off, np.arange(0, archive.elem.size + 1, 30))


def test_read_keyopt(tmpdir: LocalPath) -> None:
    filename = str(tmpdir.join("tmp.cdb"))  # type: ignore
    with open(filename, "w") as fid:
        fid.write(KEYOPT_STR)

    archive = _reader.Archive(filename)
    archive.read_keyopt_line()
    archive.read_keyopt_line()
    assert archive.keyopt[1][0] == [1, 0]
    assert archive.keyopt[1][1] == [2, 0]
