"""Test the C++ _reader wrapper."""

import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from pyvista.core.celltype import CellType

from mapdl_archive import _reader

TEST_PATH = os.path.dirname(os.path.abspath(__file__))
TESTFILES_PATH = os.path.join(TEST_PATH, "test_data")


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

NBLOCK_STR = """NBLOCK,6,SOLID,       919,         9
(3i8,6e20.13)
     635       0       0 3.7826539829200E+00 1.2788958692644E+00-1.0220880953640E+00
     637       0       0 3.7987359490873E+00 1.2312085780780E+00-1.0001885444969E+00
     638       0       0 3.8138798206653E+00 1.1833200772896E+00-9.7805743587145E-01
     667       0       0 3.7751258193793E+00 1.2956563072306E+00-9.9775569295981E-01
     668       0       0 3.7675976558386E+00 1.3124167451968E+00-9.7342329055565E-01
     844       0       0 3.8071756567432E+00 1.2018089624856E+00-9.5159140433025E-01
     845       0       0 3.8004714928212E+00 1.2202978476816E+00-9.2512537278904E-01
     851       0       0 3.7840345743299E+00 1.2663572964392E+00-9.4927433167235E-01
     919       0       0 3.8682501483615E+00 1.4211343558710E+00-9.2956245308371E-01
N,R5.3,LOC,       -1,
"""

NBLOCK_NODE_ID_ARR = np.array(
    [635, 637, 638, 667, 668, 844, 845, 851, 919],
    dtype=np.int32,
)
NBLOCK_POS_ARRAY = np.array(
    [
        [3.7826539829200e00, 1.2788958692644e00, -1.0220880953640e00],
        [3.7987359490873e00, 1.2312085780780e00, -1.0001885444969e00],
        [3.8138798206653e00, 1.1833200772896e00, -9.7805743587145e-01],
        [3.7751258193793e00, 1.2956563072306e00, -9.9775569295981e-01],
        [3.7675976558386e00, 1.3124167451968e00, -9.7342329055565e-01],
        [3.8071756567432e00, 1.2018089624856e00, -9.5159140433025e-01],
        [3.8004714928212e00, 1.2202978476816e00, -9.2512537278904e-01],
        [3.7840345743299e00, 1.2663572964392e00, -9.4927433167235e-01],
        [3.8682501483615e00, 1.4211343558710e00, -9.2956245308371e-01],
    ]
)

NBLOCK_INCOMPLETE = """NBLOCK,6,SOLID,      3,      3
(3i9,6e21.13e3)
        1        0        0 1.0000000000000E-001
        2        0        0 5.0000000000000E-001 8.0000000000000E-001
        3        0        0 1.0000000000000E-001 7.0000000000000E-001 9.0000000000000E-001
N,R5.3,LOC,       -1,
"""


# The RLBLOCK command defines a real constant
# set. The real constant sets follow each set,
# starting with

# Format1 and followed by one or more Format2's, as
# needed. The command format is:
# RLBLOCK,NUMSETS,MAXSET,MAXITEMS,NPERLINE
# Format1
# Format2
#
# where:
# Format1 - Data descriptor defining the format of the
# first line. For the RLBLOCK command, this is always
# (2i8,6g16.9).  The first i8 is the set number, the
# second i8 is the number of values in this set,
# followed by up to 6 real
#
# Format2 - Data descriptors defining the format of
# the subsequent lines (as needed); this is always
# (7g16.9).
#
# - NUMSETS : The number of real constant sets defined
# - MAXSET : Maximum real constant set number
# - MAXITEMS : Maximum number of reals in any one set
# - NPERLINE : Number of reals defined on a line

RLBLOCK_STR = """RLBLOCK,       1,       2,       6,       7
(2i8,6g16.9)
(7g16.9)
       2       6  1.00000000     0.566900000E-07  0.00000000      0.00000000      0.00000000      0.00000000
"""

CMBLOCK_NODE_STR = """CMBLOCK,INTERFACE,NODE,      16  ! users node component definition
(8i10)
        25        26        27        28        29        30        31        32
        33        34        35        36        37        38        39        40
"""
NCOMP_INTERFACE = np.array([25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40])

CMBLOCK_ELEM_STR = """CMBLOCK,ELMISC ,ELEM,       2  ! users element component definition
(8i10)
        82       -90
"""
ECOMP_ELMISC = np.arange(82, 91)


def test_read_et(tmp_path: Path) -> None:
    et_line = "ET, 4, 186\n"

    filename = str(tmp_path / "tmp.cdb")
    with open(filename, "w") as fid:
        fid.write(et_line)

    archive = _reader.Archive(filename)
    archive.read_line()
    archive.read_et_line()
    assert np.array_equal([[4, 186]], archive.elem_type)


def test_read_etblock(tmp_path: Path) -> None:
    filename = str(tmp_path / "tmp.cdb")
    with open(filename, "w") as fid:
        fid.write(ETBLOCK_STR)

    archive = _reader.Archive(filename)
    archive.read_line()
    archive.read_etblock()
    assert np.array_equal([[1, 181]], archive.elem_type)


def test_read_eblock_not_solid(tmp_path: Path) -> None:
    filename = str(tmp_path / "tmp.cdb")
    with open(filename, "w") as fid:
        fid.write(EBLOCK_STR_NOT_SOLID)

    archive = _reader.Archive(filename)
    archive.read_line()
    archive.read_eblock()
    assert archive.n_elem == 0


def test_read_eblock(tmp_path: Path) -> None:
    filename = str(tmp_path / "tmp.cdb")
    with open(filename, "w") as fid:
        fid.write(EBLOCK_STR)

    archive = _reader.Archive(filename)
    archive.read_line()
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


def test_read_keyopt(tmp_path: Path) -> None:
    filename = str(tmp_path / "tmp.cdb")
    with open(filename, "w") as fid:
        fid.write(KEYOPT_STR)

    archive = _reader.Archive(filename)
    archive.read_line()
    archive.read_keyopt_line()
    archive.read_line()
    archive.read_keyopt_line()
    assert archive.keyopt[1][0] == [1, 0]
    assert archive.keyopt[1][1] == [2, 0]


def test_read_rlblock(tmp_path: Path) -> None:
    filename = str(tmp_path / "tmp.cdb")
    with open(filename, "w") as fid:
        fid.write(RLBLOCK_STR)

    archive = _reader.Archive(filename)
    archive.read_line()
    archive.read_rlblock()
    assert archive.rnum == [2]  # set number
    assert len(archive.rdat) == 1

    expected_val = np.array([1.0, 0.5669e-07, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(archive.rdat[0], expected_val)


def test_read_nblock(tmp_path: Path) -> None:
    filename = str(tmp_path / "tmp.cdb")
    with open(filename, "w") as fid:
        fid.write(NBLOCK_STR)

    archive = _reader.Archive(filename, debug=True)
    pos = archive.read_line()
    archive.read_nblock(pos)
    archive.n_nodes == 9
    assert np.array_equal(archive.nnum, NBLOCK_NODE_ID_ARR)
    assert np.allclose(archive.nodes, NBLOCK_POS_ARRAY)
    assert archive.nblock_start == pos


def test_read_nblock_incomplete(tmp_path: Path) -> None:
    filename = str(tmp_path / "tmp.cdb")
    with open(filename, "w") as fid:
        fid.write(NBLOCK_INCOMPLETE)

    archive = _reader.Archive(filename, debug=True)
    pos = archive.read_line()
    archive.read_nblock(pos)

    archive.n_nodes == 3
    assert np.array_equal(archive.nnum, [1, 2, 3])
    assert np.allclose(archive.nodes[0], [0.1, 0.0, 0.0])
    assert np.allclose(archive.nodes[1], [0.5, 0.8, 0.0])
    assert np.allclose(archive.nodes[2], [0.1, 0.7, 0.9])


def test_read_cmblock_node(tmp_path: Path) -> None:
    filename = str(tmp_path / "tmp.cdb")
    with open(filename, "w") as fid:
        fid.write(CMBLOCK_NODE_STR)

    archive = _reader.Archive(filename, debug=True)
    archive.read_line()
    archive.read_cmblock()
    assert "INTERFACE" in archive.node_comps
    assert np.array_equal(archive.node_comps["INTERFACE"], NCOMP_INTERFACE)


def test_read_cmblock_elem(tmp_path: Path) -> None:
    filename = str(tmp_path / "tmp.cdb")
    with open(filename, "w") as fid:
        fid.write(CMBLOCK_ELEM_STR)

    archive = _reader.Archive(filename, debug=True)
    archive.read_line()
    archive.read_cmblock()
    assert "ELMISC" in archive.elem_comps
    assert np.array_equal(archive.elem_comps["ELMISC"], ECOMP_ELMISC)


def test_read_mesh200() -> None:
    filename = os.path.join(TESTFILES_PATH, "mesh200.cdb")
    archive = _reader.Archive(filename, debug=True)
    archive.read()
    assert archive.n_nodes == 4961
    assert archive.n_elem == 1000

    # spot check
    # 1290        0        0 4.0000000000000E-001 6.0000000000000E-001
    assert archive.nnum[1289] == 1290
    assert np.allclose(archive.nodes[1289], np.array([0.4, 0.6, 0.0]))
    assert np.allclose(archive.node_angles[1289], 0)


def test_read_complex_archive() -> None:
    nblock_expected = np.array(
        [
            [3.7826539829200e00, 1.2788958692644e00, -1.0220880953640e00],
            [3.7987359490873e00, 1.2312085780780e00, -1.0001885444969e00],
            [3.8138798206653e00, 1.1833200772896e00, -9.7805743587145e-01],
            [3.7751258193793e00, 1.2956563072306e00, -9.9775569295981e-01],
            [3.7675976558386e00, 1.3124167451968e00, -9.7342329055565e-01],
            [3.8071756567432e00, 1.2018089624856e00, -9.5159140433025e-01],
            [3.8004714928212e00, 1.2202978476816e00, -9.2512537278904e-01],
            [3.7840345743299e00, 1.2663572964392e00, -9.4927433167235e-01],
            [3.8682501483615e00, 1.4211343558710e00, -9.2956245308371e-01],
            [3.8656154427804e00, 1.4283573726940e00, -9.3544082975315e-01],
            [3.8629807371994e00, 1.4355803895169e00, -9.4131920642259e-01],
            [3.8698134427618e00, 1.4168612083433e00, -9.3457292477788e-01],
            [3.8645201728196e00, 1.4314324609914e00, -9.4526873324423e-01],
            [3.8713767371621e00, 1.4125880608155e00, -9.3958339647206e-01],
            [3.8687181728010e00, 1.4199362966407e00, -9.4440082826897e-01],
            [3.8660596084399e00, 1.4272845324660e00, -9.4921826006588e-01],
            [3.7847463501820e00, 1.2869612289286e00, -1.0110875234148e00],
            [3.7882161293470e00, 1.2952473975570e00, -1.0006326084202e00],
            [3.7840036708439e00, 1.3089808408341e00, -9.8189659453120e-01],
            [3.7736944340897e00, 1.3175655146540e00, -9.6829193559890e-01],
            [3.7797912123408e00, 1.3227142841112e00, -9.6316058064216e-01],
            [3.8163322819008e00, 1.1913589544053e00, -9.6740419078720e-01],
            [3.8046827481496e00, 1.2474593204382e00, -9.7922600135387e-01],
            [3.8202228218151e00, 1.1995824283636e00, -9.5733187068101e-01],
            [3.9797161316330e00, 2.5147820926190e-01, -5.1500799817626e-01],
            [3.9831382922541e00, 2.0190980565891e-01, -5.0185526897444e-01],
            [3.9810868976408e00, 2.3910377061737e-01, -5.4962360790281e-01],
            [3.9772930845240e00, 2.8865001362748e-01, -5.6276585706615e-01],
            [3.9816265976187e00, 2.1428739259987e-01, -4.6723916677654e-01],
            [3.9839413943097e00, 1.8949722823843e-01, -5.3648152416530e-01],
            [3.7962006776348e00, 1.2764624207283e00, -9.3931008487698e-01],
            [3.8126101429289e00, 1.2302105573453e00, -9.1545958911180e-01],
            [3.8065408178751e00, 1.2252542025135e00, -9.2029248095042e-01],
            [3.8164164823720e00, 1.2148964928545e00, -9.3639572989640e-01],
            [3.8972892823450e00, 2.7547119775919e-01, -5.6510422311694e-01],
            [3.9015993648189e00, 2.0235606714652e-01, -4.6987255385930e-01],
            [3.9023812010290e00, 1.7705558022279e-01, -5.3881795411458e-01],
            [3.9019902829240e00, 1.8970582368465e-01, -5.0434525398694e-01],
            [3.8998352416870e00, 2.2626338899099e-01, -5.5196108861576e-01],
            [3.8994443235820e00, 2.3891363245285e-01, -5.1748838848812e-01],
            [3.9372911834345e00, 2.8206060569333e-01, -5.6393504009155e-01],
            [3.9416129812188e00, 2.0832172987319e-01, -4.6855586031792e-01],
            [3.9431612976694e00, 1.8327640423061e-01, -5.3764973913994e-01],
            [3.8619577233846e00, 1.4192189812407e00, -9.2587403626770e-01],
            [3.8507167163959e00, 1.4238788373222e00, -9.3661710728291e-01],
            [3.8651039358730e00, 1.4201766685559e00, -9.2771824467570e-01],
            [3.8624692302920e00, 1.4273996853788e00, -9.3359662134515e-01],
            [3.8610467267790e00, 1.4182334490688e00, -9.3810025187748e-01],
            [3.8563372198902e00, 1.4215489092814e00, -9.3124557177530e-01],
            [3.8568487267976e00, 1.4297296134196e00, -9.3896815685275e-01],
            [3.8583881624179e00, 1.4255816848941e00, -9.4291768367439e-01],
            [3.8594834323787e00, 1.4225065965966e00, -9.3308978018331e-01],
        ]
    )
    filename = os.path.join(TESTFILES_PATH, "all_solid_cells.cdb")

    archive = _reader.Archive(filename, debug=True)
    archive.read()
    assert np.allclose(archive.nodes, nblock_expected)
    assert archive.n_elem == 4

    # map ansys element type 4 to VTK type 4: 3D Solid (Hexahedral, wedge,
    # pyramid, tetrahedral)
    type_map = np.empty(1000, np.int32)
    type_map[4] = 4

    offset, cell_types, cells = archive.to_vtk(type_map)

    assert offset.size == 5
    assert cells.size == offset[-1]
    cell_types_expected = [
        CellType.QUADRATIC_HEXAHEDRON.value,
        CellType.QUADRATIC_WEDGE.value,
        CellType.QUADRATIC_PYRAMID.value,
        CellType.QUADRATIC_TETRA.value,
    ]
    assert np.array_equal(cell_types, cell_types_expected)
    assert np.array_equal(np.diff(offset), [20, 15, 13, 10])
