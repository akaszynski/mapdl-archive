"""Test the C++ wrapper."""

from py.path import local

import numpy as np
import numpy.typing as npt
import pytest

from mapdl_archive import _archive

NODE_ID_ARR = np.array(
    [635, 637, 638, 667, 668, 844, 845, 851, 919],
    dtype=np.int32,
)
POS_ARRAY = np.array(
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

EXPECTED_NBLOCK = """/PREP7
NBLOCK,6,SOLID,       919,         9
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


def proto_cmblock(array: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
    """prototype cmblock code"""
    items = np.zeros_like(array)
    items[0] = array[0]

    c = 1
    in_list = False
    for i in range(array.size - 1):
        # check if part of a range
        if array[i + 1] - array[i] == 1:
            in_list = True
        elif array[i + 1] - array[i] > 1:
            if in_list:
                items[c] = -array[i]
                c += 1
                items[c] = array[i + 1]
                c += 1
            else:
                items[c] = array[i + 1]
                c += 1
            in_list = False

    # check if we've ended on a list
    # catch if last item is part of a list
    if items[c - 1] != abs(array[-1]):
        items[c] = -array[i + 1]
        c += 1

    return items[:c]


@pytest.mark.parametrize(
    "array",
    (
        np.arange(1, 10, dtype=np.int32),
        np.array([1, 5, 10, 20, 40, 80], dtype=np.int32),
        np.array([1, 2, 3, 10, 20, 40, 51, 52, 53], dtype=np.int32),
        np.array([1, 2, 3, 10, 20, 40], dtype=np.int32),
        np.array([10, 20, 40, 50, 51, 52], dtype=np.int32),
    ),
)
def test_cmblock_items_from_array(array: npt.NDArray[np.int32]) -> None:
    """Simply verify it's identical to the prototype python code"""
    assert np.allclose(proto_cmblock(array), _archive.cmblock_items_from_array(array))


def test_write_nblock(tmpdir: local) -> None:
    filename = str(tmpdir.join("out.cdb"))
    angles = np.empty((0, 0), dtype=np.double)
    _archive.write_nblock(filename, NODE_ID_ARR.max(), NODE_ID_ARR, POS_ARRAY, angles, 13, "w")

    with open(filename) as fid:
        text = fid.read()

    assert text == EXPECTED_NBLOCK
