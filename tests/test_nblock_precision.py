"""Analytically authored IEEE fixtures through the actual native NBLOCK owner.

No customer coordinate, runtime float(str) oracle, VTK or geometry calculation.
Dyadic decimal texts are finite exact expansions N*5**k /10**k; expected bits
follow binary64 exponents/significands and round-to-nearest/ties-to-even.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from mapdl_archive import Archive

CASES = json.loads((Path(__file__).parent / "data" / "nblock_ieee_cases.json").read_text())


def archive_with_fields(tmp_path, fields, width=None):
    width = width or max(24, *(len(field) for field in fields))
    assert all(len(field) <= width for field in fields)
    path = tmp_path / "literal.cdb"
    row = f"{1:8}{0:8}{0:8}" + "".join(field.rjust(width) for field in fields)
    path.write_text(
        f"/PREP7\nNBLOCK,6,SOLID,1,1\n(3i8,6e{width}.17)\n" + row + "\nN,R5.3,LOC,      -1,\n/EOF\n"
    )
    # Actual C++ reader, not a Python decimal parser or mock.
    return Archive(path, parse_vtk=False, read_eblock=False)


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_exact_binary64_literals(tmp_path, case):
    archive = archive_with_fields(tmp_path, [case["literal"], "0", "1"])
    assert archive.nnum.tolist() == [1]
    assert archive.nodes.dtype == np.float64
    assert archive.nodes.view(np.uint64).tolist() == [
        [int(case["expected_bits"], 16), 0x0000000000000000, 0x3FF0000000000000]
    ]


def test_fixed_field_does_not_consume_following_digits(tmp_path):
    # No padding between these three four-character tokens.
    archive = archive_with_fields(tmp_path, ["1e+1", "2e+1", "3e+1"], width=4)
    assert archive.nodes.view(np.uint64).tolist() == [
        [0x4024000000000000, 0x4034000000000000, 0x403E000000000000]
    ]


def test_ascii_whitespace_plus_and_optional_angles(tmp_path):
    archive = archive_with_fields(tmp_path, ["\t+1.25D+1\r", " ", "-0", " ", "+2.5", "-0"])
    assert archive.nodes.view(np.uint64).tolist() == [
        [0x4029000000000000, 0x0000000000000000, 0x8000000000000000]
    ]
    assert archive.node_angles.view(np.uint64).tolist() == [
        [0x0000000000000000, 0x4004000000000000, 0x8000000000000000]
    ]


@pytest.mark.parametrize(
    "literal", ["garbage", "1.2junk", "1D+", "+", "nan", "inf", "1,25", "1\x00x"]
)
def test_invalid_numeric_field_is_not_a_plausible_coordinate(tmp_path, literal):
    with pytest.raises(ValueError):
        archive_with_fields(tmp_path, [literal, "0", "0"])


@pytest.mark.parametrize("literal", ["1e309", "-1D309"])
def test_overflow_is_reported(tmp_path, literal):
    with pytest.raises(OverflowError):
        archive_with_fields(tmp_path, [literal, "0", "0"])


def test_nul_empty_optional_angles_preserve_end_signal(tmp_path):
    archive = archive_with_fields(tmp_path, ["1", "0", "0", "\x00" * 24, "2", "3"])
    assert archive.nodes.view(np.uint64).tolist() == [
        [0x3FF0000000000000, 0x0000000000000000, 0x0000000000000000]
    ]
    assert archive.node_angles.view(np.uint64).tolist() == [[0, 0, 0]]
