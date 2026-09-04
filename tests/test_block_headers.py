"""Tests for NBLOCK and EBLOCK headers, coordinate formats, and multiple element blocks.

Everything here is built from synthetic archives written by the test itself, so
the cases stay small and readable.  None of it needs PyVista: reading an archive
is pure NumPy.
"""

from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pytest

from mapdl_archive import Archive

# ---------------------------------------------------------------------------
# helpers for writing fixed width MAPDL records
# ---------------------------------------------------------------------------

ID_WIDTH = 8


def node_record(node_id: int, coords: Sequence[str], width: int = 16) -> str:
    """Build one NBLOCK record.

    ``coords`` are written verbatim, right justified in ``width`` characters, so
    a test can pin down the exact text the reader has to cope with.
    """
    for coord in coords:
        assert len(coord) <= width, f"{coord!r} does not fit in {width} characters"
    line = f"{node_id:>{ID_WIDTH}}{0:>{ID_WIDTH}}{0:>{ID_WIDTH}}"
    return line + "".join(coord.rjust(width) for coord in coords)


def elem_record(
    elem_id: int, nodes: Sequence[int], etype: int = 1, width: int = 8, per_line: int = 19
) -> str:
    """Build one EBLOCK record.

    MAPDL wraps at the number of fields the format line declares, so an element
    with more than eight nodes occupies a second line.
    """
    fields: List[int] = [
        1,  # material
        etype,  # element type
        1,  # real constant
        1,  # section
        0,  # element coordinate system
        0,  # birth/death
        0,  # solid model reference
        0,  # coded shape key
        len(nodes),  # number of nodes
        0,  # unused
        elem_id,
    ]
    fields.extend(nodes)
    lines = [
        "".join(f"{value:>{width}}" for value in fields[start : start + per_line])
        for start in range(0, len(fields), per_line)
    ]
    return "\n".join(lines)


HEX = (1, 2, 3, 4, 5, 6, 7, 8)
HEX20 = tuple(range(1, 21))

# Every stored element is 8 header values + the element number + a padding zero
# + its node ids.
INTS_PER_HEX = 18
INTS_PER_HEX20 = 30


def nblock(header: str, fmt: str, records: Iterable[str]) -> str:
    body = "\n".join(records)
    return f"{header}\n{fmt}\n{body}\nN,R5.3,LOC,      -1,\n"


def eblock(header: str, records: Iterable[str], fmt: str = "(19i8)") -> str:
    body = "\n".join(records)
    return f"{header}\n{fmt}\n{body}\n      -1\n"


def write_cdb(tmp_path: Path, *blocks: str, name: str = "model.cdb") -> str:
    path = tmp_path / name
    path.write_text("/PREP7\nET,1,45\n" + "".join(blocks) + "/EOF\n")
    return str(path)


# Eight nodes of a unit cube, written without an exponent and with more than one
# digit before the decimal point.  This is the "(3i8,6e16.9)" style.
PLAIN_COORDS = [
    ("29.184036609179", "1.500000000000", "-2.500000000000"),
    ("-0.67492887261", "30.431943412509", "46.073558738815"),
    ("123.45678901234", "0.100000000000", "0.200000000000"),
    ("0.300000000000", "0.400000000000", "0.500000000000"),
    ("1.000000000000", "2.000000000000", "3.000000000000"),
    ("4.000000000000", "5.000000000000", "6.000000000000"),
    ("7.000000000000", "8.000000000000", "9.000000000000"),
    ("10.500000000000", "11.250000000000", "12.125000000000"),
]

PLAIN_EXPECTED = np.array([[float(v) for v in row] for row in PLAIN_COORDS])


def plain_records() -> List[str]:
    return [node_record(i + 1, coords) for i, coords in enumerate(PLAIN_COORDS)]


def full_model(nblock_header: str, eblock_header: str) -> List[str]:
    return [
        nblock(nblock_header, "(3i8,6e16.9)", plain_records()),
        eblock(eblock_header, [elem_record(1, HEX)]),
    ]


# ---------------------------------------------------------------------------
# NBLOCK headers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "header",
    [
        "NBLOCK,6,SOLID",  # no counts at all
        "NBLOCK,6,SOLID,",  # trailing comma, blank count
        "NBLOCK,6,SOLID,        ",  # blank count padded with spaces
        "NBLOCK,6,SOLID,8",  # NDMAX only
        "NBLOCK,6,SOLID,8,8",  # NDMAX and NDSEL
        "NBLOCK,6,SOLID,      99,       8",  # NDMAX over-states, NDSEL is right
    ],
    ids=["none", "blank", "blank_padded", "ndmax", "both", "ndmax_high"],
)
def test_nblock_header_count_variants(tmp_path: Path, header: str) -> None:
    """Every one of these headers describes the same eight nodes."""
    path = write_cdb(tmp_path, *full_model(header, "EBLOCK,19,SOLID,"))
    archive = Archive(path, parse_vtk=False)

    assert archive.n_node == 8
    assert np.array_equal(archive.nnum, np.arange(1, 9, dtype=np.int32))
    assert np.allclose(archive.nodes, PLAIN_EXPECTED, rtol=1e-15, atol=0)


def test_nblock_without_count_is_not_dropped(tmp_path: Path) -> None:
    """A header the reader cannot count from must not silently yield no nodes."""
    path = write_cdb(tmp_path, *full_model("NBLOCK,6,SOLID", "EBLOCK,19,SOLID,"))
    archive = Archive(path, parse_vtk=False)

    assert archive.nodes.shape == (8, 3)
    assert archive.nodes.any(), "node block was dropped"


def test_nblock_overstated_count_stops_at_terminator(tmp_path: Path) -> None:
    """An over-estimated count must stop at "N,R5.3,LOC,-1", not read through it."""
    path = write_cdb(tmp_path, *full_model("NBLOCK,6,SOLID,9999,9999", "EBLOCK,19,SOLID,"))
    archive = Archive(path, parse_vtk=False)

    assert archive.n_node == 8
    assert np.allclose(archive.nodes, PLAIN_EXPECTED, rtol=1e-15, atol=0)


def test_nblock_unreadable_format_line_raises(tmp_path: Path) -> None:
    """A format line that cannot be understood is an error, not a silent drop."""
    path = write_cdb(
        tmp_path,
        nblock("NBLOCK,6,SOLID", "(xyz)", plain_records()),
        eblock("EBLOCK,19,SOLID,", [elem_record(1, HEX)]),
    )
    with pytest.raises(RuntimeError, match="node block format line"):
        Archive(path, parse_vtk=False)


# ---------------------------------------------------------------------------
# coordinate formats
# ---------------------------------------------------------------------------


def test_coordinates_without_exponent(tmp_path: Path) -> None:
    """Two and three digits before the decimal point, and a negative value."""
    path = write_cdb(tmp_path, *full_model("NBLOCK,6,SOLID", "EBLOCK,19,SOLID,"))
    nodes = Archive(path, parse_vtk=False).nodes

    assert nodes[0, 0] == 29.184036609179
    assert nodes[1, 0] == -0.67492887261
    assert nodes[1, 2] == 46.073558738815
    assert nodes[2, 0] == 123.45678901234


# The reader scales an integer mantissa by a power of ten rather than calling
# strtod, so a value that needs a power of ten can land one ulp off the
# correctly rounded double. That is long standing behaviour and is not what
# these tests are about, so inexact values are compared to within an ulp.
ULP = 1e-15


def test_coordinates_with_exponent(tmp_path: Path) -> None:
    """The "(3i8,6e20.13)" scientific form still reads correctly."""
    records = [
        node_record(1, ("2.9184036609179E+01", "1.5000000000000E+00", "-2.5000000000000E+00"), 20),
        node_record(2, ("-6.7492887261000E-01", "3.0431943412509E+01", "4.6073558738815E+01"), 20),
        node_record(3, ("1.0000000000000E-001", "7.0000000000000E-01", "9.0000000000000E-01"), 20),
    ]
    path = write_cdb(tmp_path, nblock("NBLOCK,6,SOLID", "(3i8,6e20.13)", records))
    nodes = Archive(path, parse_vtk=False).nodes

    assert nodes[0, 0] == pytest.approx(29.184036609179, rel=ULP)
    assert nodes[0, 1] == 1.5  # exactly representable
    assert nodes[0, 2] == -2.5  # exactly representable
    assert nodes[1, 0] == pytest.approx(-0.67492887261, rel=ULP)
    assert nodes[1, 1] == pytest.approx(30.431943412509, rel=ULP)
    assert nodes[1, 2] == pytest.approx(46.073558738815, rel=ULP)
    assert nodes[2, 0] == pytest.approx(0.1, rel=ULP)
    assert nodes[2, 2] == pytest.approx(0.9, rel=ULP)


def test_mixed_exponent_and_plain_coordinates(tmp_path: Path) -> None:
    """The same value written both ways reads to the same double."""
    records = [
        node_record(1, ("29.184036609179", "1.500000000000", "-2.500000000000"), 20),
        node_record(2, ("2.9184036609179E+01", "1.5000000000000E+00", "-2.5000000000000E+00"), 20),
    ]
    path = write_cdb(tmp_path, nblock("NBLOCK,6,SOLID", "(3i8,6e20.13)", records))
    nodes = Archive(path, parse_vtk=False).nodes

    assert nodes[0, 0] == pytest.approx(nodes[1, 0], rel=ULP)
    assert nodes[0, 0] == 29.184036609179  # the plain form is exact
    assert nodes[0, 1] == nodes[1, 1] == 1.5
    assert nodes[0, 2] == nodes[1, 2] == -2.5


def test_blank_float_field_is_zero(tmp_path: Path) -> None:
    """A blank coordinate field is zero, and must not hang or read garbage."""
    records = [
        node_record(1, ("", "1.500000000000", "-2.500000000000")),
        node_record(2, ("3.250000000000", "", "4.500000000000")),
    ]
    path = write_cdb(tmp_path, nblock("NBLOCK,6,SOLID", "(3i8,6e16.9)", records))
    nodes = Archive(path, parse_vtk=False).nodes

    assert nodes[0, 0] == 0.0
    assert nodes[0, 1] == 1.5
    assert nodes[1, 0] == 3.25
    assert nodes[1, 1] == 0.0
    assert nodes[1, 2] == 4.5


def test_short_record_pads_with_zeros(tmp_path: Path) -> None:
    """Records that stop early leave the remaining coordinates at zero."""
    records = [
        node_record(1, ("1.500000000000",)),
        node_record(2, ("2.500000000000", "3.500000000000")),
        node_record(3, ("4.500000000000", "5.500000000000", "6.500000000000")),
    ]
    path = write_cdb(tmp_path, nblock("NBLOCK,6,SOLID", "(3i8,6e16.9)", records))
    nodes = Archive(path, parse_vtk=False).nodes

    assert np.array_equal(
        nodes,
        np.array([[1.5, 0.0, 0.0], [2.5, 3.5, 0.0], [4.5, 5.5, 6.5]]),
    )


# ---------------------------------------------------------------------------
# EBLOCK headers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "header",
    [
        "EBLOCK,19,SOLID",  # no count at all
        "EBLOCK,19,SOLID,",  # trailing comma, blank count
        "EBLOCK,19,SOLID,        ",  # blank count padded with spaces
        "EBLOCK,19,SOLID      ,",  # padding before the comma too
        "EBLOCK,19,SOLID,2",  # NDMAX only
        "EBLOCK,19,SOLID,      99,       2",  # NDMAX over-states, NDSEL is right
    ],
    ids=["none", "blank", "blank_padded", "padded_key", "ndmax", "ndmax_high"],
)
def test_eblock_header_count_variants(tmp_path: Path, header: str) -> None:
    """Every one of these headers describes the same two elements."""
    path = write_cdb(
        tmp_path,
        nblock("NBLOCK,6,SOLID", "(3i8,6e16.9)", plain_records()),
        eblock(header, [elem_record(1, HEX), elem_record(2, HEX)]),
    )
    archive = Archive(path, parse_vtk=False)

    assert archive.n_elem == 2
    assert np.array_equal(archive.enum, np.array([1, 2], dtype=np.int32))
    assert np.array_equal(
        archive._archive.elem_off,
        np.array([0, INTS_PER_HEX, 2 * INTS_PER_HEX], dtype=np.int32),
    )
    for element in archive.elem:
        assert np.array_equal(element[-8:], np.array(HEX, dtype=np.int32))


def test_eblock_overstated_count_stops_at_terminator(tmp_path: Path) -> None:
    """A count larger than the block leaves no phantom elements behind."""
    path = write_cdb(
        tmp_path,
        nblock("NBLOCK,6,SOLID", "(3i8,6e16.9)", plain_records()),
        eblock("EBLOCK,19,SOLID,9999,9999", [elem_record(1, HEX), elem_record(2, HEX)]),
    )
    archive = Archive(path, parse_vtk=False)

    assert archive.n_elem == 2
    assert np.array_equal(archive.enum, np.array([1, 2], dtype=np.int32))
    assert archive._archive.elem_off[-1] == archive._archive.elem.size


# ---------------------------------------------------------------------------
# multiple element blocks
# ---------------------------------------------------------------------------


def test_single_eblock_is_unchanged(tmp_path: Path) -> None:
    """The single block case is the baseline the concatenation must not disturb."""
    path = write_cdb(
        tmp_path,
        nblock("NBLOCK,6,SOLID,8,8", "(3i8,6e16.9)", plain_records()),
        eblock("EBLOCK,19,SOLID,      99,       3", [elem_record(i, HEX) for i in (1, 2, 3)]),
    )
    archive = Archive(path, parse_vtk=False)

    assert archive.n_elem == 3
    assert np.array_equal(archive.enum, np.array([1, 2, 3], dtype=np.int32))
    assert np.array_equal(
        archive._archive.elem_off,
        np.arange(4, dtype=np.int32) * INTS_PER_HEX,
    )


def test_two_solid_eblocks_are_concatenated(tmp_path: Path) -> None:
    """Both blocks are read, in file order, as one element array."""
    path = write_cdb(
        tmp_path,
        nblock("NBLOCK,6,SOLID,8,8", "(3i8,6e16.9)", plain_records()),
        eblock("EBLOCK,19,SOLID,", [elem_record(1, HEX)]),
        eblock("EBLOCK,19,SOLID,", [elem_record(2, HEX), elem_record(3, HEX)]),
    )
    archive = Archive(path, parse_vtk=False)

    assert archive.n_elem == 3
    assert np.array_equal(archive.enum, np.array([1, 2, 3], dtype=np.int32))
    assert np.array_equal(
        archive._archive.elem_off,
        np.arange(4, dtype=np.int32) * INTS_PER_HEX,
    )
    assert archive._archive.elem.size == 3 * INTS_PER_HEX


def test_many_solid_eblocks_are_concatenated(tmp_path: Path) -> None:
    """File order is preserved across more than two blocks."""
    blocks = [eblock("EBLOCK,19,SOLID,", [elem_record(i, HEX)]) for i in range(1, 6)]
    path = write_cdb(
        tmp_path,
        nblock("NBLOCK,6,SOLID", "(3i8,6e16.9)", plain_records()),
        *blocks,
    )
    archive = Archive(path, parse_vtk=False)

    assert archive.n_elem == 5
    assert np.array_equal(archive.enum, np.arange(1, 6, dtype=np.int32))
    assert np.array_equal(
        archive._archive.elem_off,
        np.arange(6, dtype=np.int32) * INTS_PER_HEX,
    )


def test_non_solid_eblocks_are_still_ignored(tmp_path: Path) -> None:
    """Concatenating SOLID blocks must not start picking up the others."""
    path = write_cdb(
        tmp_path,
        nblock("NBLOCK,6,SOLID", "(3i8,6e16.9)", plain_records()),
        eblock("EBLOCK,19,SOLID,", [elem_record(1, HEX)]),
        eblock("EBLOCK,19,XYZ,        1", [elem_record(99, HEX)]),
        eblock("EBLOCK,19,SOLID,", [elem_record(2, HEX)]),
    )
    archive = Archive(path, parse_vtk=False)

    assert archive.n_elem == 2
    assert np.array_equal(archive.enum, np.array([1, 2], dtype=np.int32))


def test_multiline_elements_across_blocks(tmp_path: Path) -> None:
    """Records that wrap onto a second line still concatenate correctly.

    A countless block is sized by counting lines, which over-states a block of
    twenty node elements by a factor of two, so the offsets are the check that
    the over-estimate is not carried into the combined arrays.
    """
    path = write_cdb(
        tmp_path,
        nblock("NBLOCK,6,SOLID", "(3i8,6e16.9)", plain_records()),
        eblock("EBLOCK,19,SOLID,", [elem_record(1, HEX20)]),
        eblock("EBLOCK,19,SOLID,", [elem_record(2, HEX20), elem_record(3, HEX20)]),
    )
    archive = Archive(path, parse_vtk=False)

    assert archive.n_elem == 3
    assert np.array_equal(archive.enum, np.array([1, 2, 3], dtype=np.int32))
    assert np.array_equal(
        archive._archive.elem_off,
        np.arange(4, dtype=np.int32) * INTS_PER_HEX20,
    )
    for element in archive.elem:
        assert np.array_equal(element[-20:], np.array(HEX20, dtype=np.int32))


def test_eblock_running_to_end_of_file_gains_no_phantom_element(tmp_path: Path) -> None:
    """A block cut off by the end of the file must not invent a record.

    Without a "-1" terminator there is nothing to stop on, and the line count
    that sizes a countless block leaves room for one element per line.
    """
    path = tmp_path / "truncated.cdb"
    path.write_text(
        "/PREP7\nET,1,186\n"
        + nblock("NBLOCK,6,SOLID", "(3i8,6e16.9)", plain_records())
        + "EBLOCK,19,SOLID,\n(19i8)\n"
        + elem_record(7, HEX20)
        + "\n"
    )
    archive = Archive(str(path), parse_vtk=False)

    assert archive.n_elem == 1
    assert np.array_equal(archive.enum, np.array([7], dtype=np.int32))
    assert np.array_equal(archive._archive.elem_off, np.array([0, INTS_PER_HEX20], dtype=np.int32))
    assert archive._archive.elem.size == INTS_PER_HEX20


def test_read_eblock_false_reads_no_elements(tmp_path: Path) -> None:
    """The opt-out still applies with several blocks present."""
    path = write_cdb(
        tmp_path,
        nblock("NBLOCK,6,SOLID", "(3i8,6e16.9)", plain_records()),
        eblock("EBLOCK,19,solid,", [elem_record(1, HEX)]),
        eblock("EBLOCK,19,solid,", [elem_record(2, HEX)]),
    )
    archive = Archive(path, parse_vtk=False, read_eblock=False)

    assert archive.n_elem == 0
    assert archive.n_node == 8


# ---------------------------------------------------------------------------
# the whole combination, as a real deck writes it
# ---------------------------------------------------------------------------


def test_deck_with_no_counts_and_many_blocks(tmp_path: Path) -> None:
    """No NBLOCK counts, blank EBLOCK counts, plain coordinates, many blocks."""
    blocks = [eblock("EBLOCK,19,SOLID,        ", [elem_record(i, HEX)]) for i in range(1, 18)]
    path = write_cdb(
        tmp_path,
        nblock("NBLOCK,6,SOLID", "(3i8,6e16.9)", plain_records()),
        *blocks,
    )
    archive = Archive(path, parse_vtk=False)

    assert archive.n_node == 8
    assert archive.n_elem == 17
    assert np.allclose(archive.nodes, PLAIN_EXPECTED, rtol=1e-15, atol=0)
    assert np.array_equal(archive.enum, np.arange(1, 18, dtype=np.int32))
    assert np.array_equal(np.unique(archive.etype), np.array([45], dtype=np.int32))
