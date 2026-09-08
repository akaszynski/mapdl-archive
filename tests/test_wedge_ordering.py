"""Synthetic wedge file mappings across VTK's 9.7 ordering change."""

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from mapdl_archive import Archive, save_as_archive


@pytest.mark.parametrize("quadratic", [False, True])
@pytest.mark.filterwarnings("error::pyvista.PyVistaDeprecationWarning")
def test_wedge_archive_order_and_roundtrip(tmp_path, quadratic):
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]], float)
    edges = np.array([[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3], [0, 3], [1, 4], [2, 5]])
    if quadratic:
        points = np.vstack([points, points[edges].mean(axis=1)])
    ids = np.arange(len(points))
    if pv.vtk_version_info < (9, 7):
        order = [0, 2, 1, 3, 5, 4]
        if quadratic:
            order += [8, 7, 6, 11, 10, 9, 12, 14, 13]
        ids = ids[order]
    grid = pv.UnstructuredGrid([len(ids), *ids], [26 if quadratic else 13], points)
    original = grid.cells.copy()
    path = tmp_path / "prism.cdb"
    save_as_archive(path, grid)
    restored = Archive(path).grid
    np.testing.assert_array_equal(grid.cells, original)
    np.testing.assert_array_equal(restored.cells, original)
    np.testing.assert_allclose(restored.points, points)
    assert restored.volume == pytest.approx(0.5)

    # Independently check the written degenerate-hexahedron representation:
    # its first triangle winds toward the opposite triangular cap. This also
    # catches a reader/writer pair that round-trips the same wrong convention.
    raw = Archive(path, parse_vtk=False)
    node_ids = raw.elem[0][10:18]
    nodes = raw.nodes[np.searchsorted(raw.nnum, node_ids)]
    signed = np.dot(np.cross(nodes[1] - nodes[0], nodes[2] - nodes[0]), nodes[4] - nodes[0])
    assert signed > 0

    # Missing midsides are repaired after native assembly, so exercise that
    # path with the version-specific ordering already in the buffer.
    if quadratic:
        raw._elem[raw._elem_off[0] + 18] = 0
        repaired = raw._parse_vtk()
        cell_points = repaired.points[repaired.cell_connectivity]
        np.testing.assert_allclose(cell_points[6:], cell_points[edges].mean(axis=1))
        assert repaired.volume == pytest.approx(0.5)


@pytest.mark.parametrize("vtk_97_wedges", [False, True])
@pytest.mark.parametrize("quadratic", [False, True])
def test_native_loader_wedge_flag(quadratic, vtk_97_wedges):
    """Assemble each VTK convention directly from synthetic EBLOCK node slots."""
    from mapdl_archive import _reader

    # A positively ordered degenerate brick; the last twelve slots are its
    # quadratic midsides (repeated slots belong to collapsed edges).
    nodes = [1, 2, 3, 3, 4, 5, 6, 6]
    if quadratic:
        nodes += [7, 8, 3, 9, 10, 11, 6, 12, 13, 14, 15, 15]
    elem = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 1, *nodes], dtype=np.int32)
    elem_before = elem.copy()
    args = (
        elem,
        np.array([0, elem.size], dtype=np.int32),
        np.array([0, 4], dtype=np.int32),
        np.arange(1, (16 if quadratic else 7), dtype=np.int32),
    )
    offset, celltypes, cells = _reader.ans_to_vtk(*args, vtk_97_wedges=vtk_97_wedges)
    if vtk_97_wedges:
        expected = [2, 0, 1, 5, 3, 4]
        if quadratic:
            expected += [8, 6, 7, 11, 9, 10, 14, 12, 13]
    else:
        expected = [2, 1, 0, 5, 4, 3]
        if quadratic:
            expected += [7, 6, 8, 10, 9, 11, 14, 13, 12]
        np.testing.assert_array_equal(_reader.ans_to_vtk(*args)[2], cells)
    np.testing.assert_array_equal(cells, expected)
    np.testing.assert_array_equal(offset, [0, len(expected)])
    np.testing.assert_array_equal(celltypes, [26 if quadratic else 13])
    np.testing.assert_array_equal(elem, elem_before)
    assert cells.dtype == np.int32
