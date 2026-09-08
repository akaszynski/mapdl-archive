"""Synthetic wedge file mappings across VTK's 9.7 ordering change."""

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from mapdl_archive import Archive, save_as_archive


@pytest.mark.parametrize("quadratic", [False, True])
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
