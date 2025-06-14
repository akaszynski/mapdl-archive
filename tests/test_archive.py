"""Test ``mapdl_archive/archive.py``."""

import filecmp
import os
import pathlib
import sys
from typing import Union, Any, Dict

from numpy.typing import NDArray
from pathlib import Path
import numpy as np
import pytest
import pyvista as pv
from pyvista import CellType
from pyvista import examples as pyvista_examples
from pyvista.plotting import system_supports_plotting  # type: ignore

import mapdl_archive
from mapdl_archive import Archive, _archive, examples, _reader

# Windows issues Python 3.12
WINDOWS = os.name == "nt"
WIN_PY312 = sys.version_info.minor == 12 and WINDOWS
skip_plotting = pytest.mark.skipif(
    WINDOWS or not system_supports_plotting(),  # type: ignore
    reason="Requires active X Server",
)

LINEAR_CELL_TYPES = [
    CellType.TETRA,
    CellType.PYRAMID,
    CellType.WEDGE,
    CellType.HEXAHEDRON,
]
QUADRATIC_CELL_TYPES = [
    CellType.QUADRATIC_TETRA,
    CellType.QUADRATIC_PYRAMID,
    CellType.QUADRATIC_WEDGE,
    CellType.QUADRATIC_HEXAHEDRON,
]

TEST_PATH = os.path.dirname(os.path.abspath(__file__))
TESTFILES_PATH = os.path.join(TEST_PATH, "test_data")
TESTFILES_PATH_PATHLIB = pathlib.Path(TESTFILES_PATH)
DAT_FILE = os.path.join(TESTFILES_PATH, "Panel_Transient.dat")
BEAM186_DOS_FILE = os.path.join(TESTFILES_PATH, "Beam_186TetQuadAnglesDOS.cdb")
CORRUPT_CDB_FILE_A = os.path.join(TESTFILES_PATH, "corrupt_a.cdb")
CORRUPT_CDB_FILE_B = os.path.join(TESTFILES_PATH, "corrupt_b.cdb")


@pytest.fixture()
def pathlib_archive() -> Archive:
    filename = TESTFILES_PATH_PATHLIB / "ErnoRadiation.cdb"
    return Archive(filename)


@pytest.fixture()
def hex_archive() -> Archive:
    return Archive(examples.hexarchivefile)


@pytest.fixture(scope="module")
def all_solid_cells_archive() -> Archive:
    return Archive(os.path.join(TESTFILES_PATH, "all_solid_cells.cdb"))


@pytest.fixture(scope="module")
def all_solid_cells_archive_linear() -> Archive:
    return Archive(os.path.join(TESTFILES_PATH, "all_solid_cells.cdb"), force_linear=True)


def compare_dicts_with_arrays(
    dict1: Dict[str, NDArray[Any]], dict2: Dict[str, NDArray[Any]]
) -> bool:
    """Compare two dictionaries containing arrays."""
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1:
        if not np.array_equal(dict1[key], dict2[key]):
            return False
    return True


def test_load_dat() -> None:
    arch = Archive(DAT_FILE)
    assert arch.n_node == 1263  # through inspection of the dat file
    assert arch.n_elem == 160  # through inspection of the dat file


def test_repr(hex_archive: Archive) -> None:
    assert f"{hex_archive.n_node}" in str(hex_archive)
    assert f"{hex_archive.n_elem}" in str(hex_archive)


@skip_plotting
def test_plot(hex_archive: Archive) -> None:
    cpos = hex_archive.plot(return_cpos=True)
    assert isinstance(cpos, pv.CameraPosition)


def test_read_mesh200() -> None:
    archive = Archive(os.path.join(TESTFILES_PATH, "mesh200.cdb"))
    assert archive.grid.n_cells == 1000


def test_archive_init(hex_archive: Archive) -> None:
    assert isinstance(hex_archive._archive, _reader.Archive)
    assert isinstance(hex_archive.grid, pv.UnstructuredGrid)

    assert hex_archive._nblock_start > 0
    assert hex_archive._nblock_end > hex_archive._nblock_start


def test_parse_vtk(hex_archive: Archive) -> None:
    grid = hex_archive.grid
    assert grid.points.size
    assert grid.cell_connectivity.size
    assert grid.cell_connectivity.dtype == np.int32
    assert grid.offset.dtype == np.int32
    assert "ansys_node_num" in grid.point_data

    with pytest.raises(TypeError):
        hex_archive._parse_vtk(allowable_types=-1)  # type: ignore

    with pytest.raises(TypeError):
        hex_archive._parse_vtk(allowable_types=3.0)  # type: ignore


def test_invalid_archive(tmp_path: Path, hex_archive: Archive) -> None:
    nblock_filename = tmp_path / "nblock.cdb"
    mapdl_archive.write_nblock(nblock_filename, hex_archive.nnum, hex_archive.nodes)

    archive = Archive(nblock_filename)
    assert archive.grid.n_cells == 0


def test_write_angle(tmp_path: Path, hex_archive: Archive) -> None:
    nblock_filename = tmp_path / "nblock.cdb"
    mapdl_archive.write_nblock(
        nblock_filename, hex_archive.nnum, hex_archive.nodes, hex_archive.node_angles
    )

    archive = Archive(nblock_filename, parse_vtk=False)
    assert np.allclose(archive.nodes, hex_archive.nodes)


def test_missing_midside() -> None:
    allowable_types = [45, 95, 185, 186, 92, 187]
    archive_file = os.path.join(TESTFILES_PATH, "mixed_missing_midside.cdb")
    archive = Archive(archive_file, allowable_types=allowable_types)

    assert not np.any(archive.grid.celltypes == CellType.TETRA)


def test_missing_midside_write(tmp_path: Path) -> None:
    allowable_types = [45, 95, 185, 186, 92, 187]
    archive_file = os.path.join(TESTFILES_PATH, "mixed_missing_midside.cdb")
    archive = Archive(archive_file, allowable_types=allowable_types)

    filename = tmp_path / "tmp.cdb"
    with pytest.raises(RuntimeError, match="Unsupported element types"):
        mapdl_archive.save_as_archive(filename, archive.grid, exclude_missing=True)

    mapdl_archive.save_as_archive(filename, archive.grid, exclude_missing=True, reset_etype=True)
    archive_new = Archive(filename)

    with pytest.raises(TypeError, match="must be an UnstructuredGrid"):
        mapdl_archive.save_as_archive(filename, [1, 2, 3])


def test_writehex(tmp_path: Path, hex_archive: Archive) -> None:
    filename = tmp_path / "tmp.cdb"
    mapdl_archive.save_as_archive(filename, hex_archive.grid)
    archive_new = Archive(filename)

    assert np.allclose(hex_archive.grid.points, archive_new.grid.points)
    assert np.allclose(
        hex_archive.grid.cell_connectivity,
        archive_new.grid.cell_connectivity,
    )

    for node_component in hex_archive.node_components:
        assert np.allclose(
            hex_archive.node_components[node_component],
            archive_new.node_components[node_component],
        )

    for element_component in hex_archive.element_components:
        assert np.allclose(
            hex_archive.element_components[element_component],
            archive_new.element_components[element_component],
        )


def test_write_voxel(tmp_path: Path) -> None:
    filename = tmp_path / "tmp.cdb"
    grid = pv.ImageData(dimensions=(10, 10, 10))
    mapdl_archive.save_as_archive(filename, grid)

    archive = Archive(filename)
    assert np.allclose(archive.grid.points, grid.points)
    assert np.allclose(archive.grid.point_data["ansys_node_num"], range(1, 1001))
    assert archive.grid.n_cells, grid.n_cells


def test_writesector(tmp_path: Path) -> None:
    archive = Archive(examples.sector_archive_file)
    filename = tmp_path / "tmp.cdb"
    mapdl_archive.save_as_archive(filename, archive.grid)
    archive_new = Archive(filename)

    assert np.allclose(archive.grid.points, archive_new.grid.points)
    assert np.allclose(archive.grid.cells, archive_new.grid.cells)


def test_writehex_missing_elem_num(tmp_path: Path, hex_archive: Archive) -> None:
    grid = hex_archive.grid
    grid.cell_data["ansys_elem_num"][:10] = -1
    grid.cell_data["ansys_etype"] = np.ones(grid.number_of_cells) * -1
    grid.cell_data["ansys_elem_type_num"] = np.ones(grid.number_of_cells) * -1

    filename = tmp_path / "tmp.cdb"
    mapdl_archive.save_as_archive(filename, grid)
    archive_new = Archive(filename)

    assert np.allclose(hex_archive.grid.points, archive_new.grid.points)
    assert np.allclose(hex_archive.grid.cells, archive_new.grid.cells)


def test_writehex_missing_node_num(tmp_path: Path, hex_archive: Archive) -> None:
    hex_archive.grid.point_data["ansys_node_num"][:-1] = -1

    filename = tmp_path / "tmp.cdb"
    mapdl_archive.save_as_archive(filename, hex_archive.grid)
    archive_new = Archive(filename)
    assert np.allclose(hex_archive.grid.points.shape, archive_new.grid.points.shape)
    assert np.allclose(hex_archive.grid.cells.size, archive_new.grid.cells.size)


def test_write_non_ansys_grid(tmp_path: Path) -> None:
    grid = pv.UnstructuredGrid(pyvista_examples.hexbeamfile)
    del grid.point_data["sample_point_scalars"]
    del grid.cell_data["sample_cell_scalars"]
    archive_file = tmp_path / "tmp.cdb"
    mapdl_archive.save_as_archive(archive_file, grid)


def test_read_complex_archive(all_solid_cells_archive: Archive) -> None:
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
    assert np.allclose(nblock_expected, all_solid_cells_archive.nodes)

    grid = all_solid_cells_archive.grid
    assert grid.n_cells == 4
    assert np.unique(grid.celltypes).size == 4
    assert np.all(grid.celltypes > 20)


def test_read_complex_archive_linear(all_solid_cells_archive_linear: Archive) -> None:
    grid = all_solid_cells_archive_linear.grid
    assert np.all(grid.celltypes < 20)


@pytest.mark.parametrize("celltype", QUADRATIC_CELL_TYPES)
def test_write_quad_complex_archive(
    tmp_path: Path, celltype: CellType, all_solid_cells_archive: Archive
) -> None:
    grid = all_solid_cells_archive.grid
    mask = grid.celltypes == celltype
    assert mask.any()
    grid = grid.extract_cells(mask)

    tmp_archive_file = tmp_path / "tmp.cdb"

    mapdl_archive.save_as_archive(tmp_archive_file, grid)
    new_archive = Archive(tmp_archive_file)
    assert np.allclose(grid.cells, new_archive.grid.cells)
    assert np.allclose(grid.points, new_archive.grid.points)


@pytest.mark.parametrize("celltype", LINEAR_CELL_TYPES)
def test_write_lin_archive(
    tmp_path: Path, celltype: CellType, all_solid_cells_archive_linear: Archive
) -> None:
    linear_grid = all_solid_cells_archive_linear.grid

    mask = linear_grid.celltypes == celltype
    assert mask.any()
    linear_grid = linear_grid.extract_cells(mask)

    tmp_archive_file = tmp_path / "tmp.cdb"

    mapdl_archive.save_as_archive(tmp_archive_file, linear_grid)
    new_archive = Archive(tmp_archive_file)
    assert np.allclose(linear_grid.celltypes, new_archive.grid.celltypes)


@pytest.mark.parametrize("as_array", [False, True])
def test_write_component(tmp_path: Path, as_array: bool) -> None:
    items = [1, 20, 50, 51, 52, 53]
    filename = tmp_path / "tmp.cdb"

    comp_name = "TEST"
    if as_array:
        mapdl_archive.write_cmblock(filename, items, comp_name, "node")
    else:
        mapdl_archive.write_cmblock(filename, np.array(items), comp_name, "node")

    archive = Archive(filename)
    assert np.allclose(archive.node_components[comp_name], items)


def test_write_component_edge_case(tmp_path: Path) -> None:
    items = np.arange(2, 34, step=2)
    filename = tmp_path / "tmp.cdb"

    comp_name = "TEST"
    mapdl_archive.write_cmblock(filename, items, comp_name, "node")
    archive = Archive(filename)
    assert np.allclose(archive.node_components[comp_name], items)

    with pytest.raises(ValueError, match="must be either"):
        mapdl_archive.write_cmblock(filename, items, comp_name, "FOO")


def test_read_parm() -> None:
    filename = os.path.join(TESTFILES_PATH, "parm.cdb")

    with pytest.raises(RuntimeError, match="deprecated"):
        archive = Archive(filename, read_parameters=True)


def test_read_wb_nblock() -> None:
    expected = np.array(
        [
            [9.89367578e-02, -8.07092192e-04, 8.53764953e00],
            [9.65803244e-02, 2.00906704e-02, 8.53744951e00],
            [9.19243555e-02, 3.98781615e-02, 8.53723652e00],
        ]
    )
    filename = os.path.join(TESTFILES_PATH, "workbench_193.cdb")
    archive = Archive(filename)
    assert np.allclose(archive.nodes, expected)

    assert archive.node_angles is not None
    assert np.allclose(archive.node_angles, 0)


def test_read_hypermesh() -> None:
    expected = np.array(
        [
            [-6.01203, 2.98129, 2.38556],
            [-3.03231, 2.98067, 2.38309],
            [-0.03485, 2.98004, 2.3805],
            [2.98794, 2.97941, 2.37773],
            [5.98956, 2.97878, 2.37488],
            [5.98956, 5.97878, 2.37488],
        ]
    )

    filename = os.path.join(TESTFILES_PATH, "hypermesh.cdb")
    archive = Archive(filename, verbose=True)
    assert np.allclose(archive.nodes[:6], expected)


@pytest.mark.parametrize("has_angles", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_write_nblock(
    hex_archive: Archive,
    tmp_path: Path,
    dtype: np.dtype,  # type: ignore
    has_angles: bool,
) -> None:
    nblock_filename = tmp_path / "nblock.inp"

    nodes = hex_archive.nodes.astype(dtype)
    if has_angles:
        angles = hex_archive.node_angles
    else:
        angles = None
    mapdl_archive.write_nblock(nblock_filename, hex_archive.nnum, nodes, angles, mode="w")

    tmp_archive = Archive(nblock_filename)
    assert np.allclose(hex_archive.nnum, tmp_archive.nnum)
    assert np.allclose(hex_archive.nodes, tmp_archive.nodes)
    assert hex_archive.node_angles is not None
    assert tmp_archive.node_angles is not None
    if has_angles:
        assert np.allclose(hex_archive.node_angles, tmp_archive.node_angles)


@pytest.mark.parametrize("sig_digits", [8, 18])
def test_write_nblock_sig_digits(hex_archive: Archive, tmp_path: Path, sig_digits: int) -> None:
    nblock_filename = tmp_path / "nblock.inp"

    nodes = hex_archive.nodes
    angles = hex_archive.node_angles
    with pytest.raises(ValueError, match="sig_digits"):
        mapdl_archive.write_nblock(nblock_filename, hex_archive.nnum, nodes, angles, sig_digits=-1)

    mapdl_archive.write_nblock(
        nblock_filename, hex_archive.nnum, nodes, angles, sig_digits=sig_digits
    )  # more than 18 fails
    tmp_archive = Archive(nblock_filename)
    assert np.allclose(hex_archive.nnum, tmp_archive.nnum)
    assert np.allclose(hex_archive.nodes, tmp_archive.nodes)
    assert hex_archive.node_angles is not None
    assert tmp_archive.node_angles is not None
    assert np.allclose(hex_archive.node_angles, tmp_archive.node_angles)


def test_write_eblock(hex_archive: Archive, tmp_path: Path) -> None:
    filename = str(tmp_path / "eblock.inp")

    etype = np.ones(hex_archive.n_elem, np.int32)
    typenum = hex_archive.etype
    elem_nnodes = np.empty(etype.size, np.int32)
    elem_nnodes[typenum == 181] = 4
    elem_nnodes[typenum == 185] = 8
    elem_nnodes[typenum == 186] = 20
    elem_nnodes[typenum == 187] = 10
    nodenum = hex_archive.nnum

    cells, offset = hex_archive.grid.cell_connectivity, hex_archive.grid.offset
    _archive.write_eblock(
        filename,
        hex_archive.enum.size,
        hex_archive.enum,
        etype,
        hex_archive.material_type,
        np.ones(hex_archive.n_elem, np.int32),
        elem_nnodes,
        hex_archive.grid.celltypes,
        offset,
        cells,
        typenum,
        nodenum,
        "w",
    )


def test_rlblock_prior_to_nblock() -> None:
    # test edge case where RLBLOCK is immediately prior to the NBLOCK
    filename = os.path.join(TESTFILES_PATH, "ErnoRadiation.cdb")
    archive = Archive(filename)
    assert archive.n_node == 65
    assert archive.n_elem == 36


def test_etblock() -> None:
    # test edge case where RLBLOCK is immediately prior to the NBLOCK
    filename = os.path.join(TESTFILES_PATH, "etblock.cdb")
    archive = Archive(filename)
    assert archive.n_node == 4
    assert archive.n_elem == 1


def test_overwrite_nblock(tmp_path: Path, hex_archive: Archive) -> None:
    # ensure that we capture the entire NBLOCK
    with open(hex_archive._filename, "rb") as fid:
        fid.seek(hex_archive._nblock_start)
        nblock_txt = fid.read(hex_archive._nblock_end - hex_archive._nblock_start).decode()

    assert nblock_txt.startswith("NBLOCK")
    assert nblock_txt.splitlines()[-1].endswith("-1,")

    filename = tmp_path / "tmp.cdb"
    nodes = np.random.random(hex_archive.nodes.shape)
    hex_archive.overwrite_nblock(filename, nodes)

    archive_new = Archive(filename)
    assert np.allclose(nodes, archive_new.grid.points)
    assert np.allclose(hex_archive.nnum, archive_new.nnum)

    assert hex_archive.node_angles is not None
    assert archive_new.node_angles is not None
    assert np.allclose(hex_archive.node_angles, archive_new.node_angles)

    assert np.allclose(hex_archive.grid.cells.size, archive_new.grid.cells.size)

    # overwrite with original nodes (tests for zeros)
    hex_archive.overwrite_nblock(filename, hex_archive.nodes)


def test_overwrite_nblock(tmp_path: Path) -> None:
    # ensure that we capture the entire NBLOCK
    arc = mapdl_archive.Archive(BEAM186_DOS_FILE)

    filename = tmp_path / "tmp.cdb"
    arc.overwrite_nblock(filename, arc.nodes)

    arc_new = mapdl_archive.Archive(filename)
    assert np.allclose(arc_new.nodes, arc.nodes)

    assert arc.node_angles is not None
    assert arc_new.node_angles is not None
    assert np.allclose(arc_new.node_angles, arc.node_angles)


def test_pathlib_filename_property(pathlib_archive: Archive) -> None:
    assert isinstance(pathlib_archive.pathlib_filename, pathlib.Path)


def test_filename_property_is_string(pathlib_archive: Archive) -> None:
    filename = TESTFILES_PATH_PATHLIB / "ErnoRadiation.cdb"
    arch = Archive(filename)
    assert isinstance(arch.filename, str)


def test_corrupt_cdb_coordinates() -> None:
    with pytest.raises(
        RuntimeError, match="Failed to read NBLOCK coordinates. Last node number read was 100"
    ):
        Archive(CORRUPT_CDB_FILE_A)


def test_corrupt_cdb_node_number() -> None:
    with pytest.raises(
        RuntimeError, match="Failed to read NBLOCK node number. Last node number read was 100"
    ):
        Archive(CORRUPT_CDB_FILE_B)


def test_save_as_numpy(tmp_path: Path, hex_archive: Archive) -> None:
    """Test saving to and loading from a npz file."""
    npz_path = tmp_path / "data.npz"
    hex_archive.save_as_numpy(npz_path)
    hex_in = Archive(npz_path)

    assert np.array_equal(hex_in._nnum, hex_archive._nnum)
    assert np.array_equal(hex_in._nodes, hex_archive._nodes)
    assert np.array_equal(hex_in._node_angles, hex_archive._node_angles)
    assert np.array_equal(hex_in._elem, hex_archive._elem)
    assert np.array_equal(hex_in._elem_off, hex_archive._elem_off)
    assert np.array_equal(hex_in._ekey, hex_archive._ekey)
    assert np.array_equal(hex_in._rnum, hex_archive._rnum)
    assert compare_dicts_with_arrays(hex_in._node_comps, hex_archive._node_comps)
    assert compare_dicts_with_arrays(hex_in._elem_comps, hex_archive._elem_comps)
    assert hex_in._rdat == hex_archive._rdat
    # assert compare_dicts_with_arrays(hex_in._keyopt, hex_archive._keyopt)
