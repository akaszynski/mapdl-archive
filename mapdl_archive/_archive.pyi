import numpy as np
import numpy.typing as npt

def py_write_nblock(
    filename: str,
    node_id: npt.NDArray[np.int32],
    max_node_id: int,
    pos: npt.NDArray[np.float64],
    angles: npt.NDArray[np.float64],
    mode: str = "w",
    sig_digits: int = 13,
) -> None: ...
def py_write_nblock_float(
    filename: str,
    node_id: npt.NDArray[np.int32],
    max_node_id: int,
    pos: npt.NDArray[np.float32],
    angles: npt.NDArray[np.float32],
    mode: str = "w",
    sig_digits: int = 13,
) -> None: ...
def py_write_eblock(
    filename: str,
    elem_id: npt.NDArray[np.int32],
    etype: npt.NDArray[np.int32],
    mtype: npt.NDArray[np.int32],
    rcon: npt.NDArray[np.int32],
    elem_nnodes: npt.NDArray[np.int32],
    cells: npt.NDArray[np.int32],
    offset: npt.NDArray[np.int32],
    celltypes: npt.NDArray[np.uint8],
    typenum: npt.NDArray[np.int32],
    nodenum: npt.NDArray[np.int32],
    mode: str = "w",
) -> None: ...
def cmblock_items_from_array(array: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]: ...
