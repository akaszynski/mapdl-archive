import numpy as np
import numpy.typing as npt

def write_nblock(
    filename: str,
    max_node_id: int,
    node_id: npt.NDArray[np.int32],
    nodes: npt.NDArray[np.float64],
    angles: npt.NDArray[np.float64],
    sig_digits: int,
    mode: str,
) -> None: ...
def write_eblock(
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
