from typing import Dict, List, Tuple, TypedDict, Union

import numpy as np
import numpy.typing as npt

class ReadReturnDict(TypedDict):
    rnum: npt.NDArray[np.int64]
    rdat: List[List[float]]
    ekey: npt.NDArray[np.int32]
    nnum: npt.NDArray[np.int32]
    nodes: npt.NDArray[np.float64]
    elem: npt.NDArray[np.int32]
    elem_off: npt.NDArray[np.int32]
    node_comps: Dict[str, npt.NDArray[np.int32]]
    elem_comps: Dict[str, npt.NDArray[np.int32]]
    keyopt: Dict[int, List[List[int]]]
    parameters: Dict[str, np.ndarray]
    nblock_start: int
    nblock_end: int

def read(
    filename: str,
    read_parameters: bool = False,
    debug: bool = False,
    read_eblock: bool = True,
) -> ReadReturnDict: ...
def node_block_format(
    string: Union[bytes, str],
) -> Tuple[npt.NDArray[np.int32], int, int, int]: ...
def ans_vtk_convert(
    elem: npt.NDArray[np.int32],
    elem_off: npt.NDArray[np.int32],
    type_ref: npt.NDArray[np.int32],
    nnum: npt.NDArray[np.int32],
    build_offset: int,
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.uint8], npt.NDArray[np.int64]]: ...
def read_from_nwrite(
    filename: str, nnodes: int
) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.float64]]: ...
def write_array(filename: str, arr: npt.NDArray[np.float64]) -> None: ...
