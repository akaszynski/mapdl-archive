from typing import Dict, List

import numpy as np
import numpy.typing as npt

class Archive:
    elem: npt.NDArray[np.int32]
    elem_off: npt.NDArray[np.int32]
    n_elem: int
    keyopt: Dict[int, List[List[int]]]

    def __init__(
        self, fname: str, readParams: bool = False, dbg: bool = False, readEblock: bool = True
    ) -> None: ...
    def read_et_line(self) -> None: ...
    def read_etblock(self) -> None: ...
    def read_eblock(self) -> None: ...
    def read_keyopt_line(self) -> None: ...
    def get_element_types(self) -> npt.NDArray[np.int32]: ...
