import numpy as np
import numpy.typing as npt

def reset_midside(
    cellarr: npt.NDArray[np.int64],
    celltypes: npt.NDArray[np.uint8],
    offset: npt.NDArray[np.int64],
    pts: npt.NDArray[np.float64],
) -> None: ...
