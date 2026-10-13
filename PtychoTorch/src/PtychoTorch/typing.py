import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import IO, Any

import h5py
from matplotlib.axes import Axes
from torch import Tensor

TFL = Tensor | float | list[float]
TF = Tensor | float
h5_root = str | h5py.Group | os.PathLike[Any] | IO[Any]

DiffPattern = Tensor
Probe = Tensor
Object = Tensor
ESW = Tensor

Roi2D = tuple[slice, slice]

TPlotter = Callable[[Axes], None]


@dataclass
class DiffPatternSet:
    """Measurement Set per Scanningposition.

    Args:
        diff_pattern: Measured Diffraction Pattern
        scan_position_SI: Scanning position of stage, where diffraction pattern was
            measured.
        is_barebeam: If set to `True`, then Object-patch is ignored and ESW is
            calculated by only the Probe.
        background: Image of background. Added to estimated diffraction patterns.
        exposure_time_s: Factor with which estimated diffraction patterns will be
            divided by. Used for HDR.

    """

    diff_pattern: DiffPattern
    scan_position_SI: Tensor | None  # (N_modes, 2)
    scan_position_px: Tensor | None  # (N_modes, 2)
    is_barebeam: bool = False
    background: DiffPattern | None = None
    exposure_time_s: float | Tensor | None = None  # ([])

    def check_dims(self, n_dims: int = 2):
        # moved import here to fix circular import
        from PtychoTorch.utils import check_tensor_dims_greater_equal

        check_tensor_dims_greater_equal(self.diff_pattern, n_dims)
