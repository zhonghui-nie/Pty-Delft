from collections.abc import Callable, Iterable
from functools import wraps
from typing import TYPE_CHECKING, Any, Self

import torch
from torch._prims_common import DeviceLikeType, Tensor
from torch.types import Number

from PtychoTorch.propagators import BasePropagator
from PtychoTorch.utils import fft2c, fftfreq, get_circular_mask, ifft2c

from .typing import ESW, DiffPattern, DiffPatternSet, Object, Probe, Roi2D

if TYPE_CHECKING:
    from .engine import AutoGradPIE
else:
    AutoGradPIE = Any


class LossFunction:
    def __init__(
        self,
        inner: Callable[
            [DiffPatternSet, ESW, Probe | None, Object | None], Tensor | None
        ],
    ):
        self.inner = inner
        self.name = inner.__qualname__

    def __call__(
        self,
        diff_pattern_set: DiffPatternSet,
        esw: ESW,
        probe: Probe | None = None,
        object_patch: Object | None = None,
    ) -> Tensor:
        ret = self.inner(diff_pattern_set, esw, probe, object_patch)
        if ret is None:
            return torch.tensor(0.0, device=esw.device)
        return ret

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    @classmethod
    def create_constant(cls, val: Tensor | Number) -> Self:
        if not torch.is_tensor(val):
            val = torch.tensor(val)

        @cls
        @wraps(cls.__call__)
        def inner(*args: Any, **kwargs: Any) -> torch.Tensor:
            return val

        inner.name = f"{val}"
        return inner

    @classmethod
    def _check_other(cls, other: Any) -> Self:
        if isinstance(other, cls):
            return other

        if isinstance(other, Number):
            return cls.create_constant(other)

        if isinstance(other, Tensor):
            cls.create_constant(other)

        raise Exception(
            f"'other' has to be of type Number, Tensor, LossFunction. "
            f"Found: {type(other)}"
        )

    def __add__(self, other: Any):
        other = self._check_other(other)

        @LossFunction
        @wraps(LossFunction.__call__)
        def new_loss(*args: Any, **kwargs: Any) -> torch.Tensor:
            return self.inner(*args, **kwargs) + other(*args, **kwargs)

        new_loss.name = f"({self} + {other})"
        return new_loss

    def __radd__(self, other: Any):
        other = self._check_other(other)

        @LossFunction
        @wraps(LossFunction.__call__)
        def new_loss(*args: Any, **kwargs: Any):
            return other(*args, **kwargs) + self.inner(*args, **kwargs)

        new_loss.name = f"({other} + {self})"
        return new_loss

    def __iadd__(self, other: Any) -> Self:
        raise NotImplementedError

    def __sub__(self, other: Any):
        other = self._check_other(other)

        @LossFunction
        @wraps(LossFunction.__call__)
        def new_loss(*args: Any, **kwargs: Any) -> torch.Tensor:
            return self.inner(*args, **kwargs) - other(*args, **kwargs)

        new_loss.name = f"({self} - {other})"
        return new_loss

    def __rsub__(self, other: Any):
        other = self._check_other(other)

        @LossFunction
        @wraps(LossFunction.__call__)
        def new_loss(*args: Any, **kwargs: Any):
            return other(*args, **kwargs) - self.inner(*args, **kwargs)

        new_loss.name = f"({other} - {self})"
        return new_loss

    def __isub__(self, other: Any) -> Self:
        raise NotImplementedError

    def __mul__(self, other: Any):
        other = self._check_other(other)

        @LossFunction
        @wraps(LossFunction.__call__)
        def new_loss(*args: Any, **kwargs: Any) -> torch.Tensor:
            return self.inner(*args, **kwargs) * other(*args, **kwargs)

        new_loss.name = f"({self} * {other})"
        return new_loss

    def __rmul__(self, other: Any):
        other = self._check_other(other)

        @LossFunction
        @wraps(LossFunction.__call__)
        def new_loss(*args: Any, **kwargs: Any):
            return other(*args, **kwargs) * self.inner(*args, **kwargs)

        new_loss.name = f"({other} * {self})"
        return new_loss

    def __imul__(self, other: Any) -> Self:
        raise NotImplementedError

    def __truediv__(self, other: Any):
        other = self._check_other(other)

        @LossFunction
        @wraps(LossFunction.__call__)
        def new_loss(*args: Any, **kwargs: Any) -> torch.Tensor:
            return self.inner(*args, **kwargs) / other(*args, **kwargs)

        new_loss.name = f"({self} / {other})"
        return new_loss

    def __rtruediv__(self, other: Any):
        other = self._check_other(other)

        @LossFunction
        @wraps(LossFunction.__call__)
        def new_loss(*args: Any, **kwargs: Any):
            return other(*args, **kwargs) / self.inner(*args, **kwargs)

        new_loss.name = f"({other} / {self})"
        return new_loss

    def __itruediv__(self, other: Any) -> Self:
        raise NotImplementedError

    def __pow__(self, other: Any):
        other = self._check_other(other)

        @LossFunction
        @wraps(LossFunction.__call__)
        def new_loss(*args: Any, **kwargs: Any) -> torch.Tensor:
            return self.inner(*args, **kwargs) ** other(*args, **kwargs)

        new_loss.name = f"({self} ** {other})"
        return new_loss

    def __rpow__(self, other: Any):
        other = self._check_other(other)

        @LossFunction
        @wraps(LossFunction.__call__)
        def new_loss(*args: Any, **kwargs: Any):
            return other(*args, **kwargs) ** self.inner(*args, **kwargs)

        new_loss.name = f"({other} ** {self})"
        return new_loss

    def __ipow__(self, other: Any) -> Self:
        raise NotImplementedError


def Constant(val: Tensor | Number) -> LossFunction:
    return LossFunction.create_constant(val)


def ESW_to_intensity(
    esw: ESW,
    diff_pattern_set: DiffPatternSet | None = None,
) -> DiffPattern:
    """Converts amplitude to intensity.

    Args:
        esw_detector: El. field to be converted to intensity

    Returns:
        Intensity (Diffraction Pattern)

    """

    # amplitude at detector (per wavelength)
    intensity_detector = esw.abs().square()

    # sum incoherently over all dimensions other than x/y
    summing_dims = list(range(int(esw.dim()) - 2))
    if summing_dims != []:
        intensity_detector = intensity_detector.sum(dim=summing_dims)

    if diff_pattern_set is not None:
        if (bkgd := diff_pattern_set.background) is not None:
            intensity_detector = intensity_detector + bkgd

        if (exp_s := diff_pattern_set.exposure_time_s) is not None:
            intensity_detector = intensity_detector / exp_s

    return intensity_detector


@LossFunction
def MSE_amplitude(
    diff_pattern_set: DiffPatternSet,
    esw_detector: ESW,
    probe: Probe | None = None,
    object_patch: Object | None = None,
) -> torch.Tensor:
    """Mean Squared Error comparing amplitudes in detector space.

    Single Probe:

    .. math::

        L=\\sum_{\\text{pixels}}\\left( \\sqrt{I}
            - \\left\\lvert\\tilde{\\psi}\\right\\rvert \\right)^2

    For incoherenct summation with multiple wavelengths/Probes with index :math:`i`:

    .. math::

        L=\\sum_{\\text{pixels}}\\left( \\sqrt{I}
        - \\sqrt{\\sum_{i=1} \\left\\lvert\\tilde{\\psi_i}\\right\\rvert^2} \\right)^2

    Args:
        I_measured: Measured Diffraction Pattern
        esw_detector: Exit-Surface-Wave in camera plane
        probe: Not used for this loss-function.
        object_patch: Not used for this loss-function.

    Returns:
        Loss, call .backward() on this variable to calculate and store gradients.

    """
    I_measured = diff_pattern_set.diff_pattern
    I_estimated = ESW_to_intensity(
        esw=esw_detector,
        diff_pattern_set=diff_pattern_set,
    )
    return (I_measured.sqrt() - I_estimated.sqrt()).square().sum()


@LossFunction
def MSE_intensity(
    diff_pattern_set: DiffPatternSet,
    esw_detector: ESW,
    probe: Probe | None = None,
    object_patch: Object | None = None,
) -> torch.Tensor:
    """Mean Squared Error comparing intensities in detector space.

    Single Probe:

    .. math::

        L=\\sum_{\\text{pixels}}\\left( I
            - \\left\\lvert\\tilde{\\psi}\\right\\rvert \\right)^2

    For incoherenct summation with multiple wavelengths/Probes with index :math:`i`:

    .. math::

        L=\\sum_{\\text{pixels}}\\left( I
            - \\sum_{i=1} \\left\\lvert\\tilde{\\psi_i}\\right\\rvert^2 \\right)^2

    Args:
        I_measured: Measured Diffraction Pattern
        esw_detector: Exit-Surface-Wave in camera plane
        probe: Not used for this loss-function.
        object_patch: Not used for this loss-function.

    Returns:
        Loss, call .backward() on this variable to calculate and store gradients.

    """
    I_measured = diff_pattern_set.diff_pattern
    I_estimated = ESW_to_intensity(
        esw=esw_detector,
        diff_pattern_set=diff_pattern_set,
    )
    return (I_measured - I_estimated).square().sum()


class MaskProbe(LossFunction):
    def __init__(self, mask: torch.Tensor):
        """Mask for Probe.

        Multiplies mask with probe.

        Args:
            mask: Tensor with values between 0 and 1, or boolean mask.
                Dimension: (..., H, W)

        Returns:
            loss-function

        """
        super().__init__(self.inner)
        self.mask = mask

    def inner(
        self,
        diff_pattern_set: DiffPatternSet,
        esw_detector: ESW,
        probe: Probe | None = None,
        object_patch: Object | None = None,
    ) -> Tensor | None:
        if probe is None:
            return
        return (probe * self.mask).abs().sum()


class ApertureProbe(MaskProbe):
    def __init__(
        self,
        pupil_diameter: float,  # diameter of beam
        propagator: BasePropagator,
        sample_projection_angle_deg: float = 0,  # 0 = transmission
        mode: str = "hard_edge",
        device: DeviceLikeType | None = None,
    ):
        if device is None:
            device = torch.cpu.current_device()

        match mode.lower():
            case "hard_edge":
                mask = ~get_circular_mask(
                    pupil_diameter,
                    propagator.probe_shape,
                    propagator.dx,
                    sample_projection_angle_deg,
                    device,
                )
            case "hard_edge_smooth":
                mask = ~get_circular_mask(
                    pupil_diameter,
                    propagator.probe_shape,
                    propagator.dx,
                    sample_projection_angle_deg,
                    device,
                )
                mask = mask.abs()

                x = fftfreq(propagator.probe_shape[-1], device=device)
                XX, YY = torch.meshgrid([x, x], indexing="xy")
                r = torch.sqrt(XX.square() + YY.square())
                kernel = r < 1 / 50

                mask = ifft2c(fft2c(mask) * fft2c(kernel)).abs()

            case _:
                raise Exception("Mode not known")

        super().__init__(mask)


def COM_probe_center(
    correlated_dimensions: list[int] | int | None = None,
) -> LossFunction:
    if isinstance(correlated_dimensions, int):
        correlated_dimensions = [correlated_dimensions]

    if correlated_dimensions is None:
        correlated_dimensions = []

    @LossFunction
    def inner(
        diff_pattern_set: DiffPatternSet,
        esw_detector: ESW,
        probe: Probe | None = None,
        object_patch: Object | None = None,
    ) -> Tensor | None:
        if probe is None:
            return
        probe_sq = probe.abs().square()
        height, width = probe.shape[-2:]

        sum_x = probe_sq.sum(-2)
        sum_y = probe_sq.sum(-1)

        com_x = (torch.arange(width, device=probe.device) * sum_x).sum(
            dim=-1
        ) / sum_x.sum(
            dim=-1,
            keepdim=False,
        )
        com_y = (torch.arange(height, device=probe.device) * sum_y).sum(
            dim=-1
        ) / sum_y.sum(
            dim=-1,
            keepdim=False,
        )

        all_dims = list(range(probe.dim()))
        for _d in correlated_dimensions:
            _d_sum = all_dims[_d]
            com_x = com_x.mean(_d_sum, keepdim=True)
            com_y = com_y.mean(_d_sum, keepdim=True)

        shift_x = com_y - height // 2
        shift_y = com_x - width // 2
        return (shift_x.abs() + shift_y.abs()).sum()

    return inner


def same_value(roi_list: Iterable[Roi2D], engine: AutoGradPIE) -> LossFunction:
    """Enforces all rois to have the same value with respect to each other."""

    @LossFunction
    def same_value(
        diff_pattern_set: DiffPatternSet,
        esw_detector: ESW,
        probe: Probe | None = None,
        object_patch: Object | None = None,
    ) -> torch.Tensor:
        return torch.stack(
            [engine.rec_object[..., *roi].mean() for roi in roi_list]
        ).std()

    return same_value


def const_indiv_ampl(roi_list: Iterable[Roi2D], engine: AutoGradPIE) -> LossFunction:
    @LossFunction
    def const_indiv_ampl(
        diff_pattern_set: DiffPatternSet,
        esw_detector: ESW,
        probe: Probe | None = None,
        object_patch: Object | None = None,
    ) -> torch.Tensor:
        return torch.stack(
            [engine.rec_object[..., *roi].abs().mean() for roi in roi_list]
        ).std()

    return const_indiv_ampl


def const_value_roi(
    roi: Roi2D,
    engine: AutoGradPIE,
    value: complex = 1,
) -> LossFunction:
    """Enforces roi to have const value."""

    O_mask = torch.ones(engine.rec_object.shape, dtype=torch.bool, device=engine.device)
    O_mask[..., *roi] = False

    @LossFunction
    def const_value_roi(
        diff_pattern_set: DiffPatternSet,
        esw_detector: ESW,
        probe: Probe | None = None,
        object_patch: Object | None = None,
    ) -> torch.Tensor:
        return (engine.rec_object * O_mask - value).abs().sum()

    return const_value_roi
