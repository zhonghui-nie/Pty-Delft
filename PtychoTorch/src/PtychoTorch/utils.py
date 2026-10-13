import math
from collections.abc import Iterable, Mapping
from numbers import Number
from typing import TYPE_CHECKING, Any

import torch
from torch._prims_common import DeviceLikeType, Tensor

if TYPE_CHECKING:
    from PtychoTorch.typing import Roi2D

    from .dataset import DataSet
    from .engine import AutoGradPIE
else:
    Roi2D = Any
    DataSet = Any
    AutoGradPIE = Any


def fft2c(x: torch.Tensor) -> torch.Tensor:
    """Fast Fourier Transform.

    Calculate 2D fast fourier transform, keeping :math:`k=0` in the center.

    Args:
        x

    """

    x = torch.fft.ifftshift(x, dim=[-2, -1])  # type: ignore
    x = torch.fft.fft2(x, norm="ortho")  # type: ignore
    x = torch.fft.fftshift(x, dim=[-2, -1])  # type: ignore
    return x  # pyright: ignore[reportUnknownVariableType]


def ifft2c(x: torch.Tensor) -> torch.Tensor:
    """Fast Fourier Transform.

    Calculate inverse 2D fast fourier transform, assuming :math:`k=0` was in the center.

    Args:
        x

    """
    x = torch.fft.ifftshift(x, dim=[-2, -1])  # type:ignore
    x = torch.fft.ifft2(x, norm="ortho")  # type:ignore
    x = torch.fft.fftshift(x, dim=[-2, -1])  # type:ignore
    return x  # pyright: ignore[reportUnknownVariableType]


def fftfreq(
    n: int,
    d: float = 1,
    dtype: torch.dtype | None = None,
    device: DeviceLikeType | None = None,
    requires_grad: bool = False,
) -> torch.Tensor:
    ret: torch.Tensor = torch.fft.fftfreq(  # type: ignore
        n=n,
        d=d,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    return torch.fft.fftshift(ret)  # type:ignore


def pad_or_crop(
    img: torch.Tensor,
    new_shape: torch.Size | tuple[int, ...] | int,
    padding_val: float = 0,
) -> torch.Tensor:
    """Pads or crops ``img`` to `new_shape`

    If ``new_shape`` is smaller/larger than ``img.shape``, then ``img`` will be
    cropped/padded to `new_shape`, keeping ``img`` centered. Only the last two
    dimensions of ``img`` are affected.

    Args:
        img: image, or batch of images, to be padded/cropped. Must be at least
            2-dimensional.
        new_shape: shape to pad/crop to. If only integer is provided, then a square
            shape is assumed with length `new_shape`.
        padding_val: Value used for padding

    Returns:
        ``img`` with updated shape

    """
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    H_old, W_old = img.shape[-2:]
    H_new, W_new = new_shape[-2:]

    def get_padding(old_shape: int, new_shape: int, reverse: bool = False):
        diff = (new_shape - old_shape) / 2
        if (
            new_shape % 2 == 0
            and old_shape % 2 == 0
            or new_shape % 2 != 0
            and old_shape % 2 != 0
        ):  # even to even
            pad_start = int(diff)
            pad_end = int(diff)

        elif new_shape % 2 == 0 and old_shape % 2 != 0:  # even to odd
            pad_end = math.floor(diff)
            pad_start = math.ceil(diff)

        elif new_shape % 2 != 0 and old_shape % 2 == 0:  # odd to even
            pad_end = math.ceil(diff)
            pad_start = math.floor(diff)
        else:
            raise Exception

        if reverse:
            return pad_end, pad_start
        else:
            return pad_start, pad_end

    pad_bottom, pad_top = get_padding(H_old, H_new, reverse=True)
    pad_left, pad_right = get_padding(W_old, W_new)

    img_pad = torch.nn.functional.pad(
        img,
        (pad_left, pad_right, pad_top, pad_bottom),
        value=padding_val,
    )
    return img_pad


def interpolate_fourier(img: torch.Tensor, new_shape: torch.Size) -> torch.Tensor:
    """Interpolate by padding/cropping in fourier space.

    .. warning:: Bug: Account for Nyquist frequency depending on even/uneven img-shape

    Args:
        img: img (``HW``), or batch of images (``...HW``)
        new_shape: shape to interpolate to (``H'W'``)

    Returns:
        img interpolated to ``new_shape``

    """
    img_ft = fft2c(img)
    img_ft_new = pad_or_crop(img_ft, new_shape, padding_val=0)
    return ifft2c(img_ft_new)


def get_slice(position: tuple[int, int], shape: torch.Size) -> Roi2D:
    """Returns corresponding slice for Object.

    Args:
        position: Position of upper-left corner of batch. Units in pixels.
        shape: Shape of Probe used.

    Returns:
        y-slice, x-slice

    """
    pos_x = int(position[-1])
    pos_y = int(position[-2])

    slice_x = slice(pos_x, pos_x + int(shape[-1]), 1)
    slice_y = slice(pos_y, pos_y + int(shape[-2]), 1)

    return slice_y, slice_x


def shift_img_patch(
    img_patch: torch.Tensor,
    shifts_px: torch.Tensor,
):
    """Shift and/or decomposed object patch.

    Args:
        img_patch: Image patch. Dimension: (..., H, W)
        shifts_px: Relative shift with respect to object patch (relative to current
            scangrid position). Dimension: (N_modes, 2)

    Returns:
        shifted object patch. Dimension (N_modes, ..., H, W)

    """
    check_tensor_dims_equal(shifts_px, 2)

    x = fftfreq(
        img_patch.shape[-1],
        1 / (2 * torch.pi),
        device=img_patch.device,
    )
    y = fftfreq(
        img_patch.shape[-2],
        1 / (2 * torch.pi),
        device=img_patch.device,
    )

    XX_list: list[torch.Tensor] = []
    YY_list: list[torch.Tensor] = []

    for i in range(len(shifts_px)):
        _shift = shifts_px[i]
        XX, YY = torch.meshgrid([x * _shift[1], y * _shift[0]], indexing="xy")
        XX_list.append(XX)
        YY_list.append(YY)

    XX_tensor = torch.stack(XX_list)
    YY_tensor = torch.stack(YY_list)

    img_patch_ft = fft2c(img_patch)
    img_patch_ft = img_patch_ft * torch.exp(1j * XX_tensor + 1j * YY_tensor)
    img_patch_shifted = ifft2c(img_patch_ft)

    return img_patch_shifted


def check_tensor_dims_equal(t: torch.Tensor, dims: int) -> None:
    if t.dim() != dims:
        raise Exception(f"Dimensions of tensor do not match {t.dim()}!={dims=}")


def check_tensor_dims_greater_equal(t: torch.Tensor, dims: int) -> None:
    if t.dim() >= dims:
        raise Exception(f"Dimensions of tensor do not match: {t.dim()=}!>={dims=}")


def check_tensor_one_dim(t: torch.Tensor, dim: int, n_dim: int):
    if t.shape[dim] != n_dim:
        raise Exception(f"Dimension {dim=} does not match: {t.shape[dim]=} != {n_dim=}")


def dict_to_log_str(
    title: str,
    items_dict: Mapping[str, Tensor | int | float | Number | str],
    sep: str = "\t",
    check_length: bool = True,
):
    new_line = "\n"
    ret_str = title

    for key, val in items_dict.items():
        if torch.is_tensor(val):
            if check_length and (val.numel() > 100):
                val = " -- too large to display --"

            else:
                val = str(val.tolist())  # type: ignore

        elif isinstance(val, int):
            val = str(val)

        elif isinstance(val, Number):
            val = f"{val:.2e}"

        ret_str += f"{new_line}{sep}{key}={val}"

    return ret_str


def circ(x: torch.Tensor, radius: torch.Tensor | float) -> torch.Tensor:
    height, width = x.shape[-2:]
    X = torch.arange(-width // 2, width // 2, device=x.device)
    Y = torch.arange(-height // 2, height // 2, device=x.device)

    YY, XX = torch.meshgrid([Y, X], indexing="ij")

    distances = torch.sqrt(XX.square() + YY.square()).expand(x.shape)
    if torch.is_tensor(radius) and radius.dim() > 0:
        radius = radius.view(*distances.shape[:-2], 1, 1)
    return distances <= radius


def get_circular_mask(
    pupil_diameter: float,
    probe_shape: torch.Size | tuple[int, int],
    dx: float,
    sample_projection_angle_deg: float,
    device: DeviceLikeType | None = None,
):
    if device is None:
        device = torch.cpu.current_device()

    x = (
        (torch.arange(probe_shape[-1], device=device) - probe_shape[-1] / 2)
        * dx
        * math.cos(sample_projection_angle_deg / 180 * math.pi)
    )
    y = (torch.arange(probe_shape[-2], device=device) - probe_shape[-2] / 2) * dx
    XX, YY = torch.meshgrid([x, y], indexing="xy")

    ret = (XX.square() + YY.square()).sqrt() < pupil_diameter / 2
    return ret.expand(probe_shape)


def change_binning(
    engine: AutoGradPIE | Iterable[AutoGradPIE],
    ds: DataSet,
    binning_factor: int,
) -> None:
    # import here to avoid circular import
    from .engine import AutoGradPIE

    if isinstance(engine, AutoGradPIE):
        return change_binning(
            engine=[engine],
            ds=ds,
            binning_factor=binning_factor,
        )

    # first check if the propagator is compatible with the dataset
    for e in engine:
        e.propagator.check_compatibility_with_dataset(ds)

    # change binning of engine
    for e in engine:
        e.change_binning(binning_factor)

    # change binning of dataset
    ds.set_binning_factor(binning_factor)


def change_cropping(
    engine: AutoGradPIE | Iterable[AutoGradPIE],
    ds: DataSet,
    new_probe_shape1D_unbinned: int,
) -> None:
    # import here to avoid circular import
    from .engine import AutoGradPIE

    if isinstance(engine, AutoGradPIE):
        return change_cropping(
            engine=[engine],
            ds=ds,
            new_probe_shape1D_unbinned=new_probe_shape1D_unbinned,
        )

    # first check if the propagator is compatible with the dataset
    for e in engine:
        e.propagator.check_compatibility_with_dataset(ds)

    # change binning of engine
    for e in engine:
        e.change_detector_size(new_probe_shape1D=new_probe_shape1D_unbinned)

    # change binning of dataset
    ds.set_crop_shape(
        torch.Size((new_probe_shape1D_unbinned, new_probe_shape1D_unbinned))
    )
