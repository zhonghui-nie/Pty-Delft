import math

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib_scalebar.scalebar import ScaleBar
from PIL import ImageDraw, ImageFont
from PIL.Image import Image
from torch._prims_common import Tensor
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import draw_bounding_boxes, make_grid

from PtychoTorch.typing import ESW, Object, Probe, Roi2D, TPlotter
from PtychoTorch.utils import check_tensor_dims_equal, pad_or_crop


def hsv2rgb(hsv: Tensor) -> Tensor:
    """Converts HSV to RGB.

    Adapted from `limacv/RGV_HSV_HSL <https://github.com/limacv/RGB_HSV_HSL>`_

    Args:
        hsv: input with shape (Batch, Color, H, W)

    Returns:
        RGB tensor

    """
    check_tensor_dims_equal(hsv, 4)
    hsv_h, hsv_s, hsv_l = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    _c = hsv_l * hsv_s
    _x = _c * (-torch.abs(hsv_h * 6.0 % 2.0 - 1) + 1.0)
    _m = hsv_l - _c
    _o = torch.zeros_like(_c)
    idx = (hsv_h * 6.0).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3, -1, -1)
    rgb = torch.empty_like(hsv)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m
    return rgb


def complex_arr_to_rgb(
    arr: Tensor,
    overexpose: float = 1,
    window_size: int | None = None,
) -> torch.Tensor:
    """Converts complex ``arr`` to RGB tensor.

    Normalized to brightest pixel.
    The phase of ``arr`` is represented by the hue. Zero-phase corresponds to hue
    :math:`h=180^\\circ`. ``torch.nan`` pixels are defined to be white.

    Args:
        arr: img (``HW``), or batch of images (``BHW``)
        overexpose: Number by which brightest pixel will be multiplied with after
            normalization.

    Returns:
        RGB tensor with same shape as ``arr``

    """

    # arr.shape: BHW
    if arr.dim() == 2:
        arr = arr.unsqueeze(0)

    mask = arr.isnan()
    arr = arr.detach().clone()
    arr[mask] = 0

    if window_size is not None:
        H, W = mask.shape[-2:]
        cy, cx = H // 2, W // 2
        half_window = window_size // 2
        mask2 = torch.zeros_like(arr)
        mask2[
            ...,
            cy - half_window : cy + half_window,
            cx - half_window : cx + half_window,
        ] = 1
        arr = arr * mask2

    v = arr.abs()
    v /= v.max()
    v *= overexpose
    v[v > 1] = 1

    s = torch.ones_like(v)
    if arr.is_floating_point():
        h = torch.zeros_like(s) + 0.5
    else:
        h = (arr.angle() + torch.pi) / (2 * torch.pi)

    h[mask] = 0
    s[mask] = 0
    v[mask] = 1
    hsv = torch.stack([h, s, v], 1)
    return hsv2rgb(hsv)


def plot_hsv(
    arr: Probe | Object | ESW,
    ax: Axes | None = None,
    dx: float = 1.0,
    labels: list[str] | None = None,
    unit_name: str | None = None,
    remove_phase_ramp: bool = False,
    mode: str = "MIXED",
    rois: dict[str, Roi2D] | list[Roi2D] | None = None,
    window_size: int | None = None,
    overexpose: float = 1,
    show_scalebar: bool = True,
) -> AxesImage:
    """Plot Probe, Object, ESW with optional circular phase colorbar.

    Args:
        arr: img (``HW``), or batch of images (``BHW``). If batch is provided, every
            image is separated by a small padding.
        ax: Axes to use for plotting. If set to ``None``, use the current Axes.
        dx: Pixel size. (in 'm')
        labels: List (``B``) of labels to show in the plots. If ``None``, no labels are
            shown.
        remove_phase_ramp: If ``True``, plotted image will have linear phase ramps
            removed by centering the FT(image).
        colorbar: If ``True``, a circular colorbar representing phase will be added to
            the plot.
        mode: Plotting mode. One of ('MIXED', 'AMPLITUDE', 'ABS', 'PHASE', 'ANGLE').
        overexpose: Number by which brightest pixel will be multiplied with after
            normalization.

    """
    if ax is None:
        ax = plt.gca()

    n_images = 1
    nrow = 1
    ncol = 1
    padding = 2
    pad_val = 0.8

    if remove_phase_ramp:
        raise NotImplementedError
        # arr = remove_linear_phase_ramp(arr)

    match mode.upper():
        case "MIXED":
            pass
        case "AMPLITUDE" | "ABS":
            arr = arr.abs()
        case "INTENSITY" | "ABS_SQR":
            arr = arr.abs().square()
        case "MIXED_INTENSITY" | "MIXED_ABS_SQR":
            arr = torch.polar(arr.abs().square(), arr.angle())
        case "PHASE" | "ANGLE":
            arr = torch.polar(torch.ones_like(arr, dtype=torch.float), arr.angle())
        case _:
            raise NotImplementedError(f"{mode=} not implemented")

    img: Image
    if arr.dim() == 2:
        padding = 0
        rgb = complex_arr_to_rgb(arr, overexpose=overexpose, window_size=window_size)[0]
        rgb = post_process_rgb(rgb, rois, window_size=window_size)
        img = ToPILImage()(rgb.squeeze())  # pyright: ignore[reportUnknownVariableType]
    elif arr.dim() == 3:
        n_images = len(arr)
        ret = [
            complex_arr_to_rgb(img, overexpose=overexpose, window_size=window_size)[0]
            for img in arr
        ]
        for i in range(len(ret)):
            ret[i] = post_process_rgb(ret[i], rois, window_size=window_size)

        nrow = math.ceil(math.sqrt(n_images))
        ncol = math.ceil(n_images / nrow)
        img = ToPILImage()(  # pyright: ignore[reportUnknownVariableType]
            make_grid(
                ret,
                nrow=nrow,
                padding=padding,
                pad_value=pad_val,
            )
        )
    else:
        raise NotImplementedError

    padding_H = padding * (ncol + 1) if ncol != 0 else 0
    window_size_H = window_size or arr.shape[-1]
    H = dx * (window_size_H * ncol + padding_H)

    padding_W = padding * (nrow + 1) if nrow != 0 else 0
    window_size_W = window_size or arr.shape[-2]
    W = dx * (window_size_W * nrow + padding_W)
    im = ax.imshow(img, extent=(0, W, 0, H))  # pyright:ignore[reportUnknownMemberType,reportUnknownArgumentType]
    if labels is not None:
        if len(labels) != n_images:
            raise Exception(
                f"labels must have the same length as the number of subimages to plot. "
                f"Found {len(labels)=}!={n_images}"
            )
        for i in range(n_images):
            _label = labels[i]
            idx_x = i % nrow
            pos_x = padding + (arr.shape[-1] + padding) * idx_x

            idx_y = (i - idx_x) / nrow
            pos_y = (arr.shape[-2] + padding) * (ncol - idx_y)

            text_offset = 0.02 * arr.shape[-1]
            pos_x += text_offset
            pos_y -= text_offset

            ax.text(pos_x * dx, pos_y * dx, _label, color="white", va="top")  # pyright:ignore[reportUnknownMemberType]

    if unit_name is not None:
        ax.set_xlabel(unit_name)  # pyright:ignore[reportUnknownMemberType]

    if show_scalebar:
        ax.set_axis_off()

        sb = ScaleBar(1, units="m", loc="lower right", border_pad=0.5)
        ax.add_artist(sb)

    return im


def post_process_rgb(
    ret: Tensor,
    rois: dict[str, Roi2D] | list[Roi2D] | None = None,
    window_size: int | None = None,
):
    if rois is None:
        rois = {}
    if isinstance(rois, list):
        rois = {str(i + 1): val for i, val in enumerate(rois)}

    if rois:
        boxes = torch.tensor(
            [
                [roi[-1].start, roi[-2].start, roi[-1].stop, roi[-2].stop]
                for roi in rois.values()
            ],
            dtype=torch.float,
        )
        ret = draw_bounding_boxes(
            ret,
            boxes,
            width=2,
            colors="black",
            labels=list(rois.keys()),
        )

    if window_size is not None:
        ret = pad_or_crop(ret, window_size)
    return ret


def get_unit_divisor(unit: str) -> float:
    """Converts unit to label and prefix.

    Args:
        unit: One of 'm', 'mm', 'um', 'nm', or float. If unit is str, then
            corresponding prefix is returned. If unit is float, then it is treated as
            the prefix and None is returned for the unit-name.

    Raises:
        Exception: If unit is unknown.

    Returns:
        unit name, unit prefix

    """

    prefixes = {
        "m": 1e0,
        "mm": 1e-3,
        "um": 1e-6,
        "nm": 1e-9,
    }
    ret = prefixes.get(unit)

    if ret is None:
        raise Exception(
            f"Unit of type {unit=} is not supported. Use one of "
            f"{list(prefixes.keys())}."
        )

    return ret


def draw_labels(
    images: torch.Tensor,
    labels: list[str],
    relative_font_size: float = 0.04,
) -> torch.Tensor:
    """Draw labels on images.

    Args:
        images: Tensor containing all frames: (N, H, W)
        labels: List of labels to be drawn onto frames
        relative_font_size: Font size relative to vertical shape of images

    Returns:
        images with labels

    """

    ret = torch.empty_like(images)

    if images.dim() != 3:
        raise Exception("Wrong dimension. Expected: (N, H, W)")

    to_pil_transform = ToPILImage("F")
    font = ImageFont.load_default(int(images.shape[-2] * relative_font_size))

    for idx, frame in enumerate(images):
        img: Image
        img = to_pil_transform(frame)  # pyright: ignore[reportUnknownVariableType]
        draw = ImageDraw.Draw(img)  # pyright: ignore[reportUnknownArgumentType]
        draw.text((0, 0), labels[idx], fill=200, font=font)
        ret[idx] = pil_to_tensor(img)

    return ret


def show_preview(
    *plotter: TPlotter,
    fig: Figure | None = None,
    align_vertical: bool = False,
) -> Figure:
    # ) -> tuple[Figure | None, tuple[Axes | None, Axes | None]]:
    """Plot preview of Probe and Object.

    Args:
        plotter: Plotter functions
        fig: Figure to be used for plotting.
        align_vertical: Align subplots vertically.

    Returns:
        Figure, (Axes of Probe, Axes of Object)

    """

    if fig is None:
        fig = plt.gcf()

    nX = 1 if align_vertical else len(plotter)
    nY = len(plotter) if align_vertical else 1

    axes = fig.subplots(nY, nX, squeeze=False)

    for i, _plotter in enumerate(plotter):
        _plotter(axes.flatten()[i])

    return fig
