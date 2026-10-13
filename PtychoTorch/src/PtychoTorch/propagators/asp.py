from functools import lru_cache, wraps

import torch
from torch._prims_common import DeviceLikeType

from PtychoTorch.utils import circ, fftfreq


def cached_wrapper(func):  # type: ignore
    return wraps(func)(lru_cache()(func))  # type: ignore


def rsq(N: int, dx: float, device: DeviceLikeType) -> torch.Tensor:
    # source plane coordinates
    x1 = torch.arange(-N // 2, N // 2, device=device) * dx
    X1, Y1 = torch.meshgrid(x1, x1, indexing="ij")
    return X1.square() + Y1.square()


def fsq(N: int, dx: float, device: DeviceLikeType | None = None) -> torch.Tensor:
    # spatial frequencies(of source plane)
    f = fftfreq(n=N, d=dx, device=device)
    FX, FY = torch.meshgrid(f, f, indexing="ij")
    return FX.square() + FY.square()


def Q1(N: int, dx: float, dq: float, device: DeviceLikeType):
    # scaling parameter
    m = dq / dx
    _rsq = rsq(N=N, dx=dx, device=device)
    return torch.exp(1.0j * 1e6 * 2 * torch.pi / 2 * (1 - m) * _rsq)


cached_Q1 = cached_wrapper(Q1)


def Q2(N: int, dx: float, dq: float, device: DeviceLikeType):
    # scaling parameter
    m = dq / dx
    # factor 1e9 comes comes because of rounding errors otherwise
    _fsq = fsq(N=N, dx=dx, device=device)
    return torch.exp(-1.0j * torch.pi**2 * 2 * 1e-9 / m / (2 * torch.pi) * _fsq)


cached_Q2 = cached_wrapper(Q2)


def Q3(N: int, dx: float, dq: float, device: DeviceLikeType):
    # scaling parameter
    m = dq / dx
    _rsq = rsq(N=N, dx=dq, device=device)
    return torch.exp(1.0j * 1e6 * 2 * torch.pi / 2 * (m - 1) / m * _rsq)


cached_Q3 = cached_wrapper(Q3)


def bandpass_filter(
    N: int,
    dx: float,
    dq: float,
    z: torch.Tensor,
    wavelength: torch.Tensor,
    device: DeviceLikeType,
) -> torch.Tensor:
    """Bandpass filter for Q2.

    Based on fracpy (circular bandpass-filter) instead of rectangular filter-function
    defined in K.Matsushima paper.

    Args:
        N: Number of pixels
        dx: Pixel size in sample-plane (m)
        dq: Pixel size in detector-plane
        wavelength: Wavelength (m)
        z: Distance to camera (m)

    Returns:
        Filter for Q2

    """
    m = dq / dx
    f_max = m * N * dx / (2 * z * wavelength)

    # Converting maximum frequency in pixel counts
    f_max_pix = f_max * dx * N

    empty = torch.empty((N, N), device=device)
    return circ(empty, f_max_pix)  # Rectangular filter


cached_bandpass_filter = cached_wrapper(bandpass_filter)


def scaledASP_quadratic_phases_dynamic_z_wavelength(
    N: int,
    dx: float,
    dq: float,
    wavelength: torch.Tensor,
    z: torch.Tensor,
    device: DeviceLikeType,
    compute_Q3: bool = False,
    bandlimit: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Calculate quadratic phases Q1, Q2.

    Here, calculation steps involving `wavelength` and `z` are performed separately to
    the remainding of the exponent. This enables caching part of the calculations, at
    cost of numeric inaccuracies. If `wavelength` and `z` are constant, please refer to
    `cached_scaledASP_quadratic_phases_dynamic` method.

    Args:
        N: Number of pixels
        dx: Pixel size in sample-plane (m)
        dq: Pixel size in detector-plane (m)
        wavelength: Wavelength (m)
        z: Distance to camera (m)
        compute_Q3: If set to true, then the phase-correcting term Q3 will also be
            calculated. If only amplitudes are of interest, Q3 can be neglected. If set
            to False, then `None` is returned for Q3.

    Returns:
        Q1, Q2, Q3

    """
    # optical wavenumber
    if wavelength.dim() == 1:
        wavelength = wavelength.view(-1, 1, 1)

    q1 = cached_Q1(N, dx, dq, device).pow(1 / wavelength * 1e-6 / z)
    q2 = cached_Q2(N, dx, dq, device).pow(wavelength * 1e9 * z)
    q3 = None
    if compute_Q3:
        q3 = cached_Q3(N, dx, dq, device)

    if bandlimit is True:
        q2 = q2 * bandpass_filter(N, dx, dq, z, wavelength, device=device)

    return q1, q2, q3


cached_scaledASP_quadratic_phases_dynamic_z_wavelength = cached_wrapper(
    scaledASP_quadratic_phases_dynamic_z_wavelength
)


def scaledASP_quadratic_phases(
    N: int,
    dx: float,
    dq: float,
    wavelength: torch.Tensor,
    z: torch.Tensor,
    compute_Q3: bool = False,
    bandlimit: bool = True,
):
    """Calculate quadratic phases Q1, Q2.

    Q1 and Q2 are calculated in a single step, improving numeric accuracies.
    However, the whole calculation is repeated when chaning wavelength/z.
    Refer to `cached_scaledASP_quadratic_phases_dynamic_z_wavelength` for a faster
    implementation.

    Args:
        N: Number of pixels
        dx: Pixel size in sample-plane (m)
        dq: Pixel size in detector-plane (m)
        wavelength: Wavelength (m)
        z: Distance to camera (m)
        compute_Q3: If set to true, then the phase-correcting term Q3 will also be
            calculated. If only amplitudes are of interest, Q3 can be neglected. If set
            to False, then `None` is returned for Q3.

    Returns:
        Q1, Q2, Q3

    """
    device = z.device
    # optical wavenumber
    if wavelength.dim() == 1:
        wavelength = wavelength.view(-1, 1, 1)

    m = dq / dx
    k = 2 * torch.pi / wavelength
    q1 = torch.exp(1.0j / 2 * (1 - m) / z * k * rsq(N, dx, device))
    q2 = torch.exp(-1.0j * torch.pi**2 * 2 * z / m / k * fsq(N, dx, device))
    q3 = None
    if compute_Q3:
        q3 = torch.exp(1j * k / 2 * (m - 1) / (m * z) * rsq(N, dq, device))

    if bandlimit is True:
        Wf = bandpass_filter(N, dx, dq, z, wavelength, device=device)
        q2 = q2 * Wf

    return q1, q2, q3


cached_scaledASP_quadratic_phases = cached_wrapper(scaledASP_quadratic_phases)
