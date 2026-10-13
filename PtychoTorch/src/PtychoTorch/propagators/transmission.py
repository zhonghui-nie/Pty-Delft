import torch

from PtychoTorch.propagators.asp import scaledASP_quadratic_phases

from ..typing import ESW, TFL, DiffPatternSet

# from PRA.propagators.propagators_new import NA_diffraction_limit, Propagator_Base
from ..utils import fft2c, ifft2c
from .base import BasePropagator, NA_diffraction_limit


class Propagator_Fraunhofer(BasePropagator):
    @staticmethod
    def class_forward(
        u: ESW,
        dx: float,
        dq: float,
        z: torch.Tensor,
        wavelength: torch.Tensor,
        diff_pattern_set: DiffPatternSet | None = None,
    ) -> ESW:
        return fft2c(u)

    @property
    def dx(self) -> float:
        # calculate size of detector (FoV)
        Ld = self.dq * self.probe_shape[-1]
        return self.wavelength.item() * self.z.item() / Ld

    @dx.setter
    def dx(self, new_dx: TFL) -> None:
        raise NotImplementedError


class Propagator_ASP_scaled(BasePropagator):
    """Implemenation of Angular Spectrum Propagator.

    See 'Numerical simulation of optical wave propagation with examples in
    MATLAB', p.102

    """

    def __init__(
        self,
        dq: float,
        z: torch.Tensor,
        wavelength: torch.Tensor,
        probe_shape: torch.Size,
        compute_Q3: bool = False,
        bandlimit: bool = False,
        dx: float | None = None,
        device: torch.device | None = None,
    ):
        if dx is None:
            dx = (
                NA_diffraction_limit(
                    dq=dq,
                    z=z,
                    wavelength=wavelength,
                    n_pixels_detector=probe_shape[-1],
                )
                .min()
                .item()
            )

        super().__init__(
            dx=dx,
            dq=dq,
            z=z,
            wavelength=wavelength,
            probe_shape=probe_shape,
            device=device,
        )

        self.compute_Q3 = compute_Q3
        self.bandlimit = bandlimit

    # def _save_hdf5(self, hdf5_root: h5py.Group):
    #     super()._save_hdf5(hdf5_root)
    #
    #     hdf5_root.attrs["compute_Q3"] = self.compute_Q3
    #     hdf5_root.attrs["bandlimit"] = self.bandlimit
    #
    # @classmethod
    # def _from_hdf5(cls, hdf5_root: h5py.Group, device: str = "cpu"):
    #     ret = cls._from_hdf5(hdf5_root=hdf5_root, device=device)
    #     ret.compute_Q3 = hdf5_root.attrs["compute_Q3"]
    #     ret.bandlimit = hdf5_root.attrs["bandlimit"]
    #     return ret

    @staticmethod
    def class_propagate(
        u: ESW,
        dx: float,
        dq: float,
        z: torch.Tensor,
        wavelength: torch.Tensor,
        diff_pattern_set: DiffPatternSet | None = None,
        bandlimit: bool = False,
        compute_Q3: bool = False,
    ) -> ESW:
        # wavelength: either 0-dim or 1-dim with shape [N_wavelengths]
        # u.shape = [..., H, W] or
        #   for multiple wavelengths [..., 1, H, W] for non-dispersive u
        #   for multiple wavelengths [..., N_wavelengths, H, W] for dispersive u

        N = u.shape[-1]
        # Q1, Q2, Q3 = asp.scaledASP_quadratic_phases(
        Q1, Q2, Q3 = scaledASP_quadratic_phases(
            N=N,
            dx=dx,
            dq=dq,
            wavelength=wavelength,
            z=z,
            compute_Q3=compute_Q3,
            bandlimit=bandlimit,
        )

        if compute_Q3:
            if Q3 is None:
                raise Exception("Q3 is None")
            propagated_field = Q3 * ifft2c(Q2 * fft2c(Q1 * u))
        else:
            propagated_field = ifft2c(Q2 * fft2c(Q1 * u))

        return propagated_field

    def to_str_list(self) -> list[str]:
        if self.wavelength.dim() == 0:
            return [f"{self.wavelength * 1e9:.2f}"]
        else:
            return [f"{w.item() * 1e9:.2f}" for w in self.wavelength]
