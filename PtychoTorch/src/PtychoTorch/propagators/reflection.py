import torch
from torch._prims_common import Tensor

from PtychoTorch.propagators.base import BaseReflectionPropagator, NA_diffraction_limit
from PtychoTorch.propagators.transmission import Propagator_ASP_scaled
from PtychoTorch.typing import ESW, DiffPatternSet
from PtychoTorch.utils import fft2c, fftfreq, ifft2c


class Matsushima_Tilt_Projection(BaseReflectionPropagator):
    def __init__(
        self,
        dx: float,
        dq: float,
        wavelength: torch.Tensor,
        AOI_rad: torch.Tensor,
        probe_shape: torch.Size,
        device: torch.device | None = None,
    ):
        super().__init__(
            dx=dx,
            dq=dq,
            z=torch.tensor(0, device=wavelength.device),
            wavelength=wavelength,
            AOI_rad=AOI_rad,
            probe_shape=probe_shape,
            device=device,
        )

    @staticmethod
    def project_spatial_frequencies(
        G: torch.Tensor,
        dx: float,
        wavelength: torch.Tensor,
        AOI_rad: torch.Tensor,
        use_jacobian: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """G(u,v): Spatial-frequencies of exit-surface-wave, e.g.: fft(u)

        Returns:
            F(u_hat, v_hat)

        """
        N_wavelength = wavelength.numel()

        u = fftfreq(G.shape[-1], d=dx, device=G.device)
        v = fftfreq(G.shape[-2], d=dx, device=G.device)
        # w = torch.sqrt(1 / wavelength**2 - u.square() - v.square())

        u_hat = fftfreq(G.shape[-1], d=dx, device=G.device)  # / u.abs().max()
        v_hat = fftfreq(G.shape[-2], d=dx, device=G.device)  # / v.abs().max()

        # U, V = torch.meshgrid((u,v), indexing="xy")
        # W = torch.sqrt(1/wavelength**2 - U.square() - V.square())

        U_hat, V_hat = torch.meshgrid([u_hat, v_hat], indexing="xy")
        W_hat = torch.sqrt(
            1 / wavelength.view(-1, 1, 1) ** 2 - U_hat.square() - V_hat.square()
        )  # / w.abs().max()

        U_hat_sampling = U_hat * torch.cos(AOI_rad) + W_hat * torch.sin(AOI_rad)
        V_hat_sampling = V_hat

        def normalize(a: Tensor, min: Tensor, max: Tensor) -> Tensor:
            return 2 * (a - min) / (max - min) - 1

        center_U_hat = 1 / wavelength * torch.sin(AOI_rad)
        # center_V_hat = 0

        U_hat_sampling = U_hat_sampling - center_U_hat.view(-1, 1, 1)
        # V_hat_sampling = V_hat_sampling - center_V_hat

        U_hat_sampling = U_hat_sampling / (1 / dx)
        V_hat_sampling = V_hat_sampling / (1 / dx)

        U_hat_sampling = normalize(U_hat_sampling, u.min() * dx, u.max() * dx)
        V_hat_sampling = normalize(V_hat_sampling, v.min() * dx, v.max() * dx)

        # self.U_hat_sampling_min = U_hat_sampling.min()
        # self.U_hat_sampling_max = U_hat_sampling.max()
        # self.V_hat_sampling_min = V_hat_sampling.min()
        # self.V_hat_sampling_max = V_hat_sampling.max()

        grid = torch.stack(
            (U_hat_sampling, V_hat_sampling.expand_as(U_hat_sampling)), -1
        )
        G_ri = torch.stack((G.real, G.imag))  # shape: [2, ..., N_wavelength, H, W]

        G_ri = G_ri.movedim(-3, 0)  # shape: [N_wavelength, 2, ..., H, W]
        original_shape = G_ri.shape

        G_ri = G_ri.view(
            N_wavelength, -1, *G_ri.shape[-2:]
        )  # shape: [N_wavelength, -1, H, W]

        F_ri = torch.nn.functional.grid_sample(G_ri, grid).view(original_shape)

        F_ri = F_ri.movedim(0, -3)  # shape: [2, ..., wavelength, H, W]

        ret = torch.complex(F_ri[0], F_ri[1])

        if use_jacobian:
            J = torch.abs(torch.cos(AOI_rad) - U_hat / W_hat * torch.sin(AOI_rad))
            ret = ret * J
        return ret, grid

    @staticmethod
    def get_esw_FT_mask(
        probe_shape: torch.Size,
        dx: float,
        wavelength: torch.Tensor,
        AOI_rad: torch.Tensor,
    ) -> torch.Tensor:
        # only works for 1 wavelength only!

        _, g = Matsushima_Tilt_Projection.project_spatial_frequencies(
            G=torch.empty(probe_shape, device=AOI_rad.device, dtype=torch.complex64),
            dx=dx,
            wavelength=wavelength,
            AOI_rad=AOI_rad,
            use_jacobian=False,
        )

        x = torch.linspace(-1, 1, g.shape[-2], device=g.device)
        y = torch.linspace(-1, 1, g.shape[-3], device=g.device)
        XX, _ = torch.meshgrid([x, y], indexing="xy")

        dimension_scaling = torch.tensor(probe_shape[-2:], device=AOI_rad.device).T - 1
        g_index = ((g + 1) / 2).view(-1, 2) * dimension_scaling
        g_index = torch.vstack((g_index.floor(), g_index.ceil()))

        g_index_x = g_index[:, 0].flatten().to(torch.long)
        g_index_y = g_index[:, 1].flatten().to(torch.long)

        mask = torch.ones_like(XX)

        mask[g_index_y, g_index_x] = 0

        return mask

    @staticmethod
    def class_propagate(  # pyright: ignore[ reportIncompatibleMethodOverride ]
        u: ESW,
        dx: float,
        wavelength: torch.Tensor,
        AOI_rad: torch.Tensor,
        use_jacobian: bool = False,
        diff_pattern_set: DiffPatternSet | None = None,
    ) -> ESW:
        # wavelength: either 0-dim or 1-dim with shape [N_wavelengths]
        # u.shape = [..., H, W] or
        #   for multiple wavelengths [..., 1, H, W] for non-dispersive u
        #   for multiple wavelengths [..., N_wavelengths, H, W] for dispersive u

        # kx = fftfreq(u.shape[-1], dx, device=u.device) * 2 * torch.pi
        # ky = fftfreq(u.shape[-2], dx, device=u.device) * 2 * torch.pi

        # Kx, Ky = torch.meshgrid((kx, ky), indexing="xy")

        u_fft = fft2c(u)
        u_fft, _ = Matsushima_Tilt_Projection.project_spatial_frequencies(
            u_fft,
            dx=dx,
            wavelength=wavelength,
            AOI_rad=AOI_rad,
            use_jacobian=use_jacobian,
        )
        propagated_field = ifft2c(u_fft)

        return propagated_field


class ASP_TC3(BaseReflectionPropagator):
    def __init__(
        self,
        dq: float,
        z: torch.Tensor,
        wavelength: torch.Tensor,
        AOI_rad: torch.Tensor,
        probe_shape: torch.Size,
        dx: float | None = None,
        device: torch.device | None = None,
    ):
        if dx is None:
            z = z.to(device=wavelength.device)
            dx = (
                NA_diffraction_limit(
                    dq=dq, z=z, wavelength=wavelength, n_pixels_detector=probe_shape[-1]
                )
                .min()
                .item()
            )
        super().__init__(dx, dq, z, wavelength, AOI_rad, probe_shape, device=device)

    def get_esw_FT_mask(self) -> torch.Tensor:
        return Matsushima_Tilt_Projection.get_esw_FT_mask(
            probe_shape=self.probe_shape,
            dx=self.dx,
            wavelength=self.wavelength,
            AOI_rad=self.AOI_rad,
        )

    @staticmethod
    def class_propagate(
        u: ESW,
        dx: float,
        dq: float,
        z: torch.Tensor,
        wavelength: torch.Tensor,
        AOI_rad: torch.Tensor,
        diff_pattern_set: DiffPatternSet | None = None,
    ) -> ESW:
        interm_field = Matsushima_Tilt_Projection.class_propagate(
            u=u,
            dx=dx,
            wavelength=wavelength,
            AOI_rad=AOI_rad,
        )

        propagated_field = Propagator_ASP_scaled.class_propagate(
            u=interm_field,
            dx=dx,
            dq=dq,
            z=z,
            wavelength=wavelength,
            diff_pattern_set=diff_pattern_set,
            compute_Q3=False,
        )

        return propagated_field
