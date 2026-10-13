import torch
from torchvision.transforms.functional import gaussian_blur

from .engine import AutoGradPIE


class GaussianBlurProbe:
    def __init__(
        self,
        engine: AutoGradPIE,
        kernel_size: list[int],
        sigma: list[float] | None = None,
    ):
        self.engine: AutoGradPIE = engine
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self):
        RI_probe = torch.stack(
            [self.engine.rec_probe.real, self.engine.rec_probe.imag], dim=0
        )
        blurred_probe = gaussian_blur(
            RI_probe,
            kernel_size=self.kernel_size,
            sigma=self.sigma,
        )
        self.engine.set_probe(blurred_probe[0] + 1j * blurred_probe[1])
