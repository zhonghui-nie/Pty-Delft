import torch

from PtychoTorch.dataset import DataSet
from PtychoTorch.encoder import Encoder
from PtychoTorch.engine import AutoGradPIE
from PtychoTorch.propagators.base import PropagatorPlaceholder

from .utils import change_binning, change_cropping, fft2c, ifft2c


def test_fft2c_ifft2c():
    E_in = torch.rand(5, 100, 100, dtype=torch.complex64) - 0.5 - 0.5j
    torch.testing.assert_close(ifft2c(fft2c(E_in)), E_in)
    torch.testing.assert_close(fft2c(ifft2c(E_in)), E_in)


def test_binning_cropping():
    N_frames = 10
    probe_shape = torch.Size((2048, 2048))
    obj_shape = torch.Size((4500, 4500))
    ptycho = torch.rand(N_frames, *probe_shape)
    dxd = 1e-5
    enc = Encoder(torch.rand(N_frames, 2))
    ds = DataSet(
        ptychogram=ptycho,
        encoder=enc,
        camera_pixel_size=dxd,
    )

    eng = AutoGradPIE(
        torch.rand(3, 3, *probe_shape, dtype=torch.complex64),
        torch.rand(3, *obj_shape, dtype=torch.complex64),
        propagator=PropagatorPlaceholder(
            dx=dxd,
            dq=dxd,
            z=torch.tensor(10e-2),
            wavelength=torch.tensor(38e-9),
            probe_shape=probe_shape,
        ),  # pyright: ignore[reportArgumentType]
    )

    change_binning(eng, ds, 2)
    assert eng.propagator.dx == dxd
    assert eng.propagator.dq == 2 * dxd
    assert ds.camera_pixel_size == 2 * dxd
    _obj_sh = eng.rec_object.shape

    change_cropping(eng, ds, 2048)
    assert eng.propagator.dx == dxd
    assert eng.propagator.dq == 2 * dxd
    assert ds.camera_pixel_size == 2 * dxd
    assert eng.rec_object.shape == _obj_sh

    change_binning(eng, ds, 1)
    assert eng.propagator.dx == dxd
    assert eng.propagator.dq == dxd
    assert ds.camera_pixel_size == dxd

    change_cropping(eng, ds, 1024)
    assert eng.propagator.dx == 2 * dxd
    assert eng.propagator.dq == dxd
    assert ds.camera_pixel_size == dxd

    change_binning(eng, ds, 2)
    assert eng.propagator.dx == 2 * dxd
    assert eng.propagator.dq == 2 * dxd
    assert ds.camera_pixel_size == 2 * dxd

    change_cropping(eng, ds, 2048)
    change_binning(eng, ds, 1)
    assert eng.propagator.dx == dxd
    assert eng.propagator.dq == dxd
    assert ds.camera_pixel_size == dxd

    assert eng.propagator.probe_shape[-2:] == probe_shape
    assert eng.rec_probe.shape[-2:] == probe_shape
    assert eng.rec_object.shape[-2:] == obj_shape

    # TODO: Check if objects and probe are interpolated/binned correctly,
    # not just pixel sizes
