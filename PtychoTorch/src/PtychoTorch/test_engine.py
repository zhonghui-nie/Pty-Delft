import numpy as np
import torch

from PtychoTorch.propagators.base import PropagatorPlaceholder
from PtychoTorch.utils import circ

from .engine import AutoGradPIE


class TestChangeSampling:
    obj_width = 200
    feat_width = 50

    def _create_engine(self) -> AutoGradPIE:
        # Create a mock engine with a propagator

        probe = torch.randn(64, 64, dtype=torch.complex64)  # Example probe shape
        probe *= circ(probe, 15)
        obj = torch.zeros((self.obj_width, self.obj_width), dtype=torch.complex64)

        c = self.obj_width // 2
        obj[
            c - self.feat_width : c + self.feat_width,
            c - self.feat_width : c + self.feat_width,
        ] = 1.0  # Example object with a square region

        prop = PropagatorPlaceholder(
            dx=0.1,
            dq=0.1,
            z=torch.tensor(0.1),
            wavelength=torch.tensor([38e-9]),
            probe_shape=probe.shape,
        )
        engine = AutoGradPIE(
            rec_object=obj,
            rec_probe=probe,
            propagator=prop,  # pyright: ignore[reportArgumentType]
        )
        engine.log_ln("some log...")
        engine.log_summary()
        return engine

    def test_change_binning(self):
        engine = self._create_engine()
        orig_probe = engine.rec_probe.clone()
        orig_obj = engine.rec_object.clone()

        init_dx = engine.propagator.dx
        init_dq = engine.propagator.dq

        # Change binning factor to 2
        engine.change_binning(2)
        np.testing.assert_almost_equal(engine.propagator.dx, init_dx)
        np.testing.assert_almost_equal(engine.propagator.dq, init_dq * 2)

        # Change binning factor back to 1
        engine.change_binning(1)
        np.testing.assert_almost_equal(engine.propagator.dx, init_dx)
        np.testing.assert_almost_equal(engine.propagator.dq, init_dq)

        torch.testing.assert_close(engine.rec_probe, orig_probe)
        torch.testing.assert_close(engine.rec_object, orig_obj)

    def test_change_detector_crop_shape(self):
        engine = self._create_engine()
        orig_probe = engine.rec_probe.clone()
        orig_obj = engine.rec_object.clone()

        init_dx = engine.propagator.dx
        init_dq = engine.propagator.dq

        # Upsample to factor 2
        engine.change_detector_size(orig_probe.shape[-1] * 2)
        np.testing.assert_almost_equal(engine.propagator.dx, init_dx / 2)
        np.testing.assert_almost_equal(engine.propagator.dq, init_dq)

        # Downsample to orig. shape
        engine.change_detector_size(orig_probe.shape[-1])
        np.testing.assert_almost_equal(engine.propagator.dx, init_dx)
        np.testing.assert_almost_equal(engine.propagator.dq, init_dq)

        torch.testing.assert_close(engine.rec_probe, orig_probe)
        torch.testing.assert_close(engine.rec_object, orig_obj)

    def test_copy(self):
        engine = self._create_engine()

        # Copy the engine and change sampling
        copied_engine = engine.copy()

        assert engine.propagator.dx == copied_engine.propagator.dx
        assert type(engine.propagator.dx) is type(copied_engine.propagator.dx)

        assert engine.propagator.dq == copied_engine.propagator.dq
        assert type(engine.propagator.dq) is type(copied_engine.propagator.dq)

        assert engine.propagator.z == copied_engine.propagator.z
        assert type(engine.propagator.z) is type(copied_engine.propagator.z)

        torch.testing.assert_close(engine.rec_probe, copied_engine.rec_probe)
        torch.testing.assert_close(engine.rec_object, copied_engine.rec_object)
        torch.testing.assert_close(
            engine.propagator.wavelength, copied_engine.propagator.wavelength
        )

        assert id(engine.propagator) != id(copied_engine.propagator)
        assert id(engine.rec_probe) != id(copied_engine.rec_probe)
        assert id(engine.rec_object) != id(copied_engine.rec_object)

        assert engine.logging_str == copied_engine.logging_str
        assert id(engine.logging_str) != id(copied_engine.logging_str)
