import torch

from PtychoTorch.encoder import Encoder

from .dataset import DataSet


class TestDataSet:
    def test_copy(self):
        N_frames = 100
        H, W = 300, 300
        ptychogram = torch.rand(N_frames, H, W)
        dxd = 1e-4
        intensity_fluctuations = torch.linspace(1, 2, N_frames)
        background = torch.rand(H, W)
        barebeam = torch.rand(H, W)

        encoder = Encoder(
            positions=torch.rand(N_frames, 2),
        )
        encoder.position_correction = torch.rand(10, N_frames, 2)
        encoder.position_correction_stretch_x = torch.tensor(2.0)
        encoder.position_correction_stretch_y = torch.tensor(2.0)

        dataset = DataSet(
            ptychogram=ptychogram,
            encoder=encoder,
            camera_pixel_size=dxd,
            intensity_fluctuations=intensity_fluctuations,
            background=background,
            barebeam=barebeam,
        )

        dataset_copy = dataset.copy()

        torch.testing.assert_close(dataset.ptychogram, dataset_copy.ptychogram)
        assert id(dataset.ptychogram) != id(dataset_copy.ptychogram)

        torch.testing.assert_close(dataset.background, dataset_copy.background)
        assert id(dataset.background) != id(dataset_copy.background)

        torch.testing.assert_close(dataset.barebeam, dataset_copy.barebeam)
        assert id(dataset.barebeam) != id(dataset_copy.barebeam)

        torch.testing.assert_close(
            dataset.encoder.positions,
            dataset_copy.encoder.positions,
        )
        assert id(dataset.encoder) != id(dataset_copy.encoder)

        torch.testing.assert_close(
            dataset.encoder.positions, dataset_copy.encoder.positions
        )
        assert id(dataset.encoder.positions) != id(dataset_copy.encoder.positions)

        torch.testing.assert_close(
            dataset.encoder.position_correction_stretch_x,
            dataset_copy.encoder.position_correction_stretch_x,
        )
        assert id(dataset.encoder.position_correction_stretch_x) != id(
            dataset_copy.encoder.position_correction_stretch_x
        )

        torch.testing.assert_close(
            dataset.encoder.position_correction_stretch_y,
            dataset_copy.encoder.position_correction_stretch_y,
        )
        assert id(dataset.encoder.position_correction_stretch_y) != id(
            dataset_copy.encoder.position_correction_stretch_y
        )
