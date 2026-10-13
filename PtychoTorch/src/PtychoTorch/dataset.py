import os
from typing import Any, Self

import h5py
import torch
from torch._prims_common import DeviceLikeType, Tensor

from PtychoTorch.typing import DiffPattern, DiffPatternSet, h5_root

from .encoder import Encoder
from .save_load import (
    SaveLoadHDF5,
    get_read_hdf5_path,
    get_write_hdf5_path,
    tensor_from_h5,
)
from .utils import check_tensor_dims_equal, pad_or_crop


class DataSet(SaveLoadHDF5, torch.utils.data.Dataset[DiffPatternSet]):
    H5PY_KEY_PTYCHOGRAM = "ptychogram"
    H5PY_KEY_BACKGROUND = "background"
    H5PY_KEY_INTENSITY_FLUCT = "intensity_fluctuations"
    H5PY_KEY_BAREBEAM = "barebeam"
    H5PY_KEY_CAMERA_PIXEL_SIZE = "dxd"

    def __init__(
        self,
        ptychogram: Tensor,
        encoder: Encoder,
        camera_pixel_size: float,
        intensity_fluctuations: Tensor | None = None,
        background: DiffPattern | None = None,
        barebeam: DiffPattern | None = None,
        device: DeviceLikeType | None = None,
    ) -> None:
        self._device = device or torch.cpu.current_device()

        self.h5_groups: dict[str, Tensor] = {}
        self.root_attrs: dict[str, Any] = {}

        check_tensor_dims_equal(ptychogram, 3)
        self.ptychogram: Tensor = ptychogram.to(self.device)
        self.encoder: Encoder = encoder
        self.camera_pixel_size: float = camera_pixel_size
        if intensity_fluctuations is not None:
            self.intensity_fluctuations: Tensor = intensity_fluctuations
        else:
            self.intensity_fluctuations: Tensor = intensity_fluctuations or torch.ones(
                len(ptychogram),
                device=self.device,
                dtype=torch.float,
                requires_grad=True,
            )

        self.background: DiffPattern | None = None
        if background is not None:
            self.set_background(background)

        self.barebeam: DiffPattern | None = None
        if barebeam is not None:
            self.set_barebeam(barebeam)

        self.is_barebeam: bool = False
        self._crop_shape: torch.Size = ptychogram.shape[-2:]
        self._binning_factor: int = 1

    def get_binning_factor(self) -> int:
        return self._binning_factor

    def set_binning_factor(self, new_binning_factor: int) -> None:
        old_binning_factor = self.get_binning_factor()
        dxd = self.camera_pixel_size / old_binning_factor * new_binning_factor
        self.camera_pixel_size = dxd
        self._binning_factor = new_binning_factor

    def get_crop_shape(self) -> torch.Size:
        return self._crop_shape

    def set_crop_shape(self, new_detector_crop_shape: torch.Size) -> None:
        self._crop_shape = new_detector_crop_shape

    def reset_crop_shape(self) -> None:
        self.set_crop_shape(self.ptychogram.shape[-2:])

    @property
    def shape(self) -> torch.Size:
        N_frames = self.ptychogram.shape[-3]
        H, W = self._crop_shape
        binning_factor = self.get_binning_factor()
        return torch.Size((N_frames, H // binning_factor, W // binning_factor))

    @staticmethod
    def crop_diffpattern(
        frame: DiffPattern,
        crop_shape: torch.Size | tuple[int, int] | int | None = None,
        binning_factor: int = 1,
    ) -> DiffPattern:
        check_tensor_dims_equal(frame, 2)
        if crop_shape is not None:
            frame = pad_or_crop(frame, crop_shape)
        if binning_factor > 1:
            frame = torch.nn.functional.avg_pool2d(
                frame.unsqueeze(0).unsqueeze(0), kernel_size=binning_factor
            ).squeeze()
        return frame

    @classmethod
    def from_hdf5(
        cls,
        hdf5_root: h5_root,
        device: DeviceLikeType | None = None,
        camera_pixel_size: float | str | None = None,
    ) -> Self:
        if isinstance(hdf5_root, str | os.PathLike):
            hdf5_root = get_read_hdf5_path(hdf5_root)
        # check if hdf5_root is a path ('path_to_file.hdf5')
        if not isinstance(hdf5_root, h5py.Group):
            with h5py.File(hdf5_root, "r") as f:
                return cls.from_hdf5(f, device=device)

        background = tensor_from_h5(hdf5_root, cls.H5PY_KEY_BACKGROUND)
        barebeam = tensor_from_h5(hdf5_root, cls.H5PY_KEY_BAREBEAM)
        intensity_fluct = tensor_from_h5(hdf5_root, cls.H5PY_KEY_INTENSITY_FLUCT)

        dxd: float
        if isinstance(camera_pixel_size, float):
            dxd = camera_pixel_size
        elif isinstance(camera_pixel_size, str | None):
            _pixel_size_key = camera_pixel_size or cls.H5PY_KEY_CAMERA_PIXEL_SIZE
            dxd_T = tensor_from_h5(hdf5_root, _pixel_size_key)
            if dxd_T is not None:
                dxd = float(dxd_T.to(dtype=torch.float).item())
            else:
                raise ValueError(
                    "Camera pixel size not found in hdf5 file. Please provide a\n"
                    "valid key or define camera_pixel_size as a float explicitly."
                )
        else:
            raise ValueError(
                f"camera_pixel_size must be a float or str or None, but got\n"
                f"{type(camera_pixel_size)}"
            )

        ptychogram = tensor_from_h5(hdf5_root, cls.H5PY_KEY_PTYCHOGRAM)
        if ptychogram is None:
            raise Exception("Ptychogram not found")

        encoder = Encoder.from_hdf5(hdf5_root, device=device)

        ret = cls(
            ptychogram=ptychogram,
            encoder=encoder,
            camera_pixel_size=dxd,
            intensity_fluctuations=intensity_fluct,
            background=background,
            barebeam=barebeam,
            device=device,
        )

        for key, val in hdf5_root.attrs.items():  # type: ignore
            ret.root_attrs[key] = val

        for key in list(hdf5_root):  # type:ignore
            if not isinstance(key, str):
                continue
            if (tens := tensor_from_h5(hdf5_root=hdf5_root, key=key)) is not None:
                ret.h5_groups[key] = tens

        return ret

    @property
    def device(self) -> DeviceLikeType:
        return self._device

    @device.setter
    def device(self, new_device: DeviceLikeType) -> None:
        raise NotImplementedError

    def get_frame(self, index: int) -> DiffPattern:
        if index >= len(self):
            raise IndexError(
                f"Index {index} out of bounds for dataset of length {len(self)}"
            )
        return self.crop_diffpattern(
            self.ptychogram[index],
            crop_shape=self._crop_shape,
            binning_factor=self.get_binning_factor(),
        )

    def set_background(self, new_background: DiffPattern) -> None:
        if new_background.shape != self.ptychogram.shape[-2:]:
            raise ValueError(
                f"Background shape {new_background.shape} does not match\n"
                "ptychogram shape {self.ptychogram.shape[-2:]}"
            )
        new_background = new_background.to(self.device)
        new_background.requires_grad_()
        self.background = new_background

    def get_background(self) -> DiffPattern | None:
        if self.background is None:
            return None
        return self.crop_diffpattern(
            self.background,
            crop_shape=self._crop_shape,
            binning_factor=self.get_binning_factor(),
        )

    def set_barebeam(self, new_barebeam: DiffPattern) -> None:
        if new_barebeam.shape != self.ptychogram.shape[-2:]:
            raise ValueError(
                f"Barebeam shape {new_barebeam.shape} does not match ptychogram\n"
                "shape {self.ptychogram.shape[-2:]}"
            )
        new_barebeam = new_barebeam.to(self.device)
        new_barebeam.requires_grad_()
        self.barebeam = new_barebeam

    def get_barebeam(self) -> DiffPatternSet | None:
        if self.barebeam is None:
            return None

        barebeam_frame = self.crop_diffpattern(
            self.barebeam,
            crop_shape=self._crop_shape,
            binning_factor=self.get_binning_factor(),
        )

        return DiffPatternSet(
            diff_pattern=barebeam_frame,
            scan_position_SI=None,
            scan_position_px=None,
            is_barebeam=True,
            background=self.get_background(),
        )

    def __len__(self) -> int:
        return len(self.ptychogram)

    def __getitem__(self, index: int) -> DiffPatternSet:
        if self.intensity_fluctuations.shape[0] != len(self):
            raise Exception("intensity_fluctuations shape does not match ptychogram")

        frame = self.get_frame(index)
        diffraction_pattern = frame * self.intensity_fluctuations[index]

        return DiffPatternSet(
            diff_pattern=diffraction_pattern,
            scan_position_SI=self.encoder.get_postition_SI(index),
            scan_position_px=None,
            is_barebeam=self.is_barebeam,
            background=self.get_background(),
        )

    def save_hdf5(self, hdf5_root: h5_root, external_ptychogram: bool = False) -> None:
        if isinstance(hdf5_root, str | os.PathLike):
            hdf5_root = get_write_hdf5_path(hdf5_root)
        # check if hdf5_root is a path ('path_to_file.hdf5')
        if not isinstance(hdf5_root, h5py.Group):
            with h5py.File(hdf5_root, "w") as f:
                return self.save_hdf5(f, external_ptychogram=external_ptychogram)

        if self.background is not None:
            hdf5_root.create_dataset(  # pyright: ignore[reportUnknownMemberType]
                self.H5PY_KEY_BACKGROUND, data=self.background.detach().cpu().numpy()
            )

        if self.barebeam is not None:
            hdf5_root.create_dataset(  # pyright: ignore[reportUnknownMemberType]
                self.H5PY_KEY_BAREBEAM, data=self.barebeam.detach().cpu().numpy()
            )

        hdf5_root.create_dataset(  # pyright: ignore[reportUnknownMemberType]
            self.H5PY_KEY_INTENSITY_FLUCT,
            data=self.intensity_fluctuations.detach().cpu().numpy(),
        )

        hdf5_root.create_dataset(  # pyright: ignore[reportUnknownMemberType]
            self.H5PY_KEY_CAMERA_PIXEL_SIZE, data=self.camera_pixel_size
        )

        hdf5_root.create_dataset(  # pyright: ignore[reportUnknownMemberType]
            self.H5PY_KEY_PTYCHOGRAM, data=self.ptychogram.detach().cpu().numpy()
        )

        self.encoder.save_hdf5(hdf5_root)
