import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, NewType, Self

import h5py
import torch
from torch import Size
from torch._prims_common import DeviceLikeType, Tensor

from PtychoTorch.save_load import (
    H5PY_Const,
    SaveLoadHDF5,
    float_from_h5,
    get_read_hdf5_path,
    get_write_hdf5_path,
    tensor_from_h5,
)

if TYPE_CHECKING:
    from ..dataset import DataSet
else:
    DataSet = Any

from ..typing import ESW, TF, TFL, DiffPatternSet, h5_root
from ..utils import check_tensor_dims_equal, dict_to_log_str


def NA_diffraction_limit(
    dq: float,
    z: Tensor,
    wavelength: Tensor,
    n_pixels_detector: int,
) -> Tensor:
    """Calculate pixel size.

    Args:
        dq: Pixel size of detector (m)
        z: Propagation distance (m)
        n_pixel_detector: Pixels along one axis of detector

    Raises:
        Exception: Only works with squared Probe

    """
    Ld = dq * n_pixels_detector
    dx = wavelength * z / Ld
    return dx


class BasePropagator(SaveLoadHDF5, ABC):
    H5PY_ATTR_SAMPLE_PIXEL_SIZE = "dx"
    H5PY_ATTR_CAMERA_PIXEL_SIZE = "dq"
    H5PY_ATTR_DISTANCE_TO_CAMERA = "z"
    H5PY_KEY_WAVELENGTH = "wavelength"
    H5PY_KEY_PROBE_SHAPE = "probe_shape"

    def __init__(
        self,
        dx: float,
        dq: float,
        z: Tensor,
        wavelength: Tensor,
        probe_shape: torch.Size,
        device: DeviceLikeType | None = None,
    ) -> None:
        self._device = device or wavelength.device

        self.dx = dx
        self.dq = dq
        check_tensor_dims_equal(z, 0)
        self.z = self._check_tensor(z)
        self.wavelength = self._check_tensor(wavelength)
        self.probe_shape = probe_shape

    def _check_tensor(self, x: TFL) -> Tensor:
        if not torch.is_tensor(x):
            x = torch.tensor(x, device=self.device)
        if x.device != self.device:
            x = x.to(device=self.device)
        return x

    def _check_float(self, x: TFL) -> float:
        if torch.is_tensor(x):
            if x.numel() != 1:
                raise Exception("Expected only single element")
            return x.item()
        elif isinstance(x, list):
            if len(x) != 1:
                raise Exception("Expected only single element")
            return float(x[0])
        elif isinstance(x, float):
            return x
        else:
            raise TypeError(f"Expected float or tensor, got {type(x)}")

    @property
    def device(self) -> DeviceLikeType:
        return self._device

    @device.setter
    def device(self, new_device: DeviceLikeType) -> None:
        raise NotImplementedError

    @property
    def probe_shape(self) -> Size:
        """Shape of Probe."""
        return self._probe_shape

    @probe_shape.setter
    def probe_shape(self, new_probe_shape: Size):
        self._probe_shape = new_probe_shape

    @property
    def dx(self) -> float:
        """Pixelsize [m] in PO-plane."""
        return self._dx

    @dx.setter
    def dx(self, new_dx: TFL):
        self._dx = self._check_float(new_dx)

    @property
    def dq(self) -> float:
        """Pixelsize [m] in detector-plane."""
        return self._dq

    @dq.setter
    def dq(self, new_dq: float):
        self._dq = self._check_float(new_dq)

    @property
    def z(self) -> torch.Tensor:
        """Distance [m] to camera."""
        return self._z

    @z.setter
    def z(self, new_z: TFL):
        self._z = self._check_tensor(new_z)

    @property
    def wavelength(self) -> torch.Tensor:
        """Wavelength [m]"""
        return self._wavelength

    @wavelength.setter
    def wavelength(self, new_wavelength: TFL):
        self._wavelength = self._check_tensor(new_wavelength)

    @staticmethod
    @abstractmethod
    def class_propagate(
        u: ESW,
        dx: float,
        dq: float,
        z: Tensor,
        wavelength: Tensor,
        diff_pattern_set: DiffPatternSet | None = None,
    ) -> ESW:
        pass

    def propagate(
        self,
        u: ESW,
        diff_pattern_set: DiffPatternSet | None = None,
    ) -> ESW:
        return self.class_propagate(
            u=u,
            dx=self.dx,
            dq=self.dq,
            z=self.z,
            wavelength=self.wavelength,
            diff_pattern_set=diff_pattern_set,
        )

    def calc_detector_crop_shape(self, binning_factor: int) -> torch.Size:
        """Calculate shape of cropped detector, based on binning.

        Args:
            binning_factor: Binning of Engine

        Convention to read diffraction patterns is to first crop them
        (effectively changing sampling of P Object) followed by binning
        (effectively changing FOV of Probe/Object).

        """
        H, W = self.probe_shape[-2:]
        return torch.Size((H * binning_factor, W * binning_factor))

    def check_compatibility_with_dataset(
        self,
        dataset: DataSet,
    ) -> None:
        """Check if propagator is compatible with dataset."""
        if dataset.camera_pixel_size != self.dq:
            raise ValueError(
                f"camera_pixel_size {dataset.camera_pixel_size} does not match\n"
                f"propagator.dq {self.dq}"
            )

        if dataset.shape[-2:] != self.probe_shape[-2:]:
            raise ValueError(
                f"ptychogram shape {dataset.shape[-2:]} does not match\n"
                f"propagator.probe_shape {self.probe_shape[-3:]}"
            )

    def save_hdf5(self, hdf5_root: h5_root) -> None:
        if isinstance(hdf5_root, str | os.PathLike):
            hdf5_root = get_write_hdf5_path(hdf5_root)
        # check if hdf5_root is a path ('path_to_file.hdf5')
        if not isinstance(hdf5_root, h5py.Group):
            with h5py.File(hdf5_root, "w") as f:
                return self.save_hdf5(f)

        hdf5_root.attrs[H5PY_Const.ATTR_KEY_CLASS] = self.__class__.__name__
        self._save_hdf5(hdf5_root)

    def _save_hdf5(self, hdf5_root: h5py.Group) -> None:
        hdf5_root.attrs[self.H5PY_ATTR_SAMPLE_PIXEL_SIZE] = self.dx
        hdf5_root.attrs[self.H5PY_ATTR_CAMERA_PIXEL_SIZE] = self.dq
        hdf5_root.attrs[self.H5PY_ATTR_DISTANCE_TO_CAMERA] = self.z.item()

        hdf5_root.create_dataset(  # pyright: ignore[ reportUnknownMemberType ]
            self.H5PY_KEY_WAVELENGTH, data=self.wavelength.cpu().detach().numpy()
        )
        hdf5_root.create_dataset(self.H5PY_KEY_PROBE_SHAPE, data=self.probe_shape)  # pyright: ignore[ reportUnknownMemberType ]

    @classmethod
    def from_hdf5(
        cls,
        hdf5_root: h5_root,
        device: DeviceLikeType | None = None,
    ) -> Self:
        if isinstance(hdf5_root, str | os.PathLike):
            hdf5_root = get_read_hdf5_path(hdf5_root)
        # check if hdf5_root is a path ('path_to_file.hdf5')
        if not isinstance(hdf5_root, h5py.Group):
            with h5py.File(hdf5_root, "r") as f:
                return cls.from_hdf5(hdf5_root=f, device=device)
        return cls._from_hdf5(hdf5_root, device=device)

    @classmethod
    def _from_hdf5(
        cls,
        hdf5_root: h5py.Group,
        device: DeviceLikeType | None = None,
    ) -> Self:
        device = device or torch.cpu.current_device()

        dx = float_from_h5(hdf5_root.attrs, cls.H5PY_ATTR_SAMPLE_PIXEL_SIZE)
        if dx is None:
            raise Exception("dx not found in hdf5 file.")

        dq = float_from_h5(hdf5_root.attrs, cls.H5PY_ATTR_CAMERA_PIXEL_SIZE)
        if dq is None:
            raise Exception("dq not found in hdf5 file.")

        z = tensor_from_h5(hdf5_root.attrs, cls.H5PY_ATTR_DISTANCE_TO_CAMERA)
        if z is None:
            raise Exception("z not found in hdf5 file.")
        z = z.to(device)

        wavelength = tensor_from_h5(hdf5_root, cls.H5PY_KEY_WAVELENGTH)
        if wavelength is None:
            raise Exception("wavelength not found.")
        wavelength = wavelength.to(device)

        probe_shape_tens = tensor_from_h5(hdf5_root, cls.H5PY_KEY_PROBE_SHAPE)
        if probe_shape_tens is None:
            raise Exception("probe_shape not found")

        probe_shape = torch.Size(probe_shape_tens.int().tolist())  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType]
        return cls(
            dx=dx,
            dq=dq,
            z=z,
            wavelength=wavelength,
            probe_shape=probe_shape,
        )

    def __str__(self) -> str:
        title = f"Propagator: {self.__class__.__name__}"

        key_dict: dict[str, Tensor | int | float | str] = {
            "dx": self.dx,
            "dq": self.dq,
            "z": self.z,
            "wavelength": self.wavelength,
            "probe-shape": torch.tensor(self.probe_shape),
        }

        return dict_to_log_str(title, key_dict)


class BaseReflectionPropagator(BasePropagator):
    H5PY_ATTR_AOI_RAD = "AOI_rad"

    def __init__(
        self,
        dx: float,
        dq: float,
        z: torch.Tensor,
        wavelength: torch.Tensor,
        AOI_rad: torch.Tensor,
        probe_shape: torch.Size,
        device: torch.device | None = None,
    ):
        super().__init__(dx, dq, z, wavelength, probe_shape, device=device)
        check_tensor_dims_equal(AOI_rad, 0)
        self.AOI_rad = AOI_rad

    @property
    def AOI_rad(self) -> torch.Tensor:
        return self._check_tensor(self._aoi_rad)

    @AOI_rad.setter
    def AOI_rad(self, new_aoi_rad: TF):
        self._aoi_rad = self._check_tensor(new_aoi_rad)

    def get_AOI_rad(self):
        return self.AOI_rad

    def set_AOI_rad(self, new_value: TF):
        self.AOI_rad = new_value

    def get_AOI_deg(self):
        return self.AOI_rad.rad2deg()

    def set_AOI_deg(self, new_value: TF):
        self.set_AOI_rad(new_value / 180 * torch.pi)

    def get_GAOI_rad(self) -> torch.Tensor:
        return torch.pi / 2 + (-self.AOI_rad)

    def set_GAOI_rad(self, new_value: TF):
        self.set_AOI_rad(torch.pi / 2 + (-new_value))

    def get_GAOI_deg(self) -> torch.Tensor:
        return self.get_GAOI_rad().rad2deg()

    def set_GAOI_deg(self, new_value: TF):
        self.set_GAOI_rad(new_value / 180 * torch.pi)

    @staticmethod
    @abstractmethod
    def class_propagate(  # pyright: ignore[ reportIncompatibleMethodOverride ]
        u: ESW,
        dx: float,
        dq: float,
        z: torch.Tensor,
        wavelength: torch.Tensor,
        AOI_rad: torch.Tensor,
        diff_pattern_set: DiffPatternSet | None = None,
    ) -> ESW:
        """Propagation of the ESW from the Probe/Object-plane to the detector.

        Args:
            x: ESW to propagate. Dimensions: (..., H, W)

        Returns:
            ESW after propagation.

        """
        pass

    def propagate(
        self,
        u: ESW,
        diff_pattern_set: DiffPatternSet | None = None,
    ) -> torch.Tensor:
        return self.class_propagate(
            u=u,
            dx=self.dx,
            dq=self.dq,
            z=self.z,
            wavelength=self.wavelength,
            AOI_rad=self.AOI_rad,
            diff_pattern_set=diff_pattern_set,
        )

    def _save_hdf5(self, hdf5_root: h5py.Group):
        hdf5_root.attrs[self.H5PY_ATTR_AOI_RAD] = self.AOI_rad.item()
        return super()._save_hdf5(hdf5_root)

    @classmethod
    def _from_hdf5(
        cls,
        hdf5_root: h5py.Group,
        device: DeviceLikeType | None = None,
    ) -> Self:
        device = device or torch.cpu.current_device()

        dx = float_from_h5(hdf5_root.attrs, cls.H5PY_ATTR_SAMPLE_PIXEL_SIZE)
        if dx is None:
            raise Exception("dx not found in hdf5 file.")

        dq = float_from_h5(hdf5_root.attrs, cls.H5PY_ATTR_CAMERA_PIXEL_SIZE)
        if dq is None:
            raise Exception("dq not found in hdf5 file.")

        z = tensor_from_h5(hdf5_root.attrs, cls.H5PY_ATTR_DISTANCE_TO_CAMERA)
        if z is None:
            raise Exception("z not found in hdf5 file.")
        z = z.to(device)

        AOI_rad = tensor_from_h5(hdf5_root.attrs, cls.H5PY_ATTR_AOI_RAD)
        if AOI_rad is None:
            raise Exception("AOI_rad not found")

        wavelength = tensor_from_h5(hdf5_root, cls.H5PY_KEY_WAVELENGTH)
        if wavelength is None:
            raise Exception("wavelength not found.")
        wavelength = wavelength.to(device)

        probe_shape_tens = tensor_from_h5(hdf5_root, cls.H5PY_KEY_PROBE_SHAPE)
        if probe_shape_tens is None:
            raise Exception("probe_shape not found")

        probe_shape = torch.Size(probe_shape_tens.int().tolist())  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType]
        return cls(
            dx=dx,
            dq=dq,
            z=z,
            AOI_rad=AOI_rad,
            wavelength=wavelength,
            probe_shape=probe_shape,
        )

    def __str__(self):
        ret = super().__str__()
        return dict_to_log_str(
            ret,
            {"AOI (deg)": f"{self.get_AOI_deg()}"},
        )


class PropagatorPlaceholder(BasePropagator):
    @staticmethod
    def class_propagate(
        u: ESW,
        dx: float,
        dq: float,
        z: Tensor,
        wavelength: Tensor,
        diff_pattern_set: DiffPatternSet | None = None,
    ) -> ESW:
        return u


TBasePropagator = NewType("TBasePropagator", BasePropagator)
TBaseReflectionPropagator = NewType(
    "TBaseReflectionPropagator", BaseReflectionPropagator
)
