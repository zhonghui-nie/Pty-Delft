import io
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Self

import h5py
import numpy as np
import torch
from torch._prims_common import DeviceLikeType

from .typing import h5_root

if TYPE_CHECKING:
    from PtychoTorch.engine import AutoGradPIE
    from PtychoTorch.propagators.base import BasePropagator

    from .dataset import DataSet
else:
    AutoGradPIE = Any
    BasePropagator = Any
    DataSet = Any


class H5PY_Const:
    ATTR_KEY_CLASS = "base_class"


class SaveLoadHDF5(ABC):
    @abstractmethod
    def save_hdf5(self, hdf5_root: h5_root) -> None:
        """Save the object to an HDF5 file or group."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_hdf5(
        cls,
        hdf5_root: h5_root,
        device: DeviceLikeType | None = None,
    ) -> Self:
        """Load the object from an HDF5 file or group."""
        raise NotImplementedError

    def copy(self, device: DeviceLikeType | None = None) -> Self:
        """Create a copy of the object."""
        with io.BytesIO() as f:
            self.save_hdf5(f)
            return self.from_hdf5(f, device=device)


def get_read_hdf5_path(p: os.PathLike[Any] | str) -> Path:
    """Get the path to an existing HDF5 file.

    Args:
        p: The path to check, which can be a string or a Path object.

    Returns:
        Path: The existing HDF5 file path.

    Raises:
        FileNotFoundError: If the file does not exist at the specified path

    """

    p = Path(p)

    if p.exists():
        return p

    if p.suffix == "":
        if (ret := p.with_suffix(".h5")).exists():
            return ret

        if (ret := p.with_suffix(".hdf5")).exists():
            return ret

    raise FileNotFoundError(f"File '{p}' not found.")


def get_write_hdf5_path(p: os.PathLike[Any] | str) -> Path:
    """Get write path to an hdf5 file.

    Args:
        p: Path to HDF5 file used for writing

    Returns:
        Return the original path if it has an extension; otherwise, append '.h5' and
            return the new path.

    """
    p = Path(p)

    if p.suffix != "":
        return p

    return p.with_suffix(".h5")


def from_hdf5(
    hdf5_root: str | os.PathLike[Any] | IO[Any] | h5py.Group,
    device: DeviceLikeType | None = None,
) -> AutoGradPIE | BasePropagator | DataSet:
    if isinstance(hdf5_root, str | os.PathLike):
        hdf5_root = get_read_hdf5_path(hdf5_root)
    # check if hdf5_root is a path ('path_to_file.hdf5')
    if not isinstance(hdf5_root, h5py.Group):
        with h5py.File(hdf5_root, "r") as f:
            return from_hdf5(f, device=device)

    from PtychoTorch import dataset, engine, propagators

    obj_name: Any
    obj_name = hdf5_root.attrs[H5PY_Const.ATTR_KEY_CLASS]  # pyright: ignore[reportUnknownVariableType]
    if obj_name is None:
        raise Exception(f"Key '{H5PY_Const.ATTR_KEY_CLASS}' not found")

    obj_class: Any | None = None
    obj_class = obj_class or getattr(engine, str(obj_name), None)
    obj_class = obj_class or getattr(propagators, str(obj_name), None)
    obj_class = obj_class or getattr(dataset, str(obj_name), None)
    # TODO: implement load Reconstruction
    # obj_class = obj_class or getattr(sys.modules[__name__], str(obj_name), None)

    if obj_class is None:
        print(__name__)
        raise Exception(f"Baseclass '{obj_name}' not found.")

    return obj_class.from_hdf5(hdf5_root=hdf5_root, device=device)


def tensor_from_h5(
    hdf5_root: h5_root | h5py.AttributeManager,
    key: str,
) -> torch.Tensor | None:
    if isinstance(hdf5_root, str | os.PathLike):
        hdf5_root = get_read_hdf5_path(hdf5_root)
    # check if hdf5_root is a path ('path_to_file.hdf5')
    if not isinstance(hdf5_root, h5py.Group | h5py.AttributeManager):
        with h5py.File(hdf5_root, "r") as f:
            return tensor_from_h5(hdf5_root=f, key=key)

    if key not in hdf5_root:
        return None

    numpy_obj: np.ndarray | np.number | Any
    numpy_obj = hdf5_root[key][()]  # pyright: ignore

    if isinstance(numpy_obj, np.number):
        return torch.tensor(numpy_obj)
    elif isinstance(numpy_obj, np.ndarray):
        return torch.from_numpy(numpy_obj)  # pyright: ignore[reportUnknownMemberType]
    else:
        raise TypeError(f"Unsupported type for '{key=}'")


def float_from_h5(
    hdf5_root: h5_root | h5py.AttributeManager,
    key: str,
) -> float | None:
    if isinstance(hdf5_root, str | os.PathLike):
        hdf5_root = get_read_hdf5_path(hdf5_root)
    # check if hdf5_root is a path ('path_to_file.hdf5')
    if not isinstance(hdf5_root, h5py.Group | h5py.AttributeManager):
        with h5py.File(hdf5_root, "r") as f:
            return float_from_h5(hdf5_root=f, key=key)

    if key not in hdf5_root:
        return None

    numpy_obj: np.ndarray | np.number | Any
    numpy_obj = hdf5_root[key][()]  # pyright: ignore

    if isinstance(numpy_obj, np.number):
        return float(numpy_obj.item())  # pyright: ignore[reportUnknownArgumentType]
    else:
        raise TypeError(f"Unsupported type for '{key=}'")


class Reconstruction(SaveLoadHDF5):
    H5PY_KEY_CLASS_ENGINE = "engine"
    H5PY_KEY_CLASS_DATASET = "dataset"

    def __init__(
        self,
        engine: AutoGradPIE | None = None,
        dataset: DataSet | None = None,
    ):
        self.engine: AutoGradPIE | None = engine
        self.dataset: DataSet | None = dataset

    def save_hdf5(
        self,
        hdf5_root: h5_root,
        external_ptychogram: bool = False,
    ) -> None:
        if isinstance(hdf5_root, str | os.PathLike):
            hdf5_root = get_write_hdf5_path(hdf5_root)
        # check if hdf5_root is a path ('path_to_file.hdf5')
        if not isinstance(hdf5_root, h5py.Group):
            with h5py.File(hdf5_root, "w") as f:
                return self.save_hdf5(
                    hdf5_root=f,
                    external_ptychogram=external_ptychogram,
                )

        hdf5_root.attrs[H5PY_Const.ATTR_KEY_CLASS] = self.__class__.__name__

        if self.engine is not None:
            ds = hdf5_root.create_group(self.H5PY_KEY_CLASS_ENGINE)  # pyright:ignore[reportUnknownMemberType]
            self.engine.save_hdf5(ds)

        if self.dataset is not None:
            ds = hdf5_root.create_group(self.H5PY_KEY_CLASS_DATASET)  # pyright:ignore[reportUnknownMemberType]
            self.dataset.save_hdf5(ds, external_ptychogram=external_ptychogram)

    @classmethod
    def from_hdf5(
        cls,
        hdf5_root: h5_root,
        device: DeviceLikeType | None = None,
        load_engine: bool = True,
        load_dataset: bool = True,
    ) -> Self:
        if isinstance(hdf5_root, str | os.PathLike):
            hdf5_root = get_read_hdf5_path(hdf5_root)
        # check if hdf5_root is a path ('path_to_file.hdf5')
        if not isinstance(hdf5_root, h5py.Group):
            with h5py.File(hdf5_root, "r") as f:
                return cls.from_hdf5(
                    f,
                    load_engine=load_engine,
                    load_dataset=load_dataset,
                    device=device,
                )

        # moved here to prevent circular import
        from PtychoTorch.engine import AutoGradPIE

        engine: Any | None = None
        if load_engine and cls.H5PY_KEY_CLASS_ENGINE in hdf5_root:
            engine_grp = hdf5_root[cls.H5PY_KEY_CLASS_ENGINE]
            if not isinstance(engine_grp, str | h5py.Group | os.PathLike | IO):
                raise Exception("Failed to find Engine group")

            engine = from_hdf5(engine_grp, device=device)
            if not isinstance(engine, AutoGradPIE):
                raise Exception("Failed to load Engine")

        dataset: Any | None = None
        if load_dataset and cls.H5PY_KEY_CLASS_DATASET in hdf5_root:
            dataset_grp = hdf5_root[cls.H5PY_KEY_CLASS_DATASET]
            if not isinstance(dataset_grp, str | h5py.Group | os.PathLike | IO):
                raise Exception("Failed to find DataSet group")

            dataset = from_hdf5(dataset_grp, device=device)
            if not isinstance(dataset, DataSet):
                raise Exception("Failed to load DataSet")

        return cls(engine=engine, dataset=dataset)
