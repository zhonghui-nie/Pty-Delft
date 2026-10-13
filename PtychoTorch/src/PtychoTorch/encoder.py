import os
from enum import Enum
from typing import Any, Self

import h5py
import pandas as pd
import torch
from torch._prims_common import DeviceLikeType, Tensor

from PtychoTorch.typing import h5_root
from PtychoTorch.utils import (
    check_tensor_dims_equal,
    check_tensor_one_dim,
    dict_to_log_str,
)

from .propagators.base import BasePropagator
from .save_load import (
    SaveLoadHDF5,
    get_read_hdf5_path,
    get_write_hdf5_path,
    tensor_from_h5,
)


class Encoder(SaveLoadHDF5):
    H5PY_KEY_CLASS = "encoder"
    H5PY_KEY_CLASS_CORR = "encoder_position_correction"
    H5PY_KEY_CLASS_CORR_STRETCH_X = "encoder_position_correction_stretch_x"
    H5PY_KEY_CLASS_CORR_STRETCH_Y = "encoder_position_correction_stretch_y"

    Encoder_Mode = Enum("Encoder_Mode", ["CORNER", "CENTERED"])

    def __init__(
        self,
        positions: Tensor,
        mode: Encoder_Mode = Encoder_Mode.CENTERED,
        device: DeviceLikeType | None = None,
    ):
        self._device = device or positions.device

        check_tensor_dims_equal(positions, 2)
        check_tensor_one_dim(positions, -1, 2)
        self.positions: Tensor = positions.to(self.device)
        self.set_mode(mode)

        self.position_correction: Tensor
        self.position_correction_stretch_x: Tensor
        self.position_correction_stretch_y: Tensor
        self.reset_position_correction()

    def reset_position_correction(self) -> None:
        self.position_correction = torch.zeros(
            (1, len(self), 2),
            requires_grad=True,
            device=self.device,
        )
        self.position_correction_stretch_x: Tensor = torch.tensor(
            1.0,
            device=self.device,
            requires_grad=True,
        )
        self.position_correction_stretch_y: Tensor = torch.tensor(
            1.0,
            device=self.device,
            requires_grad=True,
        )

    @property
    def device(self) -> DeviceLikeType:
        return self._device

    @device.setter
    def device(self, new_device: DeviceLikeType) -> None:
        raise NotImplementedError

    def get_mode(self) -> Encoder_Mode:
        """Get mode of encoder."""
        return self._mode

    def set_mode(self, new_mode: Encoder_Mode):
        """Set mode of encoder."""

        if new_mode not in self.Encoder_Mode:
            raise ValueError(
                f"new_mode must be one of {self.Encoder_Mode._member_names_}."
                f"Found: {new_mode=}"
            )

        self._mode = new_mode

    @classmethod
    def from_hdf5(
        cls,
        hdf5_root: h5_root,
        device: DeviceLikeType | None = None,
        mode: Encoder_Mode = Encoder_Mode.CENTERED,
    ) -> Self:
        if isinstance(hdf5_root, str | os.PathLike):
            hdf5_root = get_read_hdf5_path(hdf5_root)
        # check if hdf5_root is a path ('path_to_file.hdf5')
        if not isinstance(hdf5_root, h5py.Group):
            with h5py.File(hdf5_root, "r") as f:
                return cls.from_hdf5(f)
        device = device or torch.cpu.current_device()

        positions = tensor_from_h5(
            hdf5_root,
            cls.H5PY_KEY_CLASS,
        )
        if positions is None:
            raise Exception("Positions not found")

        ret = cls(positions=positions, mode=mode, device=device)

        pos_cor = tensor_from_h5(hdf5_root, cls.H5PY_KEY_CLASS_CORR)
        if pos_cor is not None:
            pos_cor = pos_cor.to(device)
            pos_cor.requires_grad_()
            ret.position_correction = pos_cor

        sx = tensor_from_h5(hdf5_root, cls.H5PY_KEY_CLASS_CORR_STRETCH_X)
        if sx is not None:
            sx = sx.to(device)
            sx.requires_grad_()
            ret.position_correction_stretch_x = sx

        sy = tensor_from_h5(hdf5_root, cls.H5PY_KEY_CLASS_CORR_STRETCH_Y)
        if sy is not None:
            sy = sy.to(device)
            sy.requires_grad_()
            ret.position_correction_stretch_y = sy

        return ret

    def save_hdf5(self, hdf5_root: h5_root) -> None:
        if isinstance(hdf5_root, str | os.PathLike):
            hdf5_root = get_write_hdf5_path(hdf5_root)
        # check if hdf5_root is a path ('path_to_file.hdf5')
        if not isinstance(hdf5_root, h5py.Group):
            with h5py.File(hdf5_root, "r") as f:
                return self.save_hdf5(f)

        hdf5_root.create_dataset(  # pyright: ignore[reportUnknownMemberType]
            self.H5PY_KEY_CLASS,
            data=self.positions.detach().cpu().numpy(),
        )

        hdf5_root.create_dataset(  # pyright: ignore[reportUnknownMemberType]
            self.H5PY_KEY_CLASS_CORR,
            data=self.position_correction.detach().cpu().numpy(),
        )

        hdf5_root.create_dataset(  # pyright: ignore[reportUnknownMemberType]
            self.H5PY_KEY_CLASS_CORR_STRETCH_X,
            data=self.position_correction_stretch_x.detach().cpu().cpu(),
        )

        hdf5_root.create_dataset(  # pyright: ignore[reportUnknownMemberType]
            self.H5PY_KEY_CLASS_CORR_STRETCH_Y,
            data=self.position_correction_stretch_y.detach().cpu().cpu(),
        )

    @classmethod
    def from_scanposition_csv(
        cls,
        filepath: str | os.PathLike[Any],
        mode: Encoder_Mode = Encoder_Mode.CENTERED,
        device: DeviceLikeType | None = None,
    ):
        """Returns Encoder-instance with loaded positions.

        Assumes csv-file contains positions in [um].

        Returns:
            Encoder

        """

        header_length = 0
        with open(filepath) as f:
            while line := f.readline():
                if line[0].isdigit() or line[0] == "-":
                    break

                header_length += 1

        df: pd.DataFrame = pd.read_csv(  # type: ignore
            filepath,
            skiprows=header_length,
            delimiter=" ",
            names=["y", "x"],
        )
        positions = torch.tensor(df.to_numpy()) * 1e-6  # assumes positions in um
        return cls(positions)

    def to_scanposition_csv(self, filepath: str) -> None:
        """Saves positions to csv-file.

        Saves with default stepsize of 1um.

        Args:
            filepath (str): Path to

        """
        positions_numpy = self.positions.detach().cpu().numpy()
        df = pd.DataFrame(positions_numpy / 1e-6)

        with open(filepath, "w", newline="\n") as file:
            file.write(f"number of positions: {len(df)}")
            file.write("\r\ny (row) [um] | x (col) [um]\r\n")
            df.to_csv(file, header=False, index=False, sep=" ")

    def __len__(self) -> int:
        return len(self.positions)

    def get_postition_SI(self, index: int | slice, corrected: bool = True) -> Tensor:
        if corrected is False:
            return self.positions[index]

        check_tensor_dims_equal(self.position_correction_stretch_x, 0)
        check_tensor_dims_equal(self.position_correction_stretch_y, 0)

        stretch = torch.stack(
            [
                self.position_correction_stretch_y,
                self.position_correction_stretch_x,
            ]
        )
        return self.positions[index] * stretch + self.position_correction[:, index, :]

    def generate_mask_radius(self, radius_m: float) -> torch.Tensor:
        """Generate mask based on radius.

        Args:
            radius_m (float): Set radius for datapoints to consider

        Returns:
            Booelan mask, where the indices of scanpositions within the given radius
                are set to True.

        """
        mask = torch.linalg.vector_norm(self.positions, dim=1) <= radius_m  # type:ignore
        return mask  # type:ignore

    def get_position_px(
        self,
        scan_position_SI: torch.Tensor,
        prop: BasePropagator,
        object_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        """Returns coordinates at given index in rasterized pixel-coordinates.

        Left-most and top-most scanning-positions are alligned to the left and top edge
        of the Object. Thus, the scanning-position (0,0) will in most cases not end up
        in the center of the Object.

        Args:
            scan_position_SI: Scanning position [m]. Dimensions: (N_modes, 2).
            prop: Propagator, required to get pixel size in sample-plane.
            object_shape: Shape of whole object. Required if Encoder is using
                `Encoder_Mode.CENTERED`.

        Returns:
            (N_modes, Y, X) positions of object patch in pixels

        """
        offset_px: torch.Tensor

        if self.get_mode() == self.Encoder_Mode.CORNER:
            min_position, _ = self.positions.min(dim=0)
            offset_px = min_position / prop.dx

        elif self.get_mode() == self.Encoder_Mode.CENTERED:
            if object_shape is None:
                raise ValueError("object_shape required for mode='{self.get_mode()}'")

            center_probe = (
                torch.tensor(prop.probe_shape, device=self.positions.device)[-2:] / 2
            )
            center_obj = (
                torch.tensor(object_shape, device=self.positions.device)[-2:] / 2
            )
            offset_px = torch.floor(center_probe - center_obj)

        else:
            raise NotImplementedError("Encoder mode not implemented")

        ret = scan_position_SI / prop.dx - offset_px
        return ret

    def find_minimal_object_shape_px(self, prop: BasePropagator) -> torch.Size:
        """Returns minimal shape that the object needs to have.

        Args:
            probe (torch.Tensor): Probe-Tensor.
            dx (float): Pixelsize of probe.

        Returns:
            tuple[int, int]: Minimal height and width for the object.

        """

        x_max = torch.max(self.positions[:, 1])
        x_min = torch.min(self.positions[:, 1])
        y_max = torch.max(self.positions[:, 0])
        y_min = torch.min(self.positions[:, 0])

        if self.get_mode() == self.Encoder_Mode.CORNER:
            obj_shape = torch.tensor(prop.probe_shape, dtype=torch.int)
            obj_shape[-2] = int((y_max - y_min) / prop.dx + prop.probe_shape[-2])
            obj_shape[-1] = int((x_max - x_min) / prop.dx + prop.probe_shape[-1])
            obj_shape[-2:] += 1  # add one extra pixel to deal with rounding issues

        elif self.get_mode() == self.Encoder_Mode.CENTERED:
            height = (
                torch.tensor([y_min, y_max]).abs().max() * 2 / prop.dx
                + prop.probe_shape[-2]
            )
            width = (
                torch.tensor([x_min, x_max]).abs().max() * 2 / prop.dx
                + prop.probe_shape[-1]
            )
            obj_shape = torch.stack((height, width)).ceil().to(torch.int)

        else:
            raise NotImplementedError(
                "find_minimal_object_shape_px only works for"
                "mode='Encoder_Mode.CORNER' and mode='Encoder_Mode.CENTERED'"
            )

        H, W = obj_shape[-2].item(), obj_shape[-1].item()

        return torch.Size((int(H), int(W)))

    def __str__(self) -> str:
        title = "Encoder:"
        key_dict = {"Mode": self.get_mode().name}
        return dict_to_log_str(title, key_dict)
