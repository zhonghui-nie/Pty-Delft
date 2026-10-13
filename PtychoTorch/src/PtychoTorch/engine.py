import math
import os
from collections.abc import Callable
from typing import IO, Any, Self

import h5py
import numpy as np
import torch
import tqdm
from matplotlib.axes import Axes
from torch._prims_common import DeviceLikeType, Tensor

from PtychoTorch import decorators
from PtychoTorch.dataset import DataSet
from PtychoTorch.encoder import Encoder
from PtychoTorch.optimizer import optimizer_to_str
from PtychoTorch.plotting import plot_hsv
from PtychoTorch.propagators.base import TBasePropagator
from PtychoTorch.typing import Roi2D
from PtychoTorch.utils import (
    check_tensor_dims_equal,
    check_tensor_one_dim,
    get_slice,
    interpolate_fourier,
    pad_or_crop,
    shift_img_patch,
)

from .dataloader import DataLoader
from .loss_functions import ESW_to_intensity, LossFunction
from .propagators import BasePropagator
from .save_load import (
    SaveLoadHDF5,
    from_hdf5,
    get_read_hdf5_path,
    get_write_hdf5_path,
    tensor_from_h5,
)
from .typing import ESW, TFL, DiffPatternSet, Object, Probe, h5_root


class AutoGradPIE(SaveLoadHDF5):
    H5PY_ATTR_KEY_CLASS = "base_class"
    H5PY_ATTR_BINNING = "binning"
    H5PY_ATTR_EPOCH_CTR = "epoch_counter"

    H5PY_KEY_PROBE = "probe"
    H5PY_KEY_OBJECT = "object"
    H5PY_KEY_PROPAGATOR = "propagator"
    H5PY_KEY_RECONSTRUCTION_LOG = "training/reconstruction_log"
    # H5PY_KEY_TRAINING = "training"
    H5PY_KEY_TRAINING_LOSS = "training/training_loss"
    # H5PY_KEY_SNAPSHOTS = "snapshots"

    def __init__(
        self,
        rec_probe: Tensor,
        rec_object: Tensor,
        propagator: TBasePropagator,
        device: DeviceLikeType | None = None,
    ) -> None:
        self.device = device or torch.cpu.current_device()

        self.propagator: TBasePropagator = propagator.copy(self.device)
        self.rec_object: Tensor
        self.set_object(rec_object)

        self.rec_probe: Tensor
        self.set_probe(rec_probe)

        self.device = device or torch.cpu.current_device()

        self.epoch_counter: int = 0
        self.binning_factor: int = 1
        self.is_reconstructing: bool = False
        self.logging_str: str = ""
        self.loss: list[float] = []
        self._backup_snapshot: Self | None = None

    def restore_backup_snapshot(self) -> Self:
        if self._backup_snapshot is None:
            raise ValueError("Backup Snapshot not set.")
        return self._backup_snapshot.copy()

    def create_backup_snapshot(self) -> None:
        self._backup_snapshot = self.copy()

    def log_ln(self, msg: str) -> None:
        self.logging_str += f"{msg}\n"

    def log_sep(self) -> None:
        self.log_ln("-" * 50)

    def log_summary(self) -> None:
        self.log_ln(str(self.propagator))
        self.log_ln(f"Binning={self.binning_factor}")
        self.log_ln(f"Epoch={self.epoch_counter}")

    def _check_tensor(self, x: TFL) -> Tensor:
        if not torch.is_tensor(x):
            x = torch.tensor(x, device=self.device)
        if x.device != self.device:
            x = x.to(device=self.device)
        return x

    def set_probe(self, rec_probe: Probe | np.ndarray | Any) -> None:
        """Replace probe of Engine.

        Calling this method makes sure that the new probe is on the correct device,
        requires gradient, checks for dtype and updates Engine.propagator.probe_shape.

        Args:
            probe: New probe

        """
        if isinstance(rec_probe, np.ndarray):
            self.rec_probe = torch.from_numpy(rec_probe).to(device=self.device)  # type:ignore
        elif not torch.is_tensor(rec_probe):
            self.rec_probe = torch.tensor(rec_probe, device=self.device)
        else:
            self.rec_probe = rec_probe.detach().clone().to(self.device)

        if self.rec_probe.device != self.device:
            self.rec_probe = self.rec_probe.to(self.device)

        if self.rec_probe.grad is None:
            self.rec_probe.grad = torch.zeros_like(self.rec_probe)

        self.rec_probe.requires_grad_()

        if self.rec_probe.dtype != torch.complex64:
            raise Exception(f"{self.rec_probe.dtype} != torch.complex64")

        self.propagator.probe_shape = self.rec_probe.shape

    def set_object(self, rec_object: Object | np.ndarray | Any) -> None:
        """Replace object of Engine.

        Calling this method makes sure that the new probe is on the correct device and
        requires gradient.

        Args:
            object: New Object

        """
        if isinstance(rec_object, np.ndarray):
            self.rec_object = torch.from_numpy(rec_object).to(device=self.device)  # type:ignore
        elif not torch.is_tensor(rec_object):
            self.rec_object = torch.tensor(rec_object, device=self.device)
        else:
            self.rec_object = rec_object.detach().clone().to(self.device)

        if self.rec_object.device != self.device:
            self.rec_object = self.rec_object.to(self.device)

        if self.rec_object.grad is None:
            self.rec_object.grad = torch.zeros_like(self.rec_object)

        self.rec_object.requires_grad_()

    def set_propagator(self, propagator: TBasePropagator) -> None:
        # TODO: check binning factor, detector_shape

        if self.propagator.dx != propagator.dx:
            raise Exception("propagator.dx not the same")

        if self.propagator.dq != propagator.dq:
            raise Exception("propagator.dq not the same")

        self.propagator = propagator.copy(device=self.device)

    def get_object_patch(self, position_px: torch.Tensor) -> Object:
        check_tensor_dims_equal(position_px, 2)
        check_tensor_one_dim(position_px, -1, 2)

        pos0 = position_px[0]  # get exact position of first decomposition
        pos0_approx = (int(pos0[-2].item()), int(pos0[-1].item()))
        roi = get_slice(pos0_approx, self.rec_probe.shape)

        shift_px = position_px - torch.tensor(pos0_approx, device=pos0.device)
        object_patch = self.rec_object[..., *roi]
        object_patch = shift_img_patch(object_patch, shift_px)

        if object_patch.shape[-2:] != self.rec_probe.shape[-2:]:
            raise Exception(
                "Shape of ESW does not match shape of Probe."
                "Check if Object is large enough."
            )

        return object_patch

    def calculate_esw(
        self,
        position_px: torch.Tensor,
        is_barebeam: bool = False,
    ) -> tuple[ESW, Object | None]:
        if is_barebeam is True:
            return self.rec_probe, None

        object_patch = self.get_object_patch(position_px)
        esw = self.rec_probe * object_patch
        return esw, object_patch

    def forward(
        self,
        loss_function: LossFunction,
        diff_pattern_set: DiffPatternSet,
    ) -> torch.Tensor:
        if diff_pattern_set.scan_position_px is None:
            raise Exception("scan_position_px is None")

        esw, object_patch = self.calculate_esw(
            position_px=diff_pattern_set.scan_position_px,
            is_barebeam=diff_pattern_set.is_barebeam,
        )

        esw_detector = self.propagator.propagate(
            u=esw,
            diff_pattern_set=diff_pattern_set,
        )

        loss = loss_function(
            diff_pattern_set,
            esw_detector,
            probe=self.rec_probe,
            object_patch=object_patch,
        )
        return loss

    def simulate(
        self,
        x: Encoder,
        is_barebeam: bool = False,
        device: DeviceLikeType | None = None,
    ) -> DataSet:
        device = device or self.device
        n_pos = len(x)
        ptychoram = torch.empty((n_pos, *self.rec_probe.shape[-2:]), device=device)

        for i in range(n_pos):
            pos_si = x.get_postition_SI(i)
            pos_px = x.get_position_px(
                pos_si,
                self.propagator,
                object_shape=self.rec_object.shape,
            )

            diff_pattern_set = DiffPatternSet(
                None,  # pyright: ignore
                scan_position_SI=pos_si,
                scan_position_px=pos_px,
                is_barebeam=is_barebeam,
            )

            esw, _ = self.calculate_esw(pos_px, is_barebeam=is_barebeam)
            esw_detector = self.propagator.propagate(esw, diff_pattern_set)
            I_estimated = ESW_to_intensity(esw_detector, diff_pattern_set)

            ptychoram[i] = I_estimated

        return DataSet(
            ptychogram=ptychoram,
            encoder=x,
            camera_pixel_size=self.propagator.dq,
            device=ptychoram.device,
        )

    @decorators.is_reconstruct
    def reconstruct(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_func: LossFunction,
        N_epoch: int,
        hooks_after_step: list[Callable[[], None]] | None = None,
        hooks_after_epoch: list[Callable[[], None]] | None = None,
    ) -> None:
        self.create_backup_snapshot()
        self.log_sep()
        self.log_ln("Start new Reconstruction")
        self.log_summary()
        self.log_ln(optimizer_to_str(optimizer))
        self.log_ln(f"Loss-Function: {loss_func}")

        pbar = tqdm.trange(N_epoch, leave=True)
        epoch_loss: float = torch.inf

        N_positions = len(train_loader)

        # N_pixels is used for normalization, s.t. loss is independent of binning
        N_pixels: int = self.rec_probe.numel()

        for _ in pbar:
            pbar.set_description(
                f"{self.__class__.__name__}:epoch_loss: {epoch_loss:.6e}"
            )
            epoch_loss: float = 0.0
            optimizer.zero_grad()

            train_iter = train_loader.generator(self.propagator, self.rec_object.shape)
            for _step_ctr, diff_pattern_set_batch in enumerate(train_iter):
                batch_size = len(diff_pattern_set_batch)

                for diff_pattern_set in diff_pattern_set_batch:
                    loss = self.forward(loss_func, diff_pattern_set)
                    loss.backward()  # type: ignore
                    epoch_loss += loss.item() / N_positions / N_pixels / batch_size

                optimizer.step()
                optimizer.zero_grad()

                with torch.no_grad():
                    for h in hooks_after_step or []:
                        h()

            with torch.no_grad():
                for h in hooks_after_epoch or []:
                    h()

            if math.isnan(epoch_loss):
                raise Exception("loss is NaN")

            self.epoch_counter += 1

    def change_binning(self, new_binning_factor: int) -> None:
        old_binning_factor = self.binning_factor
        dxd_unbinned = self.propagator.dq / old_binning_factor

        H, W = self.rec_probe.shape[-2:]
        old_probe_shape = (H, W)

        H_new = (H * old_binning_factor) / new_binning_factor
        W_new = (W * old_binning_factor) / new_binning_factor
        if H_new % 1 != 0 or W_new % 1 != 0:
            raise Exception(
                f"new_binning_factor {new_binning_factor} not compatible with "
                f"probe shape {old_probe_shape}"
            )
        new_probe_shape = torch.Size((int(H_new), int(W_new)))

        new_probe = pad_or_crop(
            self.rec_probe.detach().clone(),
            new_shape=new_probe_shape,
        )

        H_obj, W_obj = self.rec_object.shape[-2:]
        H_obj += new_probe_shape[-2] - old_probe_shape[-2]
        W_obj += new_probe_shape[-1] - old_probe_shape[-1]
        new_object_shape = (H_obj, W_obj)

        new_object = pad_or_crop(
            self.rec_object.detach().clone(),
            new_shape=new_object_shape,
        )

        # actually doing changes.
        self.binning_factor = new_binning_factor
        self.propagator.dq = dxd_unbinned * new_binning_factor
        self.propagator.probe_shape = new_probe_shape
        self.set_probe(new_probe)
        self.set_object(new_object)

    def change_detector_size(self, new_probe_shape1D: int) -> None:
        """Only squared probe-shapes allowed!"""
        if self.rec_probe.shape[-2] != self.rec_probe.shape[-1]:
            raise NotImplementedError("Only works for squared probe shapes")

        old_probe_shape1D = self.rec_probe.shape[-1] * self.binning_factor
        old_probe_shape = torch.Size((old_probe_shape1D, old_probe_shape1D))
        new_probe_shape = torch.Size(
            (
                new_probe_shape1D // self.binning_factor,
                new_probe_shape1D // self.binning_factor,
            )
        )

        old_pixel_size = self.propagator.dx
        new_pixel_size = old_pixel_size * old_probe_shape1D / new_probe_shape1D

        H_obj, W_obj = self.rec_object.shape[-2:]
        H_obj = math.ceil(round(H_obj * old_pixel_size / new_pixel_size, 8))
        W_obj = math.ceil(round(W_obj * old_pixel_size / new_pixel_size, 8))
        new_obj_shape = torch.Size((H_obj, W_obj))

        new_object = interpolate_fourier(
            self.rec_object,
            new_obj_shape,
        ).to(dtype=self.rec_object.dtype)

        new_probe = interpolate_fourier(
            self.rec_probe,
            new_probe_shape,
        ).to(dtype=self.rec_probe.dtype)

        intensity_scaling = new_probe_shape.numel() / old_probe_shape.numel()

        # TODO: check if object also requires intensity scaling
        new_probe /= math.sqrt(intensity_scaling)

        self.propagator.dx = new_pixel_size
        self.propagator.probe_shape = new_probe.shape
        self.set_object(new_object)
        self.set_probe(new_probe)

    def save_hdf5(
        self,
        hdf5_root: h5_root,
        save_snapshots: bool = True,
        save_preview: bool = True,
    ) -> None:
        if isinstance(hdf5_root, str | os.PathLike):
            hdf5_root = get_write_hdf5_path(hdf5_root)
        # check if hdf5_root is a path ('path_to_file.hdf5')
        if not isinstance(hdf5_root, h5py.Group):
            with h5py.File(hdf5_root, "w") as f:
                return self.save_hdf5(
                    f, save_snapshots=save_snapshots, save_preview=save_preview
                )

        hdf5_root.attrs[self.H5PY_ATTR_KEY_CLASS] = self.__class__.__name__
        hdf5_root.attrs[self.H5PY_ATTR_BINNING] = self.binning_factor
        hdf5_root.attrs[self.H5PY_ATTR_EPOCH_CTR] = self.epoch_counter

        # saving probe
        hdf5_root.create_dataset(  # pyright: ignore[ reportUnknownMemberType ]
            self.H5PY_KEY_PROBE, data=self.rec_probe.detach().cpu().numpy()
        )

        # saving object
        hdf5_root.create_dataset(  # pyright: ignore[ reportUnknownMemberType ]
            self.H5PY_KEY_OBJECT, data=self.rec_object.detach().cpu().numpy()
        )

        # save propagator
        grp_propagator = hdf5_root.create_group(self.H5PY_KEY_PROPAGATOR)  # pyright: ignore[reportUnknownMemberType]
        self.propagator.save_hdf5(grp_propagator)

        # save loss and training log
        hdf5_root.create_dataset(  # pyright: ignore[reportUnknownMemberType]
            self.H5PY_KEY_TRAINING_LOSS, data=torch.tensor(self.loss)
        )
        hdf5_root.create_dataset(  # pyright: ignore[reportUnknownMemberType]
            self.H5PY_KEY_RECONSTRUCTION_LOG, data=self.logging_str
        )

        # # save all snapshots (dont save previews)
        # if save_snapshots is True:
        #     for snapshot_name in self.snapshots.keys():
        #         snap = self.get_snapshot(snapshot_name)
        #         snap_grp = hdf5_root.create_group(
        #             f"{self.H5PY_KEY_SNAPSHOTS}/{snapshot_name}"
        #         )
        #         snap.save_hdf5(snap_grp, save_preview=False)

    def _save_hdf5(self, _: h5py.Group) -> None:
        pass

    @classmethod
    def from_hdf5(
        cls,
        hdf5_root: str | os.PathLike[Any] | IO[Any] | h5py.Group,
        device: DeviceLikeType | None = None,
    ) -> Self:
        if isinstance(hdf5_root, str | os.PathLike):
            hdf5_root = get_read_hdf5_path(hdf5_root)
        # check if hdf5_root is a path ('path_to_file.hdf5')
        if not isinstance(hdf5_root, h5py.Group):
            with h5py.File(hdf5_root, "r") as f:
                return cls.from_hdf5(f, device=device)
        return cls._from_hdf5(hdf5_root, device=device)

    @classmethod
    def _from_hdf5(
        cls,
        hdf5_root: h5py.Group,
        device: DeviceLikeType | None = None,
    ):
        # load probe
        rec_probe = tensor_from_h5(hdf5_root, cls.H5PY_KEY_PROBE)
        if rec_probe is None:
            raise Exception("Probe not found in hdf5-file.")

        # load object
        rec_object = tensor_from_h5(hdf5_root, cls.H5PY_KEY_OBJECT)
        if rec_object is None:
            raise Exception("Object not found in hdf5-file.")

        # load propagator
        if cls.H5PY_KEY_PROPAGATOR not in hdf5_root:
            raise Exception("Propagator not found in hdf5-file")

        grp_propagator = hdf5_root[cls.H5PY_KEY_PROPAGATOR]
        if not isinstance(grp_propagator, h5py.Group):
            raise Exception(f"Expected Group object. Found: {type(grp_propagator)}")
        prop = from_hdf5(grp_propagator, device=device)
        if not issubclass(type(prop), BasePropagator):
            raise Exception("Failed to load Propagator")

        ret = cls(
            rec_probe=rec_probe,
            rec_object=rec_object,
            propagator=prop,  # pyright: ignore[reportArgumentType]
            device=device,
        )

        bin_fac = hdf5_root.attrs.get(cls.H5PY_ATTR_BINNING, 1)  # pyright:ignore  [reportUnknownMemberType,reportUnknownVariableType]

        if isinstance(bin_fac, np.number):
            ret.binning_factor = bin_fac.item()

        # load loss and log
        # if (loss := hdf5_root[cls.H5PY_KEY_TRAINING_LOSS]) is not None:
        if (loss := tensor_from_h5(hdf5_root, cls.H5PY_KEY_TRAINING_LOSS)) is not None:
            ret.loss = loss.to(torch.float).tolist()  # pyright:ignore[reportUnknownMemberType]

        if cls.H5PY_KEY_RECONSTRUCTION_LOG in hdf5_root:
            ret.logging_str = hdf5_root[cls.H5PY_KEY_RECONSTRUCTION_LOG][()].decode()  # pyright:ignore

        # epoch_counter
        epoch_ctr = hdf5_root.attrs.get(cls.H5PY_ATTR_EPOCH_CTR)  # pyright:ignore  [reportUnknownMemberType,reportUnknownVariableType]
        if isinstance(epoch_ctr, int):
            ret.epoch_counter = epoch_ctr

        # TODO: Load snapshots

        return ret

    def plotter_probe(
        self,
        fov_m: float | None = None,
        labels: list[str] | None = None,
        slices: tuple[slice] | None = None,
        remove_phase_ramp: bool = False,
        mode: str = "MIXED",
        overexpose: float = 1,
    ) -> Callable[[Axes], None]:
        """Plot preview of Probe.

        Probe dimensions: (..., N_wavelengths, H, W)

        Args:
            fov_probe_m: Field of view for Probe (in meters).
            unit: Unit used for x- and y-axes or one of 'm', 'mm', 'um', 'nm', or float.
                This is not affecting the fov.
            labels: List of labels for all the object modes.
            slices: Tuple of slices/indices, to select for plotting. Select single
                dimension by specifying using `None` or `slice()`.
                This is required if Probe.dim() > 3.
            remove_prase_ramp: If set to True, then plotted image will have linear
                phase ramps removed, by centering the FT(image).
            mode: Plotting mode. One of ('MIXED', 'AMPLITUDE', 'ABS', 'PHASE', 'ANGLE').
            overexpose: Number by which brightest pixel will be multiplied
                with after normalization.

        Returns:
            Plotter for probe

        """
        if self.rec_object.dim() > 3:
            if slices is None:
                raise ValueError("slices required but found None.")
            if self.rec_probe.dim() - 2 != len(slices):
                raise ValueError(
                    f"slices have wrong dimensions. "
                    f"Expected: {self.rec_probe.dim() - 2}. Found: {len(slices)}."
                )

        # if labels is None:
        #     labels = self.propagator.to_str_list()
        #     if len(labels) == 1 and self.probe.dim() == 3:
        #         labels = [*labels] * self.probe.shape[-3]

        #     if slices != () and self.probe.dim() - 2 == len(slices):
        #         labels = np.array(labels)[slices[-1]]

        probe_cropped = self.rec_probe
        if fov_m is not None:
            probe_cropped = pad_or_crop(probe_cropped, int(fov_m / self.propagator.dx))

        def plotter(ax: Axes | None = None):
            plot_hsv(
                probe_cropped[slices],
                ax,
                dx=self.propagator.dx,
                labels=labels,
                remove_phase_ramp=remove_phase_ramp,
                mode=mode,
                overexpose=overexpose,
                show_scalebar=True,
            )

        return plotter

    def plotter_object(
        self,
        fov_m: float | None = None,
        rel_illum_threshold: float = 0.01,
        x: DataLoader | Encoder | None = None,
        labels: list[str] | None = None,
        normalize_mean: bool = False,
        slices: tuple[slice] | None = None,
        remove_phase_ramp: bool = False,
        mode: str = "MIXED",
        rois: dict[str, Roi2D] | list[Roi2D] | None = None,
        overexpose: float = 1,
    ) -> Callable[[Axes], None]:
        """Plot preview of Object.

        Args:
            fov_object_m: Field of view for Object (in meters).
            unit: Unit used for x- and y-axes or one of 'm', 'mm', 'um', 'nm', or float.
                This is not affecting the fov.
            rel_illum_threshold: Used for creating illumination mask. Pixels darker than
                `rel_illum_threshold * object.abs().max()` will be set to nan (white).
            x: DataLoader or Encoder to create illumination mask. The same dataloader as
                for the reconstruction should be useded.
            labels: List of labels for all the object modes.
            normalize_mean: If set to true, the masked object will be normalized to
                the mean. This will only affect the phase (color) of the plotted object.
            slices: Tuple of slices/indices, to select for plotting. Select single
                dimension by specifying using `None` or `slice()`. This is required if
                Object.dim() > 3.
            remove_prase_ramp: If set to True, then plotted image will have linear phase
                ramps removed, by centering the FT(image).
            mode: Plotting mode. One of ('MIXED', 'AMPLITUDE', 'ABS', 'PHASE', 'ANGLE').
            overexpose: Number by which brightest pixel will be multiplied
                with after normalization.

        Returns:
            Plotter for object

        """
        if self.rec_object.dim() > 3:
            if slices is None:
                raise ValueError("slices required but found None.")
            if self.rec_object.dim() - 2 != len(slices):
                raise ValueError(
                    f"slices have wrong dimensions. Expected: "
                    f"{self.rec_object.dim() - 2}. Found: {len(slices)}."
                )

        if x is None:
            masked_obj = self.rec_object
            mask = torch.ones_like(masked_obj, dtype=torch.bool)
        else:
            raise NotImplementedError
            mask = self.illumination_mask(x)
            mask = mask < rel_illum_threshold * mask.max()
            masked_obj = self.rec_object.detach().clone()
            if (sum_dims := tuple(range(mask.dim() - masked_obj.dim()))) != ():
                if len(sum_dims) == 1:
                    sum_dims = sum_dims[0]
                mask = mask.amin(sum_dims)
            masked_obj[mask] = torch.nan

        if normalize_mean:
            masked_obj = masked_obj / masked_obj[~mask].mean()

        def plotter(ax: Axes | None = None):
            plot_hsv(
                masked_obj[slices],
                ax,
                dx=self.propagator.dx,
                labels=labels,
                remove_phase_ramp=remove_phase_ramp,
                mode=mode,
                rois=rois,
                window_size=int(fov_m / self.propagator.dx)
                if fov_m is not None
                else None,
                overexpose=overexpose,
                show_scalebar=True,
            )

        return plotter
