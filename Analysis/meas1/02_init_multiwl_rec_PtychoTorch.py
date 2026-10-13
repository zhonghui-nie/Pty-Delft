# 2024_08_22_ReflPtychoAOI30_circuitCD0p5um_Fermat_FoV67um_stepSize3p76um_301points_zeros50_1p6s
#%%
%load_ext autoreload
%autoreload 2
# %%
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import math
import os
from PtychoTorch import dataset, hooks
from PtychoTorch.encoder import Encoder
from PtychoTorch import dataloader
from PtychoTorch.propagators.reflection import ASP_TC3
from PtychoTorch.engine import AutoGradPIE
import PtychoTorch.loss_functions as lf
from PtychoTorch.utils import change_binning, change_cropping
from PtychoTorch.save_load import Reconstruction

import pandas as pd
import torch

from PtychoTorch.server import HTTP_Server

if (server := locals().get("server")):
    server.stop()
server = HTTP_Server()

# %%
# device = "cuda" if torch.cuda.is_available() else "cpu"

device = "cuda:1"
print(f"running on {device=}")

project_folder = Path(r"/tudelft.net/staff-bulk/tnw/ImPhys/OP/scratch/ZhonghuiN/2025_10_09_FirstPtychoMeas_RESDHM_Au_Cr/Kr_Rect_2umRand_FOV100_N676_xstep4p0_ystep4p0_randPath_barebeam50iter_1p9s")

#%%

results_folder = project_folder / "results"
# measurement_file = project_folder / "preprocess" / "sub0p00bg_ud.hdf5"
measurement_file = project_folder / "preprocess" / "sub0p00bg.hdf5"

if not measurement_file.is_file():
    raise Exception("measurement_file not found")

if not results_folder.is_dir():
    os.makedirs(results_folder)


#%%
# Create new DataSet-Object containing all the diffraction data.
ds = dataset.DataSet.from_hdf5(
    measurement_file,
    # sl=slice(0,10),
    # device="cpu",
    device=device,
)
ds.device

#%%
brightest_ptycho_pixel = ds.ptychogram.max()

ds.encoder.set_mode(Encoder.Encoder_Mode.CENTERED)
ds.ptychogram /= brightest_ptycho_pixel
ds.encoder.position_correction.requires_grad_(True)
ds.encoder.positions.requires_grad_(False)

print(f"brightest pixel is {brightest_ptycho_pixel}")

#%%
ds.set_background(ds.h5_groups["background"].to(device) / brightest_ptycho_pixel)
# ds.set_barebeam(ds["barebeam"].to(device) / brightest_ptycho_pixel)

#%%
wl = torch.tensor([39.1e-9]).to(device)

prop = ASP_TC3(
    dq=ds.camera_pixel_size,
    z=ds.h5_groups["zo"][0],
    wavelength=wl,
    AOI_rad = ds.h5_groups["AOI_deg"]/180*torch.pi,
    probe_shape=(wl.numel(), *ds.shape[-2:]),
    device=device,
)

# Define initial guess for the probe. dxp is the Pixel size for the Probe
from PtychoTorch.plotting import plot_hsv
from PtychoTorch.utils import circ

initial_probe = torch.zeros(prop.probe_shape, dtype=torch.complex64, device=device)
circ_mask  = circ(initial_probe, 50  )
initial_probe[circ_mask] = 1.0

# Define initial guess for the Object. In this case just consists of ones.
# ds.encoder.find_minimal_object_shape_px returns the minimal shape for the object.
obj_shape = ds.encoder.find_minimal_object_shape_px(prop)
obj_shape = (4000,4000)

initial_obj = torch.ones(
    obj_shape,
    # dtype=torch.float32,
    dtype=torch.complex64,
    device=device,
)
#%%

engine = AutoGradPIE(
    rec_probe=initial_probe,
    rec_object=initial_obj,
    propagator=prop,
    device=device,
)

#%%
engine_backup = engine.copy()

#%%
print(f"z (cm): {engine.propagator.z*1e2:.5f}")
print(f"AOI (deg): {engine.propagator.get_AOI_deg():.5f}")

#%%
a = 1


#%%
a += 1
print(a)

#%%
engine.plotter_object(fov_m=100e-6)()
engine.plotter_probe(fov_m=100e-6, slices=(0,))()

#%%
engine.propagator.set_GAOI_deg(38.5)

#%%

new_probe = engine.rec_probe.unsqueeze(0).expand((2, -1, -1, -1))
engine.set_probe(new_probe)


#%%
change_binning(engine=engine, ds=ds, binning_factor=2)


engine.propagator.z = 0.17

optim = torch.optim.Adam([
            {"params":[engine.rec_probe], },
            {"params":[engine.rec_object], },
        ],
        lr=1e-1,
    )


loss_func = lf.MSE_amplitude + lf.ApertureProbe(30e-6, engine.propagator, engine.propagator.get_AOI_deg().item(), device=device)

engine.reconstruct(
    train_loader=dataloader.BatchDataLoader(
        ds,
        batch_size=10,
    ),
    optimizer=optim,
    loss_func=loss_func,
    N_epoch=50,
)

#%%
fig, axes = plt.subplots(1,2)
engine.plotter_object(fov_m=100e-6, overexpose=2)(axes[0])
engine.plotter_probe(fov_m=100e-6, slices=(0,), overexpose=2)(axes[1])
fig.tight_layout()

#%%

# %% save
Reconstruction(
    engine=engine,
).save_hdf5(results_folder / "rec2.h5",)

# %% load
engine_loaded = Reconstruction.from_hdf5(results_folder/"rec1.h5", device=device).engine

# %%
engine_loaded.plotter_object(fov_m=100e-6, mode="abs")()
