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
from PtychoTorch.propagators.transmission import Propagator_ASP_scaled
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
#device = "cuda" if torch.cuda.is_available() else "cpu"
# device="cpu"

device = "cuda:0"
print(f"running on {device=}")

project_folder = Path(r"/tudelft.net/staff-bulk/tnw/ImPhys/OP/scratch/ZhonghuiN/20251030_lensPaper")

#%%

results_folder = project_folder / "results"
# measurement_file = project_folder / "preprocess" / "sub0p00bg_ud.hdf5"
measurement_file = project_folder / "preprocess" / "lenspaper.hdf5"

if not measurement_file.is_file():
    raise Exception("measurement_file not found")

if not results_folder.is_dir():
    os.makedirs(results_folder)


#%%
# Create new DataSet-Object containing all the diffraction data.
ds = dataset.DataSet.from_hdf5(
    measurement_file,
    # device="cpu",
    device=device,
)
ds.device
print(ds.shape)

#%%
brightest_ptycho_pixel = ds.ptychogram.max()

ds.encoder.set_mode(Encoder.Encoder_Mode.CENTERED)
ds.ptychogram /= brightest_ptycho_pixel
ds.encoder.position_correction.requires_grad_(True)
ds.encoder.positions.requires_grad_(False)

print(f"brightest pixel is {brightest_ptycho_pixel}")

ds.set_background(ds.h5_groups["background"].to(device) / brightest_ptycho_pixel)
# ds.set_barebeam(ds["barebeam"].to(device) / brightest_ptycho_pixel)

plt.imshow(ds[0].diff_pattern.detach().cpu().log10(),cmap='Blues',interpolation = 'none')
plt.colorbar(location='right', shrink = 0.8)

#%%
#wl = torch.tensor([756e-9]).to(device)
wl= ds.h5_groups["wavelength"]

prop = Propagator_ASP_scaled (
    dq=ds.camera_pixel_size,
    #dq=ds.h5_groups["dxd"][0],
    z=ds.h5_groups["zo"][0],
    wavelength = wl,
    probe_shape=(wl.numel(), *ds.shape[-2:]),
    #probe_shape=ds.shape[-2:],
    device=device,
)

# Define initial guess for the probe. dxp is the Pixel size for the Probe
from PtychoTorch.utils import circ

initial_probe = torch.zeros(prop.probe_shape, dtype=torch.complex64, device=device)
circ_mask  = circ(initial_probe, int(1000e-6 / prop.dx))
initial_probe[circ_mask] = 1.0

plt.imshow(abs(initial_probe[0].cpu()),cmap='Blues',interpolation = 'none')

# Define initial guess for the Object. In this case just consists of ones.
# ds.encoder.find_minimal_object_shape_px returns the minimal shape for the object.
obj_shape = ds.encoder.find_minimal_object_shape_px(prop)
#obj_shape = (4000,4000)

initial_obj = torch.ones(
    obj_shape,
    # dtype=torch.float32,
    dtype=torch.complex64,
    device=device,
)

engine = AutoGradPIE(
    rec_probe=initial_probe,
    rec_object=initial_obj,
    propagator=prop,
    device=device,
)

print(f"z (cm): {engine.propagator.z*1e2:.3f}")
#print(f"AOI (deg): {engine.propagator.get_AOI_deg():.3f}")

fig, (axe1,axe2) = plt.subplots(1,2)
engine.plotter_object(fov_m=1000e-6)(axe1)
engine.plotter_probe(fov_m=1000e-6, slices=(0,), overexpose=2)(axe2)
fig.tight_layout()

#%%
binning_factor = 2
engine.change_binning(binning_factor)
ds.set_binning_factor(binning_factor)
change_binning(engine=engine, ds=ds, binning_factor=binning_factor)

#engine.propagator.z = 0.086

optim = torch.optim.Adam([
            {"params":[engine.rec_probe], },
            {"params":[engine.rec_object], },
        ],
        lr=1e-3,
    )


loss_func = lf.MSE_amplitude
    # + lf.ApertureProbe(500e-6, engine.propagator, device=device) \
    # + lf.COM_probe_center()

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
engine.plotter_object(fov_m=3e-3,overexpose =1)(axes[0])
engine.plotter_probe(fov_m=3e-3, slices=(0,), overexpose=20,)(axes[1])
fig.tight_layout()

# %% save
Reconstruction(
    engine=engine,
).save_hdf5(results_folder / "rec.h5",)

# %% load
engine_loaded = Reconstruction.from_hdf5(results_folder/"rec1.h5", device=device).engine
engine_loaded.plotter_object(fov_m=100e-6, mode="abs")()

# %%
