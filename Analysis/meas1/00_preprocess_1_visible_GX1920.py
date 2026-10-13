# %%
# This script contains some example of preprocessing an dataset into a hdf5 file that can be read into the
summary = """
Preprocessing for visible setup:
  - 2D PI stages for sample scan 
  - GX1920 resolution 1936 (H) x 1456 (V) pixel size = 4.54 um
  - transmission / reflection (flips axes)
"""
print(summary)

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import imageio
import tqdm
import h5py
from pathlib import Path
import os
import re
import glob

data_folder = Path(r"/tudelft.net/staff-bulk/tnw/ImPhys/OP/scratch/ZhonghuiN/20260918_Rens_visible Pty/20260918_USAF_Gaussian")

print(f"processing: {data_folder.stem}")

# check the folder nanme "data" and "background"
filePathForRead = data_folder / "data" / "Camera"
filePathForReadBG = data_folder / "background" / "Camera"
folderPathForSave = data_folder / "preprocess"

if not folderPathForSave.is_dir():
    os.makedirs(folderPathForSave)

scan_grid_file = list(filter(lambda x: x.stem != "result", filePathForRead.parent.glob("*.txt")))
if len(scan_grid_file) != 1:
    
    raise Exception("Scangridfile not found")

scan_grid_file = scan_grid_file[0]


#%%
# barebeam:
barebeam_path:Path|None = None
# barebeam_path = list(data_folder.parent.glob("Ar_barebeam_*.tif"))
# barebeam_path = list(data_folder.parent.glob("Ar_barebeam_1s_pos*.tif"))

# set experimental specifications:

# True: reflection
# False: transmission
reflection_geometry: bool = False

entrancePupilDiameter = 500e-6
# GAOI_deg = float(re.search(r"GAOI([0-9]+)",data_folder.parent.stem).groups()[0])
GAOI_deg = 30
GRAZING_APOLLO_ANGLE = GAOI_deg * 2
print(f"{GAOI_deg=} degree")

## set experimental specifications
zo = 12e-2  # distance: sample <-> camera
cameraPixelSize = 4.54e-6 

# wavelength = 1e-9 * np.array([35.45564516, 38.06854839, 41.04435484, 44.70967742, 48.99193548,
#        54.21774194, 60.82258065, 68.98790323, 77.08064516]) # Krypton?
# wavelength = 1e-9 * np.array([31.28, 33.28, 35.53, 38.21, 41.3, 44.89, 49.21])
wavelength = 1e-9 * np.array([1030])  # MLM

substract_bkgd: float = 0  # factor by how much background should be substracted

## get positions
# get file name (this assumes there is only one text file in the raw data folder)

# take raw data positions
df_stage_positions = pd.read_csv(
    #scan_grid_file, sep=" ", skiprows=2, names=["y", "x", "wedge_pos"]
    scan_grid_file, sep=" ", skiprows=2, names=["x", "y"]
)
df_stage_positions = df_stage_positions * 1e-6

# --- TRANSFORM APOLLO TO EQUIVALENT VIS SETUP
#df_stage_positions.x = -df_stage_positions.x
#df_stage_positions.y = -df_stage_positions.y
# ---

# Flip the y axes for visible setup
#df_stage_positions.x = -df_stage_positions.x
df_stage_positions.y = -df_stage_positions.y

# Flipping for 3D Smaract Stage
# df_stage_positions.x = -df_stage_positions.x  # flip x

# show positions
plt.figure(figsize=(5, 5))
plt.plot(df_stage_positions.x * 1e6, df_stage_positions.y * 1e6, "o-")
plt.xlabel("(um)")
plt.ylabel("(um)")
plt.show(block=False)

## read data and correct for darks
framesList = list(filePathForRead.glob("*.tif"))
framesList.sort()

framesListBG = list(filePathForReadBG.glob("*.tif"))
framesListBG.sort()

# %%

def load_ptychogram(list_tif_files: list[Path]) -> np.ndarray:

    N_frames = len(list_tif_files)
    ret = None

    for idx, img_path in enumerate(tqdm.tqdm(list_tif_files)):

        img = imageio.imread(img_path)
        
        # --- TRANSFORM APOLLO TO EQUIVALENT VIS SETUP
        # img = np.fliplr(img) # flip horizontally
        # ---

        # up-down reverse in Visible setup
        img = np.flipud(img)
        #img = np.fliplr(img)

        if reflection_geometry:
            # Reflection
            img = np.fliplr(img)
            img = np.flipud(img)
       
        if ret is None:
            ret = np.zeros((N_frames, *img.shape), dtype=np.uint64)
        ret[idx] = img

    return ret

def avg_bkgd(list_bkgd_tif_files: list[Path]):
    ptycho_bkgd = load_ptychogram(list_bkgd_tif_files)
    return np.mean(ptycho_bkgd, 0)


def process_ptychogram(ptychogram: np.ndarray, dark: np.ndarray) -> np.ndarray:
    # ptychogram: [Frame, H, W]
    # dark: [H, W]
    
    ptychogram_proc = (ptychogram - substract_bkgd * dark).clip(0)
    # ptychogram_proc = ptychogram
    # ptychogram_proc = np.flip(ptychogram_proc, axis=-2) #only for CCD
    return ptychogram_proc

dark = avg_bkgd(framesListBG)

ptychogram = load_ptychogram(framesList)
ptychogram = process_ptychogram(ptychogram, dark)

barebeam = None
if barebeam_path is not None:
    barebeam = avg_bkgd(barebeam_path)
    barebeam = barebeam / exposure_time_barebeam_s * exposure_time_s # because of different exposure times
    barebeam = process_ptychogram(barebeam, dark)

fig = plt.figure()
ax1, ax2 = fig.subplots(1, 2)
diff = ax1.imshow(ptychogram[0],cmap='Blues',interpolation = 'none')
fig.colorbar(diff, ax=ax1, location='right', shrink = 0.4)
bg = ax2.imshow(dark, cmap='Blues',interpolation = 'none')
fig.colorbar(bg, ax = ax2, location='right', shrink = 0.4)
ax1.set_title("diff")
ax2.set_title("background")
fig.show()


# %% select central square ROI
Nd_row, Nd_col=ptychogram[0].shape
pixel_difference = int(abs(Nd_row-Nd_col)/2)
ptychogram = ptychogram[:,:,pixel_difference:Nd_col-pixel_difference]
dark = dark[:,pixel_difference:Nd_col-pixel_difference]

Nd = ptychogram.shape[-1]
# conversion_as_per_um = 85.256

# export data
exportBool = True


def f2s(number: float) -> str:
    return f"{number:.2f}".replace(".", "p")


# filePathForSave = folderPathForSave / (
#     f"sub{f2s(substract_bkgd)}bg" + ".hdf5"
# )

filePathForSave = folderPathForSave / "spiral.hdf5"

if filePathForSave.exists():
    raise Exception("File already exists!")

if exportBool:
    with h5py.File(filePathForSave, "w") as hf:
        hf.attrs["version"] = "1.0"
        hf.create_dataset("encoder", data=df_stage_positions[["y", "x"]], dtype="f")
        hf.create_dataset("dxd", data=(cameraPixelSize,), dtype="f")
        hf.create_dataset("Nd", data=(Nd,), dtype="i")
        hf.create_dataset("zo", data=(zo,), dtype="f")
        hf.create_dataset("wavelength", data=(wavelength,), dtype="f")
        hf.create_dataset(
            "entrancePupilDiameter", data=(entrancePupilDiameter,), dtype="f"
        )

        hf.attrs["reflection_measurement"] = reflection_geometry
        if reflection_geometry:
            hf.create_dataset("GAOI_deg", data=GAOI_deg, dtype="f")
            hf.create_dataset("AOI_deg", data=90 - GAOI_deg, dtype="f")

        with open(__file__) as f:
            hf.attrs["preproccess"] = f.read()

        hf.create_dataset("background", data=dark, dtype="f")
        if barebeam is not None:
            hf.create_dataset("barebeam", data=barebeam, dtype="f")
        hf.create_dataset("ptychogram", data=ptychogram, dtype="f")

    print(f"An hd5f file has been saved to: {filePathForSave}")

# %%
