#%%
# This script contains some example of preprocessing an dataset into a hdf5 file that can be read into the
import shutil
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


data_folder = Path(r"/tudelft.net/staff-bulk/tnw/ImPhys/OP/scratch/ZhonghuiN/20251030_lensPaper")

print(f"processing: {data_folder.stem}")

filePathForRead = data_folder /"data" / "AVT camera (GT3400)"
filePathForReadBG = data_folder / "background" / "AVT camera (GT3400)"

fileName = data_folder.stem

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

## set experimental parameters

## geometry
flip = False # If your camera flips the image vertically For Visible setup ==> False

# True: reflection
# False: transmission
#reflection_measurement = False
reflection_geometry: bool = False

# GAOI_deg = float(re.search(r"GAOI([0-9]+)",data_folder.parent.stem).groups()[0])
GAOI_deg = 39.0
GRAZING_APOLLO_ANGLE = GAOI_deg * 2
print(f"{GAOI_deg=}")

## laser source
entrancePupilDiameter = 50e-6
# wavelength = 1e-9 * np.array([35.45564516, 38.06854839, 41.04435484, 44.70967742, 48.99193548,
#        54.21774194, 60.82258065, 68.98790323, 77.08064516]) # Krypton?
# wavelength = 1e-9 * np.array([31.28, 33.28, 35.53, 38.21, 41.3, 44.89, 49.21])
wavelength = 1e-9 * np.array([756])

## detectors 
apply_square_sensor = True # If sensor not square

# distance: object -> detector plane  (initial guess)
zo = 8.7e-2

# detector coordinates
cameraPixelSize = 3.69e-6
Nd = 2704                             # number of detector pixels

substract_bkgd: float = 0.00  # factor by how much background should be substracted

## get positions
# get file name (this assumes there is only one text file in the raw data folder)

# take raw data positions
df_stage_positions = pd.read_csv(scan_grid_file, sep=" ", skiprows=2, names=["y","x"])
df_stage_positions = df_stage_positions * 1e-6

# When smartact goes down sample position goes up then we reflect y-coordinates!
df_stage_positions.y = -df_stage_positions.y 
if reflection_geometry:
    df_stage_positions.x = -df_stage_positions.x

# show positions
plt.figure(figsize=(5, 5))
plt.plot(df_stage_positions.x * 1e6, df_stage_positions.y * 1e6, 'o-')
plt.xlabel('(um)')
plt.ylabel('(um)')
plt.show(block=False)

## read data and correct for darks
framesList = list(filePathForRead.glob('*.tif'))
framesList.sort()

framesListBG = list(filePathForReadBG.glob('*.tif'))
framesListBG.sort()

#%%

def load_ptychogram(list_tif_files:list[Path]) -> np.ndarray:

    N_frames = len(list_tif_files)
    ret: np.ndarray | None = None

    for idx, img_path in enumerate(tqdm.tqdm(list_tif_files)):
        
        img = imageio.imread(img_path)

        if ret is None:
            ret = np.zeros((N_frames, *img.shape), dtype=np.float64)

        if flip:
            ret[idx] = np.flipud(img) # flip image vertically
        else: ret[idx] = img # no flip
        
    if ret is None:
        raise Exception()

    return ret

def avg_bkgd(list_bkgd_tif_files:list[Path]) -> np.ndarray:
    ptycho_bkgd = load_ptychogram(list_bkgd_tif_files)
    return np.mean(ptycho_bkgd, 0)

def process_ptychogram(ptychogram:np.ndarray, dark:np.ndarray) -> np.ndarray:
    # ptychogram: [Frame, H, W]
    # dark: [H, W]
    backgroundOffset = 0
    ptychogram_proc = (ptychogram - substract_bkgd * dark- backgroundOffset).clip(0)
    return ptychogram_proc

# Process background and square diffraction patterns
dark = avg_bkgd(framesListBG)
ptychogram = load_ptychogram(framesList)
ptychogram_proc = process_ptychogram(ptychogram, dark)
Nx = ptychogram_proc.shape[-1]
Ny = ptychogram_proc.shape[-2]

if Nx>Ny:
    start = int((Nx-Ny)/2)
    end = Nx-start
    ptychogram_proc = ptychogram_proc[:,:,start:end]
    dark = dark[:,start:end]



barebeam = None
if barebeam_path is not None:
    barebeam = avg_bkgd(barebeam_path)
    barebeam = barebeam / exposure_time_barebeam_s * exposure_time_s # because of different exposure times
    barebeam = process_ptychogram(barebeam, dark)


fig = plt.figure()
ax1, ax2 = fig.subplots(1, 2)
ax1.imshow(np.log10(ptychogram[0]))
ax2.imshow(dark)
ax1.set_title("diff")
ax2.set_title("background")
fig.show()


#%%

# export data
exportBool = True

filePathForSave = folderPathForSave / (
    f"lenspaper" + ".hdf5"
)


#filePathForSave = 'usaf_strp.hdf5'

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
        hf.create_dataset("ptychogram", data=ptychogram_proc, dtype="f")

    print(f"An hd5f file has been saved to: {filePathForSave}")

# %%
