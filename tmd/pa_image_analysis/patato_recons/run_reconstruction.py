import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import patato as pat
from patato.io.hdf.hdf5_interface import HDF5Writer

DAS = False
MB = True

base_path = "/home/kris/Data/Dye_project/PAT_Data/Processed_example_data/"
example_path = "/home/kris/Data/Dye_project/PAT_Data/Processed_Data/Study_9/Scan_1.hdf5"
studies = sorted(glob.glob(os.path.join(base_path, "Study_*")))
wavelengths = np.arange(700, 851, 10)
SPACING = 0.1

nx = 400 # number of pixels
lx = 4e-2 # m

if DAS:
    pre_bp = pat.MSOTPreProcessor(lp_filter=7e6, hp_filter=5e3, absolute="real")
    das = pat.Backprojection(field_of_view=(lx, 0, lx), n_pixels=(nx, 1, nx),)

# examples_images = [("Study_6", "Scan_22"), ("Study_7", "Scan_11"), ("Study_8", "Scan_17"),
#                    ("Study_9", "Scan_4"), ("Study_10", "Scan_8"), ("Study_11", "Scan_19")]

examples_images = [("Study_17", "Scan_1"), ("Study_18", "Scan_1"), ("Study_19", "Scan_1"), ("Study_20", "Scan_1")]

if MB:
    mb = pat.ModelBasedReconstruction(model_c=1540, model_max_iter=200, n_pixels=(nx, 1, nx), field_of_view=(lx, 0, lx),
                                      pa_example=pat.PAData.from_hdf5(example_path), model_regulariser="laplacian",
                                      gpu=True, model_regulariser_lambda=5.6e6)
    pre_mb = pat.PreProcessor()

for study in studies:
    scans = sorted(glob.glob(os.path.join(study, "Scan_*")))
    for scan in scans:
        if not any([(st in scan) and (sc in scan) for st, sc in examples_images]):
            continue
        print(scan)
        pa_data = pat.PAData.from_hdf5(scan)

        if DAS:
            ts_bp, settings_bp, _ = pre_bp.run(pa_data.get_time_series(), pa_data)
            rec_backprojection, _, _ = das.run(ts_bp, pa_data, **settings_bp)
            save_path = scan.replace("Processed_Data", "Example_reconstructions_patato_das")

        if not DAS and MB:
            ts_model_based = pa_data.get_time_series().copy()
            ts_model_based.raw_data = ts_model_based.raw_data/pa_data.get_overall_correction_factor()[:, :, None, None]
            rec_backprojection, _, _ = mb.run(ts_model_based, pa_data)
            save_path = scan.replace("Processed_example_data", "Example_reconstructions_patato_mb")

        save_path = save_path.replace("hdf5", "npz")
        dirname = os.path.dirname(save_path)

        os.makedirs(dirname, exist_ok=True)

        recon_data = rec_backprojection.raw_data
        wavelengths = rec_backprojection.wavelengths

        middle_frame = int(pa_data.shape[0] / 2)
        recon_middle = recon_data[middle_frame-1: middle_frame+2, ...]
        recon_middle = np.mean(recon_middle, axis=0)

        np.savez(save_path, recon=np.squeeze(recon_middle), wavelengths=wavelengths)

        pre_mb = pat.PreProcessor()

