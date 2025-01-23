import os
import glob
import numpy as np
import simpa as sp
import json
from tqdm import tqdm


base_path = f"/home/kris/Data/Dye_project/PAT_Data/iThera_data/Reconstructions_das/"
studies = sorted(glob.glob(os.path.join(base_path, "Study_*")))
nr_of_averaging_images = 3


def constraint_rolling_average(arr, n=nr_of_averaging_images):
    return [np.mean(arr[k:k+n]) for k in range(len(arr) + 1 - n)]


motion_dict = dict()

for study in tqdm(studies):
    study_nr = study.split("_")[-1]
    scans = sorted(glob.glob(os.path.join(study, "Scan_*")))
    motion_dict[f"Study_{study_nr}"] = dict()
    for scan in scans:
        # if scan != "/home/kris/Data/Dye_project/PAT_Data/iThera_data/Reconstructions_das/Study_61/Scan_24_recon.hdf5":
        #     continue
        scan_nr = scan.split("_")[-2]
        wavelengths = sp.load_data_field(scan, sp.Tags.SETTINGS)[sp.Tags.WAVELENGTHS]
        reconstruction = sp.load_data_field(scan, sp.Tags.DATA_FIELD_RECONSTRUCTED_DATA)
        reconstruction_array = np.stack([np.rot90(reconstruction[str(wl)][:, :, ...], 3) for wl in wavelengths])

        diff_array = np.abs(reconstruction_array[:, :-1, ...] - reconstruction_array[:, 1:, ...])
        differences = np.mean(diff_array, axis=(0, 2, 3))

        beginning_of_minimum_motion_sequence = int(np.argmin(constraint_rolling_average(differences,
                                                                                        n=nr_of_averaging_images - 1)))
        motion_dict[f"Study_{study_nr}"][f"Scan_{scan_nr}"] = (beginning_of_minimum_motion_sequence,
                                                               beginning_of_minimum_motion_sequence
                                                               + nr_of_averaging_images - 1)


with open(os.path.join(base_path, "minimum_motion_frames.json"), "w+") as out_file:
    json.dump(motion_dict, out_file, indent=1)
