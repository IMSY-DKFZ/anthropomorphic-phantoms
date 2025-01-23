import os
import glob
import patato as pat
import numpy as np

from tqdm import tqdm

base_path = "/home/kris/Data/Dye_project/PAT_Data/Depth_test_data/Processed_data"

study_list = sorted(os.listdir(base_path))
maximum = 4095

for study in study_list:
    if "60" in study:
        continue
    print(study)
    study_dir = os.path.join(base_path, study)
    scan_list = sorted(glob.glob(os.path.join(study_dir, "Scan_*")))

    for scan in tqdm(scan_list):
        if "recon" in scan:
            continue
        pa_data = pat.PAData.from_hdf5(scan)
        time_series = pa_data.get_time_series().raw_data

        glob_max = np.max(time_series[:, :, :, 10:])

        if glob_max == maximum:
            print(f"Attention!! Saturation in {scan}!!")
    print("Success for ", study)