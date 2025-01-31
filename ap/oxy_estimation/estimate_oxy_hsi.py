import numpy as np
import matplotlib.pyplot as plt
import simpa as sp
import nrrd
import os
import json
from htc import DataPath

from ap.oxy_estimation.linear_unmixing import LinearUnmixingOxyEstimator

if __name__ == "__main__":
    base_path = "/home/kris/Data/Dye_project/HSI_Data/"
    wavelengths = np.arange(700, 851, 10)
    hsi_wavelengths = np.arange(500, 1000, 5)
    estimator = LinearUnmixingOxyEstimator({
        'estimation_type': 'proxy',
        "unmixing_wavelengths": wavelengths
    })

    examples_images = {
        1: {"oxy": 0.5, "path": "2024_02_20_15_28_13"},
        2: {"oxy": 0.3, "path": "2024_02_20_15_44_02"},
        3: {"oxy": 0, "path": "2024_02_20_16_12_35"},
        4: {"oxy": 0.7, "path": "2024_02_20_16_24_05"},
        5: {"oxy": 1, "path": "2024_02_20_16_50_30"},
    }

    forearm_dict = {
        1: 0,
        2: 0.5,
        3: 1,
        4: 0,
        5: 0.5,
    }

    results = dict()
    for forearm_nr, forearm_specs in examples_images.items():
        print(f"Processing Forearm {forearm_nr}")
        results[forearm_nr] = dict()
        path = os.path.join(base_path, "HSI_Data", forearm_specs["path"])
        htc_data = DataPath(path)
        rgb = htc_data.read_rgb_reconstructed()
        hsi = htc_data.read_cube()

        slices_for_correct_wavelengths = np.where(np.isin(hsi_wavelengths, wavelengths))[0]
        hsi = hsi[:, :, slices_for_correct_wavelengths]
        hsi = np.moveaxis(hsi, 2, 0)

        labels, _ = nrrd.read(os.path.join(path, f"{forearm_specs['path']}-labels.nrrd"))
        data_shape = labels.shape

        json_path = os.path.join(base_path, "Paper_Results", "HSI_Measurement_Correlation",
                                 f"HSI_spectrum_correlation_oxy_{int(100 * forearm_specs['oxy']):0d}.json")
        with open(json_path, "r") as json_file:
            regression_data = json.load(json_file)

        oxy_estimates = estimator.estimate(hsi)
        cal_oxy_estimates = estimator.estimate(
            (hsi - regression_data["intercept"])/regression_data["slope"])

        vessels = labels == 3
        tissue = labels == 2

        oxy = np.zeros_like(labels) * np.nan
        oxy[vessels] = forearm_specs["oxy"]
        oxy[tissue] = forearm_dict[forearm_nr]

        res = {
            "oxy_error": np.abs(oxy - oxy_estimates),
            "cal_oxy_error": np.abs(oxy - cal_oxy_estimates),
            "vessels": vessels,
            "tissue": tissue
        }

        save_dir = os.path.join(base_path, "Paper_Results", "Oxy_Results")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"hsi_forearm_{forearm_nr}.npz")

        np.savez(save_path, **res)
