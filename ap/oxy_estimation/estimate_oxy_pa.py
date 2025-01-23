import numpy as np
import matplotlib.pyplot as plt
import simpa as sp
import nrrd
import os
import json

from ap.oxy_estimation.linear_unmixing import LinearUnmixingOxyEstimator

if __name__ == "__main__":
    base_path = "/home/kris/Data/Dye_project/PAT_Data/iThera_2_data/"
    wavelengths = np.arange(700, 851, 10)
    estimator = LinearUnmixingOxyEstimator({
        'estimation_type': 'proxy',
        "unmixing_wavelengths": wavelengths
    })
    z_det_pos = 152

    examples_images = {
        1: {"oxy": 0.5, "path": os.path.join("Study_25", "Scan_25"), "sim_path": os.path.join("Study_25", "Forearm_1")},
        2: {"oxy": 0.3, "path": os.path.join("Study_26", "Scan_12"), "sim_path": os.path.join("Study_26", "Forearm_2")},
        3: {"oxy": 0, "path": os.path.join("Study_27", "Scan_19"), "sim_path": os.path.join("Study_27", "Forearm_3")},
        4: {"oxy": 0.7, "path": os.path.join("Study_28", "Scan_5"), "sim_path": os.path.join("Study_28", "Forearm_4")},
        5: {"oxy": 1, "path": os.path.join("Study_31", "Scan_9"), "sim_path": os.path.join("Study_31", "Forearm_5")},
    }

    oxy_dict = {
        0: 4,
        0.3: 5,
        0.5: 6,
        0.7: 7,
        1: 8
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
        pa_path = os.path.join(base_path, "Reconstructions_das", forearm_specs["path"] + "_recon1.hdf5")
        us_path = os.path.join(base_path, "US_analysis", forearm_specs["path"] + "_us.nrrd")
        sim_path = os.path.join(base_path, "US_analysis", forearm_specs["sim_path"] + ".hdf5")
        labels_path = os.path.join(base_path, "US_analysis", forearm_specs["path"] + "_pa-labels.nrrd")

        reconstruction = sp.load_data_field(pa_path, sp.Tags.DATA_FIELD_RECONSTRUCTED_DATA)
        reconstruction_array = np.stack([np.fliplr(np.rot90(reconstruction[str(wl)][:, :, ...], 3)) for wl in wavelengths])

        fluence = sp.load_data_field(sim_path, sp.Tags.DATA_FIELD_FLUENCE)
        fluence_array = np.stack([np.fliplr(np.rot90(fluence[str(wl)][:, :, ...], 3)) for wl in wavelengths])

        mua_sim = sp.load_data_field(sim_path, sp.Tags.DATA_FIELD_ABSORPTION_PER_CM)
        mua_sim_array = np.stack([np.fliplr(np.rot90(mua_sim[str(wl)][:, :, ...], 3)) for wl in wavelengths])

        # plt.subplot(3, 2, 1)
        # plt.imshow(mua_sim_array[9], vmin=0.01, vmax=5.5)
        # plt.colorbar()
        # plt.subplot(3, 2, 2)
        # plt.title(f"wl: {wavelengths[9]}")
        # plt.imshow(fluence_array[9], vmin=0.0001, vmax=0.01)
        # plt.colorbar()
        # plt.subplot(3, 2, 3)
        # plt.imshow(mua_sim_array[9], vmin=0.01, vmax=5.5)
        # plt.colorbar()
        # plt.subplot(3, 2, 2)
        # plt.title(f"wl: {wavelengths[9]}")
        # plt.imshow(fluence_array[9], vmin=0.0001, vmax=0.01)
        # plt.colorbar()
        # plt.subplot(3, 2, 5)
        # plt.title("Absorption")
        # plt.plot(wavelengths, mua_sim_array[:, 0, 0], label="Water", c="blue")
        # plt.plot(wavelengths, mua_sim_array[:, 150, 100], label="Background", c="pink")
        # plt.show()
        # plt.close()

        fluence_corr_recon = reconstruction_array / fluence_array

        # recon_norm = np.linalg.norm(reconstruction_array, axis=0, ord=1)
        # reconstruction_array = reconstruction_array / recon_norm[np.newaxis, :]

        us_image = nrrd.read(us_path)[0]
        us_image = np.fliplr(np.rot90(us_image, 3))[z_det_pos:z_det_pos+200, :]

        labels = nrrd.read(labels_path)[0]
        labels = np.squeeze(np.fliplr(np.rot90(labels, 3))[z_det_pos:z_det_pos+200, :])

        vessels = (4 <= labels) & (labels <= 8)
        tissue = (3 <= labels) & (labels <= 8)

        oxy = np.zeros_like(labels) * np.nan
        for oxy_val, label in oxy_dict.items():
            oxy[labels == label] = oxy_val

        oxy[labels == 3] = forearm_dict[forearm_nr]

        oxy_estimates = estimator.estimate(reconstruction_array)
        fluence_corr_estimates = estimator.estimate(fluence_corr_recon)
        with open(pa_path.replace("hdf5", "json"), "r") as json_file:
            regression_data = json.load(json_file)
        cal_oxy_estimates = estimator.estimate(
            (reconstruction_array - regression_data["intercept"])/regression_data["slope"])
        #
        # fluence_corr_cal_oxy_estimates = estimator.estimate(
        #     (reconstruction_array - regression_data["intercept"]) / regression_data["slope"] / fluence_array)


        # oxy_nans = np.isnan(oxy_estimates)
        # cal_oxy_nans = np.isnan(cal_oxy_estimates)
        #
        # plt.subplot(3, 2, 1)
        # plt.imshow(oxy, vmin=0, vmax=1)
        # plt.title("Ground Truth")
        # plt.subplot(3, 2, 2)
        # plt.imshow(mua_sim_array[0])
        # plt.title("Absorption")
        # plt.subplot(3, 2, 4)
        # plt.imshow(fluence_array[0])
        # plt.title("Fluence")
        # plt.subplot(3, 2, 3)
        # plt.imshow(oxy_estimates)
        # plt.title("LSU estimates")
        # plt.subplot(3, 2, 6)
        # plt.imshow(fluence_corr_estimates)
        # plt.title("Fluence corrected LSU estimates")
        # plt.subplot(3, 2, 5)
        # # plt.imshow(cal_oxy_estimates)
        # plt.show()
        # plt.close()
        # exit()

        res = {
            "oxy_error": np.abs(oxy - oxy_estimates),
            "cal_oxy_error": np.abs(oxy - cal_oxy_estimates),
            "fluence_corr_error": np.abs(oxy - fluence_corr_estimates),
            "vessels": vessels,
            "tissue": tissue
        }
        np.savez(f"/home/kris/Data/Dye_project/PAT_Data/Oxy_Results/forearm_{forearm_nr}.npz", **res)
