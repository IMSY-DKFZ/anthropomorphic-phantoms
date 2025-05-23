import numpy as np
import matplotlib.pyplot as plt
import simpa as sp
import nrrd
import os
import json
from ap.oxy_estimation.linear_unmixing import LinearUnmixingOxyEstimator
plt.switch_backend("TkAgg")

if __name__ == "__main__":

    try:
        run_by_bash: bool = bool(os.environ["RUN_BY_BASH"])
        print("This runner script is invoked in a bash script!")
    except KeyError:
        run_by_bash: bool = False

    if run_by_bash:
        base_path = os.environ["BASE_PATH"]
    else:
        # In case the script is run from an IDE, the base path has to be set manually
        base_path = ""

    wavelengths = np.arange(710, 851, 10)
    estimator = LinearUnmixingOxyEstimator({
        "estimation_type": "proxy",
        "unmixing_wavelengths": wavelengths,
        "spectra_path": os.path.join(base_path, "Measured_Spectra")
    })
    z_det_pos = 152

    examples_images = {
        1: {"oxy": 0.5, "path": "Scan_25"},
        2: {"oxy": 0.3, "path": "Scan_12"},
        3: {"oxy": 0, "path": "Scan_19"},
        4: {"oxy": 0.7, "path": "Scan_5"},
        5: {"oxy": 1, "path": "Scan_9"},
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
        4: 1,
        5: 0.5,
    }

    results = dict()
    for forearm_nr, forearm_specs in examples_images.items():
        print(f"Processing Forearm {forearm_nr}")
        results[forearm_nr] = dict()
        pa_path = os.path.join(base_path, "PAT_Data", f"Phantom_0{forearm_nr}",
                               forearm_specs["path"] + "_recon.hdf5")
        sim_path = os.path.join(base_path, "PAT_Data", f"Phantom_0{forearm_nr}",
                                "Simulations", f"Forearm_{forearm_nr}" + ".hdf5")
        labels_path = os.path.join(base_path, "PAT_Data", f"Phantom_0{forearm_nr}",
                                   forearm_specs["path"] + "_pa-labels.nrrd")

        reconstruction = sp.load_data_field(pa_path, sp.Tags.DATA_FIELD_RECONSTRUCTED_DATA)
        reconstruction_array = np.stack([np.rot90(reconstruction[str(wl)][:, :, ...], 3) for wl in wavelengths])

        fluence = sp.load_data_field(sim_path, sp.Tags.DATA_FIELD_FLUENCE)
        fluence_array = np.stack([np.fliplr(np.rot90(fluence[str(wl)][:, :, ...], 3)) for wl in wavelengths])

        mua_sim = sp.load_data_field(sim_path, sp.Tags.DATA_FIELD_ABSORPTION_PER_CM)
        mua_sim_array = np.stack([np.fliplr(np.rot90(mua_sim[str(wl)][:, :, ...], 3)) for wl in wavelengths])

        fluence_corr_recon = reconstruction_array / fluence_array


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

        json_path = os.path.join(base_path, "Paper_Results", "PAT_Measurement_Correlation",
                                 f"PAT_spectrum_correlation_oxy_{int(100 * forearm_specs['oxy']):0d}_p0.json")
        with open(json_path, "r") as json_file:
            regression_data = json.load(json_file)
        cal_oxy_estimates = estimator.estimate(
            (reconstruction_array - regression_data["intercept"])/regression_data["slope"])

        res = {
            "oxy_error": np.abs(oxy - oxy_estimates),
            "cal_oxy_error": np.abs(oxy - cal_oxy_estimates),
            "fluence_corr_error": np.abs(oxy - fluence_corr_estimates),
            "vessels": vessels,
            "tissue": tissue
        }

        save_path = os.path.join(base_path, "Paper_Results", "Oxy_Results", f"pat_forearm_{forearm_nr}.npz")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, **res)
