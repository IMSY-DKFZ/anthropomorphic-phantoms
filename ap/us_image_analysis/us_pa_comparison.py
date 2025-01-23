import os
import glob

import matplotlib.pyplot as plt
import simpa as sp
import numpy as np
from simpa import Tags
import patato as pat
from functools import partial
import pandas as pd
import json
from skimage.transform import rescale
from ap.oxy_estimation.linear_unmixing import LinearUnmixingOxyEstimator
import nrrd
#
#
# from ap.utils.default_settings import (get_default_das_reconstruction_settings,
#                                         )
# from ap.utils.recon_utils import correct_er_sensors
# from ap.simulations.pat.custom_msot_acuity import MSOTAcuityEcho
#
# from simpa.core.simulation_modules.reconstruction_module.reconstruction_utils import \
#     tukey_bandpass_filtering_with_settings



# base_path = "/home/kris/Data/Dye_project/PAT_Data/Processed_Data/"
# base_path = "/home/kris/Data/Dye_project/phantom_data/Kris_Vessels_90-100_processed/"
# base_path = "/home/kris/Data/Dye_project/PAT_Data/iThera_data/Processed_Data/"
# base_path = "/home/kris/Data/Dye_project/PAT_Data/Depth_test_data/Processed_data"
base_path = "/home/kris/Data/Dye_project/PAT_Data/iThera_2_data/Processed_Data"

studies = sorted(glob.glob(os.path.join(base_path, "Study_*")))

wavelengths = np.arange(700, 851, 10)
SPACING = 0.1

examples_images = [
    # ("Study_25", "Scan_25"),
    # ("Study_26", "Scan_12"),
    ("Study_27", "Scan_18"),
    ("Study_28", "Scan_6"),
    # ("Study_31", "Scan_9"),
]

estimator = LinearUnmixingOxyEstimator({
        'estimation_type': 'proxy',
        "unmixing_wavelengths": np.arange(700, 851, 10)
    })


for study in studies:
    study_nr = study.split("_")[-1]
    scans = sorted(glob.glob(os.path.join(study, "Scan_*")))
    for scan in scans:
        if not any([(st in scan) and (sc in scan) for st, sc in examples_images]):
            continue
        scan_nr = scan.split("_")[-1][:-5]
        if scan_nr == "1" or int(scan_nr) == len(scans):
            continue

        pa_data = pat.PAData.from_hdf5(scan)

        time_series = pa_data.get_time_series().raw_data
        us_image = pa_data.get_ultrasound().raw_data
        pa_recon = pa_data.get_scan_reconstructions()[('iThera BP-40mm(res:100μm)', '0')].raw_data
        corr_factor = pa_data.get_overall_correction_factor()
        pa_recon = np.squeeze(pa_recon) / corr_factor[..., None, None]
        ses_field = pa_data.get_impulse_response()[()]

        # filter1 = np.abs(np.fft.fftshift(np.fft.fft(ses_field)))
        # filter1 /= filter1.max()
        # plt.plot(filter1)
        # plt.show()
        # exit()
        # print(pa_recon.shape)
        # plt.subplot(1, 3, 1)
        # plt.title("PA Recon")
        # plt.imshow(np.flipud(np.squeeze(pa_recon[16, 10, ...])))
        # plt.subplot(1, 3, 2)
        # plt.title("US Image")
        # plt.imshow(rescale(np.flipud(np.squeeze(us_image[16, 10, ...]))[5:-5, 5:-5], 2))
        # # plt.imshow(np.flipud(np.squeeze(pa_recon[16, 10, ...])), alpha=0.5)
        # plt.subplot(1, 3, 3)
        # plt.title("Unmixed result")
        # est_input = np.squeeze(pa_recon[16, :16, ...])
        # inp_range = np.max(est_input) - np.min(est_input)
        # est_input = (est_input - np.min(est_input)) / inp_range
        # oxy_estimates = estimator.estimate(est_input)
        # plt.imshow(np.flipud(oxy_estimates))
        # plt.show()
        # exit()

        for array_name, arr in zip(["_us.nrrd", "_pa.nrrd"], [rescale(np.rot90(np.flipud(np.squeeze(us_image[16, 10, ...])), 1)[5:-5, 5:-5], 2), np.rot90(np.flipud(np.squeeze(pa_recon[16, 10, ...])), 1)]):
            save_path = scan.replace(base_path, "/home/kris/Data/Dye_project/PAT_Data/iThera_2_data/US_analysis/")
            save_path = save_path.replace(".hdf5", array_name)
            os.makedirs(os.path.split(save_path)[0], exist_ok=True)
            nrrd.write(save_path, arr)


