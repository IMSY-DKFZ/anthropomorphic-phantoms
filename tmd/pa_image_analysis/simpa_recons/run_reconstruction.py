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


from tmd.utils.default_settings import (get_default_das_reconstruction_settings)
from tmd.utils.recon_utils import correct_er_sensors
from tmd.simulations.pat.custom_msot_acuity import MSOTAcuityEcho

from simpa.core.simulation_modules.reconstruction_module.reconstruction_utils import \
    tukey_bandpass_filtering_with_settings



# base_path = "/home/kris/Data/Dye_project/PAT_Data/Processed_Data/"
# base_path = "/home/kris/Data/Dye_project/phantom_data/Kris_Vessels_90-100_processed/"
# base_path = "/home/kris/Data/Dye_project/PAT_Data/iThera_data/Processed_Data/"
# base_path = "/home/kris/Data/Dye_project/PAT_Data/Depth_test_data/Processed_data"
base_path = "/home/kris/Data/Dye_project/PAT_Data/iThera_2_data/Processed_Data"
frame_averaging = (9, 13)
studies = sorted(glob.glob(os.path.join(base_path, "Study_*")))
recon_settings = get_default_das_reconstruction_settings()
wavelengths = np.arange(700, 851, 10)
SPACING = 0.1
general_settings = {
            # These parameters set the general properties of the simulated volume
            Tags.GPU: True,
            Tags.WAVELENGTHS: wavelengths,
            Tags.DO_FILE_COMPRESSION: True,
            Tags.DIM_VOLUME_X_MM: 80,
            Tags.DIM_VOLUME_Y_MM: 20,
            Tags.DIM_VOLUME_Z_MM: 50,
            Tags.SPACING_MM: SPACING,
            Tags.RANDOM_SEED: 42,
            Tags.CONTINUE_SIMULATION: True,
        }

# examples_images = [("Study_6", "Scan_22"), ("Study_7", "Scan_11"), ("Study_8", "Scan_17"),
#                    ("Study_9", "Scan_4"), ("Study_10", "Scan_8"), ("Study_11", "Scan_19")]

# examples_images = [("Study_17", "Scan_1"), ("Study_18", "Scan_1"), ("Study_19", "Scan_1"), ("Study_20", "Scan_1")]
examples_images = [("Study_25", "Scan_25"),
                   ("Study_26", "Scan_12"),
                   ("Study_27", "Scan_18"),
                   ("Study_28", "Scan_6"),
                   ("Study_31", "Scan_9")]
# examples_images = [("Study_27", "Scan_19")]

# study_nrs = np.arange(61, 67, 1)
# vessel_scan_nrs = pd.read_csv("/home/kris/Downloads/Forearm Motion Annotation - Tabellenblatt1.csv", header=[0, 1])
# vessel_scan_nrs = vessel_scan_nrs.values[:, ::3]
#
# examples_images = [(f"Study_{study_nr}", f"Scan_{vessel_scan_nrs[i, study_idx]}") for i in range(3) for study_idx, study_nr in enumerate(study_nrs)]
# with open("/home/kris/Data/Dye_project/PAT_Data/iThera_data/Reconstructions_das/minimum_motion_frames.json") as in_file:
#     vessel_frames = json.load(in_file)


settings = sp.Settings(general_settings)
settings.set_reconstruction_settings(recon_settings)
settings[Tags.K_WAVE_SPECIFIC_DT] = 2.5e-8

for study in studies:
    study_nr = study.split("_")[-1]
    # if "25" not in study:
    #     continue
    scans = sorted(glob.glob(os.path.join(study, "Scan_*")))
    for scan in scans:
        if not any([(st in scan) and (sc in scan) for st, sc in examples_images]):
            continue
        scan_nr = scan.split("_")[-1][:-5]
        if scan_nr == "1" or int(scan_nr) == len(scans):
            continue
        print(scan)
        # frame_averaging = tuple(vessel_frames[f"Study_{study_nr}"][f"Scan_{scan_nr}"])
        pa_data = pat.PAData.from_hdf5(scan)

        time_series = pa_data.get_time_series().raw_data
        # us_image = pa_data.get_ultrasound().raw_data
        # plt.imshow(np.flipud(np.squeeze(us_image[16, 10, ...])))
        # plt.show()
        # print(us_image)
        # exit()
        nr_frames = time_series.shape[0]
        if not frame_averaging:
            frame_ranges = range(0, nr_frames)
        else:
            if not isinstance(frame_averaging, tuple):
                middle_frame = int(nr_frames / 2)
                frame_ranges = range(middle_frame - frame_averaging // 2, middle_frame + frame_averaging // 2 + 1)
            else:
                end_frame = frame_averaging[1] if frame_averaging[1] < nr_frames else nr_frames - 1
                frame_ranges = range(frame_averaging[0], end_frame + 1)

        frames = list()
        corr_fac = pa_data.get_overall_correction_factor()
        for fr_idx, frame in enumerate(frame_ranges):
            if frame == 0 or frame == nr_frames - 1:
                continue

            correction_factors = corr_fac[frame, :, None, None]
            time_series_corr = time_series[frame, ...] / correction_factors

            bandpass_filter = partial(tukey_bandpass_filtering_with_settings, global_settings=settings, component_settings=recon_settings, device=None)

            time_series_corr = np.moveaxis(correct_er_sensors(np.moveaxis(time_series_corr, 0, 2),
                                                              er_sensor_list=np.arange(-1, 255, 8),
                                                              filter_func=bandpass_filter), 2, 0)

            save_path = scan.replace("Processed_Data", "Reconstructions_das")
            save_path = save_path.replace(".hdf5", "_recon1.hdf5")
            dirname = os.path.dirname(save_path)
            settings[Tags.SIMULATION_PATH] = dirname
            settings[Tags.VOLUME_NAME] = os.path.basename(save_path).split(".")[0]
            os.makedirs(dirname, exist_ok=True)
            simpa_output = dict()
            simpa_output[Tags.SETTINGS] = settings
            sp.save_hdf5(simpa_output, save_path)
            save_dict = {wl: time_series_corr[wl_idx] for wl_idx, wl in enumerate(wavelengths)}
            sp.save_data_field(data=save_dict,
                               file_path=save_path,
                               data_field=Tags.DATA_FIELD_TIME_SERIES_DATA)

            device = MSOTAcuityEcho(field_of_view_extent_mm=np.array([-20, 20, 0, 0, 0, 20]))
            recon = sp.DelayAndSumAdapter(settings)

            sp.simulate(simulation_pipeline=[recon], digital_device_twin=device, settings=settings)

            recon = sp.load_data_field(save_path, Tags.DATA_FIELD_RECONSTRUCTED_DATA)
            frames.append(recon)

        if not frame_averaging:
            save_frame = {wl: np.stack([frame[str(wl)] for frame in frames], axis=1) for wl in wavelengths}
        else:
            save_frame = {wl: np.mean(np.array([frame[str(wl)] for frame in frames]), axis=0) for wl in wavelengths}
        sp.save_data_field(save_frame, save_path, Tags.DATA_FIELD_RECONSTRUCTED_DATA)
