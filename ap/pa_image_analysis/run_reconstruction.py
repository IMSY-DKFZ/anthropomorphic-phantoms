import os
import glob
import simpa as sp
import numpy as np
from simpa import Tags
import patato as pat
from functools import partial
from pathlib import Path


from ap.utils.default_settings import (get_default_das_reconstruction_settings)
from ap.utils.recon_utils import correct_er_sensors
from ap.simulations.pat.custom_msot_acuity import MSOTAcuityEcho

from simpa.core.simulation_modules.reconstruction_module.reconstruction_utils import \
    tukey_bandpass_filtering_with_settings


base_path = "/path/to/publication/data"
frame_averaging = (9, 13)
scans = sorted(glob.glob(os.path.join(base_path, "PAT_Data", "Phantom_*", "Scan_*_time_series*")))
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

settings = sp.Settings(general_settings)
settings.set_reconstruction_settings(recon_settings)
settings[Tags.K_WAVE_SPECIFIC_DT] = 2.5e-8

for scan in scans:
    scan_nr = scan.split("_")[-1][:-5]

    pa_data = pat.PAData.from_hdf5(scan)
    time_series = pa_data.get_time_series().raw_data

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

        bandpass_filter = partial(tukey_bandpass_filtering_with_settings, global_settings=settings,
                                  component_settings=recon_settings, device=None)

        time_series_corr = np.moveaxis(correct_er_sensors(np.moveaxis(time_series_corr, 0, 2),
                                                          er_sensor_list=np.arange(-1, 255, 8),
                                                          filter_func=bandpass_filter), 2, 0)

        scan_path = Path(scan.replace("time_series", "recon"))
        save_path = Path(base_path) / "Paper_Results" / "PAT_Reconstructions" / scan_path.parent.stem
        os.makedirs(save_path, exist_ok=True)
        settings[Tags.VOLUME_NAME] = str(scan_path.stem)

        settings[Tags.SIMULATION_PATH] = str(save_path)
        simpa_output = dict()
        simpa_output[Tags.SETTINGS] = settings
        save_path = str(save_path / scan_path.name)
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
