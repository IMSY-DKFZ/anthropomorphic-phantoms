import numpy as np
import nrrd
from ap.utils.default_settings import run_seg_based_simulation
from ap.simulations.pat.custom_msot_acuity import MSOTAcuityEcho
from ap.us_image_analysis.simulation_comparison_visualization import visualize_comparison
import matplotlib.pyplot as plt

# FIXME temporary workaround for newest Intel architectures
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# If VISUALIZE is set to True, the simulation result will be plotted
VISUALIZE = False

# TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
#  point to the correct file in the PathManager().

base_path = "/home/kris/Data/Dye_project/PAT_Data/iThera_2_data/US_analysis/"

forearm_dict = {
    "Forearm_1": "Study_25/Scan_25_pa-labels.nrrd",
    "Forearm_2": "Study_26/Scan_12_pa-labels.nrrd",
    "Forearm_3": "Study_27/Scan_19_pa-labels.nrrd",
    "Forearm_4": "Study_28/Scan_5_pa-labels.nrrd",
    "Forearm_5": "Study_31/Scan_9_pa-labels.nrrd"
}

input_spacing = 0.1
wavelengths = np.arange(700, 851, 10)

device_pos = 151
z_padding = np.ceil(MSOTAcuityEcho().probe_height_mm / input_spacing).astype(int) - device_pos
device_pos += z_padding

for forearm_nr, forearm_path in forearm_dict.items():
    for simulate_air_bubbles in [False, True]:
        path = os.path.join(base_path, forearm_path)
        volume_name = forearm_nr + "_air" if simulate_air_bubbles else forearm_nr

        label_mask, _ = nrrd.read(path)
        label_mask = np.squeeze(label_mask)
        # plt.imshow(np.fliplr(np.rot90(label_mask, 3)))
        # plt.show()
        # exit()
        label_mask = np.expand_dims(label_mask, 1)
        label_mask = np.tile(label_mask, (1, 200, 1))
        label_mask[200, 100, 100] = 11
        y_middle_slice = 100

        # pad the label mask to a size that accomodates the device
        label_mask = np.pad(label_mask, ((160, 160), (0, 0), (z_padding, 0)), mode="edge")

        if simulate_air_bubbles:
            # relabel the air bubble    s from 9 (membrane) to 11 (air) and set y-dim of air bubbles to 0.4 mm
            segment_start_slice = slice(0, y_middle_slice - int(0.2/input_spacing))
            label_mask[:, segment_start_slice, :][label_mask[:, segment_start_slice, :] == 9] = 3

            segment_middle_slice = slice(y_middle_slice - int(0.2/input_spacing), y_middle_slice + int(0.2/input_spacing))
            label_mask[:, segment_middle_slice, :][label_mask[:, segment_middle_slice, :] == 9] = 11

            segment_end_slice = slice(y_middle_slice + int(0.2/input_spacing), None)
            label_mask[:, segment_end_slice, :][label_mask[:, segment_end_slice, :] == 9] = 3
        else:
            label_mask[label_mask == 9] = 3

        # insert membrane at the device position
        label_mask[:, :, device_pos - int(1/input_spacing):device_pos] = 9

        # insert couplant until membrane
        label_mask[:, :, :device_pos - int(1/input_spacing)] = 1

        # plt.imshow(label_mask[:, y_middle_slice, :])
        # plt.title(forearm_nr)
        # plt.show()
        # exit()

        run_seg_based_simulation(save_path=path, volume_name=volume_name,
                                 label_mask=label_mask,
                                 spacing=input_spacing, device_position=device_pos * input_spacing,
                                 wavelengths=wavelengths, forearm_nr=forearm_nr.split("_")[-1])

    visualize_comparison(simulation_path=path,
                         forearm_nr=forearm_nr,
                         wavelengths=wavelengths,
                         comparison_dict={"short": "air",
                                          "description": "air bubbles"},
                         save_fig=not VISUALIZE)
