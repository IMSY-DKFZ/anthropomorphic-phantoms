import numpy as np
import nrrd
from ap.simulations.pat.custom_msot_acuity import MSOTAcuityEcho
from ap.simulations.pat.seg_simulation import run_seg_based_simulation
from ap.us_image_analysis.simulation_comparison_visualization import visualize_comparison
from ap.us_image_analysis.vessel_diameter_comparison import check_for_vessel_diameter_changes

# FIXME temporary workaround for newest Intel architectures
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# If VISUALIZE is set to True, the simulation result will be plotted
VISUALIZE = False

# TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
#  point to the correct file in the PathManager().

base_path = "/home/kris/Data/Dye_project/publication_data/"

forearm_dict = {
    "Forearm_1": "Scan_25_pa-labels.nrrd",
    "Forearm_2": "Scan_12_pa-labels.nrrd",
    "Forearm_3": "Scan_19_pa-labels.nrrd",
    "Forearm_4": "Scan_5_pa-labels.nrrd",
    "Forearm_5": "Scan_9_pa-labels.nrrd"
}

input_spacing = 0.1
wavelengths = np.arange(700, 851, 10)

device_pos = 151
z_padding = np.ceil(MSOTAcuityEcho().probe_height_mm / input_spacing).astype(int) - device_pos
device_pos += z_padding

for f_idx, (forearm_nr, forearm_path) in enumerate(forearm_dict.items()):
    sos_adjustments = range(-100, 101, 50)
    for sos_adjustment in sos_adjustments:
        if sos_adjustment == 0:
            continue
        path = os.path.join(base_path, "PAT_Data", f"Phantom_0{f_idx + 1}", forearm_path)
        volume_name = forearm_nr + f"_sos{sos_adjustment}" if sos_adjustment else forearm_nr

        label_mask, _ = nrrd.read(path)
        label_mask = np.squeeze(label_mask)
        # plt.imshow(np.fliplr(np.rot90(label_mask, 3)))
        # plt.show()
        # exit()
        label_mask = np.expand_dims(label_mask, 1)
        label_mask = np.tile(label_mask, (1, 200, 1))
        # label_mask[200, 100, 100] = 11
        y_middle_slice = 100

        # pad the label mask to a size that accomodates the device
        label_mask = np.pad(label_mask, ((160, 160), (0, 0), (z_padding, 0)), mode="edge")

        # relabel the air bubbles from 9 (membrane) to 3 (background)
        label_mask[label_mask == 9] = 3

        # insert membrane at the device position
        label_mask[:, :, device_pos - int(1/input_spacing):device_pos] = 9

        # insert couplant until membrane
        label_mask[:, :, :device_pos - int(1/input_spacing)] = 1

        save_path = os.path.join(base_path, "Paper_Results", "PAT_Simulations", f"Phantom_0{f_idx + 1}", forearm_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        run_seg_based_simulation(save_path=save_path, volume_name=volume_name,
                                 label_mask=label_mask,
                                 spacing=input_spacing, device_position=device_pos * input_spacing,
                                 wavelengths=wavelengths, forearm_nr=forearm_nr.split("_")[-1],
                                 phantom_sos_adjustment=sos_adjustment)

    visualize_comparison(simulation_path=save_path,
                         forearm_nr=forearm_nr,
                         wavelengths=wavelengths,
                         comparison_dict={"short": [f"sos{sos_adjustment}" for sos_adjustment in sos_adjustments
                                                    if sos_adjustment != 0],
                                          "description": "changed sos"},
                         save_fig=not VISUALIZE)

    check_for_vessel_diameter_changes(simulation_path=path,
                                      forearm_nr=forearm_nr,
                                      comparison_dict={"short": [f"sos{sos_adjustment}" for sos_adjustment in sos_adjustments
                                                                 if sos_adjustment != 0],
                                                       "description": "changed sos"},
                                      save_fig=not VISUALIZE)
