import numpy as np
import nrrd
from ap.simulations.pat.custom_msot_acuity import MSOTAcuityEcho
from ap.simulations.pat.seg_simulation import run_seg_based_simulation
from ap.simulations.pat.simulation_comparison_visualization import visualize_comparison
from ap.utils.get_env_variable import env

# FIXME temporary workaround for newest Intel architectures
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# If VISUALIZE is set to True, the simulation result will be plotted

# TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
#  point to the correct file in the PathManager().

run_by_bash = env("RUN_BY_BASH")

if run_by_bash:
    print("This runner script is invoked in a bash script!")
    base_path = env("BASE_PATH")
    VISUALIZE = env("VISUALIZE")
    run_simulation = env("RUN_SIMULATION")
else:
    # In case the script is run from an IDE, the base path has to be set manually
    base_path = ""
    VISUALIZE = False
    run_simulation = False

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
    for simulate_air_bubbles in [False, True]:
        path = os.path.join(base_path, "PAT_Data", f"Phantom_0{f_idx + 1}", forearm_path)
        volume_name = forearm_nr + "_air" if simulate_air_bubbles else forearm_nr
        if run_simulation:
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
                segment_start_slice = slice(0, y_middle_slice - int(0.2 / input_spacing))
                label_mask[:, segment_start_slice, :][label_mask[:, segment_start_slice, :] == 9] = 3

                segment_middle_slice = slice(y_middle_slice - int(0.2 / input_spacing),
                                             y_middle_slice + int(0.2 / input_spacing))
                label_mask[:, segment_middle_slice, :][label_mask[:, segment_middle_slice, :] == 9] = 11

                segment_end_slice = slice(y_middle_slice + int(0.2 / input_spacing), None)
                label_mask[:, segment_end_slice, :][label_mask[:, segment_end_slice, :] == 9] = 3
            else:
                label_mask[label_mask == 9] = 3

            # insert membrane at the device position
            label_mask[:, :, device_pos - int(1 / input_spacing):device_pos] = 9

            # insert couplant until membrane
            label_mask[:, :, :device_pos - int(1 / input_spacing)] = 1

            save_path = os.path.join(base_path, "Paper_Results", "PAT_Simulations", f"Phantom_0{f_idx + 1}",
                                     forearm_path.replace("-labels.nrrd", ".hdf5"))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            run_seg_based_simulation(save_path=save_path, volume_name=volume_name,
                                     label_mask=label_mask,
                                     spacing=input_spacing, device_position=device_pos * input_spacing,
                                     wavelengths=wavelengths, forearm_nr=forearm_nr.split("_")[-1],
                                     phantom_sos_adjustment=0, path_to_data=base_path)
        else:
            save_path = os.path.join(base_path, "PAT_Data", f"Phantom_0{f_idx + 1}", "Simulations",
                                     forearm_path.replace("-labels.nrrd", ".hdf5"))

    results_path = os.path.join(base_path, "Paper_Results", "PAT_Simulations",
                                f"Phantom_0{f_idx + 1}")
    os.makedirs(results_path, exist_ok=True)
    visualize_comparison(simulation_path=save_path,
                         forearm_nr=forearm_nr,
                         wavelengths=wavelengths,
                         comparison_dict={"short": "air",
                                          "description": "air bubbles"},
                         save_fig=not VISUALIZE,
                         results_path=results_path)
