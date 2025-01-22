from simpa import Tags
import simpa as sp
import numpy as np
import nrrd
import matplotlib.pyplot as plt
from tmd.utils.default_settings import run_seg_based_simulation

# FIXME temporary workaround for newest Intel architectures
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# If VISUALIZE is set to True, the simulation result will be plotted
VISUALIZE = True

# TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
#  point to the correct file in the PathManager().

base_path = "/home/kris/Data/Dye_project/PAT_Data/iThera_2_data/Reconstructions_das"

forearm_dict = {
    "Forearm_1": {
        "nr": 1,
        "label_path": "Study_26/Scan_4_recon-labels.nrrd",
        "device_pos": 460
    },
    # "Forearm_3": {
    #     "nr": 3,
    #     "label_path": "Study_18/Scan_1_image-labels.nrrd",
    #     "device_pos": 453,
    # },
    # "Forearm_6": {
    #     "nr": 6,
    #     "label_path": "Study_19/Scan_1_image-labels.nrrd",
    #     "device_pos": 459
    # },
}

for forearm in forearm_dict:

    path = os.path.join(base_path, forearm_dict[forearm]["label_path"])
    volume_name = forearm

    label_mask, _ = nrrd.read(path)
    label_mask = np.squeeze(label_mask)
    # plt.imshow(np.fliplr(np.rot90(label_mask, 3)))
    # plt.show()
    # exit()
    label_mask = np.expand_dims(label_mask, 1)
    input_spacing = 0.1
    label_mask = np.tile(label_mask, (1, 200, 1))
    label_mask[200, 100, 100] = 11
    y_middle_slice = 100

    label_mask = np.pad(label_mask, ((160, 160), (0, 0), (460, 100)), mode="edge")
    # label_mask = np.pad(label_mask, ((180, 180), (0, 0), (0, 0)), mode="edge")
    # plt.imshow(label_mask[:, y_middle_slice, :])
    # plt.title(f"Forearm {forearm_dict[forearm]['nr']}")
    # plt.show()
    # exit()
    # continue

    run_seg_based_simulation(save_path=path, volume_name=volume_name, label_mask=label_mask,
                             spacing=input_spacing, device_position=forearm_dict[forearm]["device_pos"] * input_spacing,
                             wavelengths=np.arange(700, 701, 10), forearm_nr=forearm_dict[forearm]["nr"])

    WAVELENGTH = 700
    if VISUALIZE:

        data = sp.load_data_field(os.path.join(os.path.dirname(path), f"{volume_name}.hdf5"),
                                  Tags.DATA_FIELD_RECONSTRUCTED_DATA, wavelength=WAVELENGTH)
        plt.imshow(np.fliplr(np.rot90(data, 3)))
        plt.show()
        print("hallo")
