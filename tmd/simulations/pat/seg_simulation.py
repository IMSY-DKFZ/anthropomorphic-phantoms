from simpa import Tags
import simpa as sp
import numpy as np
import nrrd
import matplotlib.pyplot as plt
from tmd.utils.dye_tissue_properties import get_vessel_tissue, get_background_tissue
from tmd.utils.default_settings import get_default_das_reconstruction_settings, get_default_acoustic_settings

# FIXME temporary workaround for newest Intel architectures
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# If VISUALIZE is set to True, the simulation result will be plotted
VISUALIZE = False

# TODO: Please make sure that a valid path_config.env file is located in your home directory, or that you
#  point to the correct file in the PathManager().
path_manager = sp.PathManager()
base_path = "/home/kris/Data/Dye_project/PAT_Data/Example_reconstructions_das"

forearm_dict = {
    "Forearm_1": {
        "nr": 1,
        "label_path": "Study_17/Scan_1_image-labels.nrrd",
        "device_pos": 440
    },
    "Forearm_3": {
        "nr": 3,
        "label_path": "Study_18/Scan_1_image-labels.nrrd",
        "device_pos": 453,
    },
    "Forearm_6": {
        "nr": 6,
        "label_path": "Study_19/Scan_1_image-labels.nrrd",
        "device_pos": 459
    },
}

for forearm in forearm_dict:

    path = os.path.join(base_path, forearm_dict[forearm]["label_path"])
    volume_name = forearm

    label_mask, _ = nrrd.read(path)
    label_mask = np.rot90(label_mask[0], 1)
    label_mask = np.expand_dims(label_mask, 1)
    input_spacing = 0.1
    label_mask = np.tile(label_mask, (1, 200, 1))
    y_middle_slice = 100

    label_mask = np.pad(label_mask, ((180, 180), (0, 0), (400, 100)), mode="edge")
    labels_shape = label_mask.shape
    # plt.imshow(label_mask[:, y_middle_slice, :])
    # plt.title(f"Forearm {forearm_dict[forearm]['nr']}")
    # plt.show()
    # continue


    def segmentation_class_mapping():
        ret_dict = dict()
        ret_dict[1] = (sp.MolecularCompositionGenerator()
                       .append(sp.MOLECULE_LIBRARY.heavy_water())
                       .get_molecular_composition(1))
        ret_dict[2] = (sp.MolecularCompositionGenerator()
                       .append(sp.MOLECULE_LIBRARY.water())
                       .get_molecular_composition(2))
        ret_dict[3] = get_background_tissue(forearm_dict[forearm]["nr"])
        for i in range(4, 9):
            ret_dict[i] = get_vessel_tissue(i)
        return ret_dict


    settings = sp.Settings()
    settings[Tags.SIMULATION_PATH] = os.path.dirname(path)
    settings[Tags.VOLUME_NAME] = volume_name
    settings[Tags.RANDOM_SEED] = 1234
    settings[Tags.WAVELENGTHS] = np.arange(700, 851, 10)
    settings[Tags.SPACING_MM] = input_spacing
    settings[Tags.DIM_VOLUME_X_MM] = labels_shape[0] * input_spacing
    settings[Tags.DIM_VOLUME_Y_MM] = labels_shape[1] * input_spacing
    settings[Tags.DIM_VOLUME_Z_MM] = labels_shape[2] * input_spacing
    settings[Tags.GPU] = True

    settings.set_volume_creation_settings({
        Tags.INPUT_SEGMENTATION_VOLUME: label_mask,
        Tags.SEGMENTATION_CLASS_MAPPING: segmentation_class_mapping(),

    })

    settings.set_optical_settings({
        Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e8,
        Tags.OPTICAL_MODEL_BINARY_PATH: path_manager.get_mcx_binary_path(),
        Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50,
    })

    settings.set_reconstruction_settings(get_default_das_reconstruction_settings())
    settings.set_acoustic_settings(get_default_acoustic_settings(path_manager))

    pipeline = [
        sp.SegmentationBasedVolumeCreationAdapter(settings),
        sp.MCXAdapter(settings),
        sp.KWaveAdapter(settings),
        sp.DelayAndSumAdapter(settings),
        sp.FieldOfViewCropping(settings),
    ]

    device = sp.MSOTAcuityEcho(device_position_mm=np.array([
        settings[Tags.DIM_VOLUME_X_MM]/2,
        settings[Tags.DIM_VOLUME_Y_MM]/2,
        forearm_dict[forearm]["device_pos"] * input_spacing]),
        field_of_view_extent_mm=np.array([-20, 20, 0, 0, -4, 16]))

    sp.simulate(pipeline, settings, device)

    if Tags.WAVELENGTH in settings:
        WAVELENGTH = settings[Tags.WAVELENGTH]
    else:
        WAVELENGTH = 700

    if VISUALIZE:
        sp.visualise_data(path_to_hdf5_file=os.path.join(os.path.dirname(path), f"{volume_name}.hdf5"),
                          wavelength=WAVELENGTH,
                          show_initial_pressure=True,
                          show_segmentation_map=True)
