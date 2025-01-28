from simpa import Tags
import simpa as sp
import numpy as np
import os
from ap.simulations.pat.custom_msot_acuity import MSOTAcuityEcho
from ap.utils.default_settings import get_default_acoustic_settings, get_default_das_reconstruction_settings, \
    segmentation_class_mapping


def run_seg_based_simulation(save_path, volume_name, label_mask, spacing, device_position,
                             wavelengths, forearm_nr: str = "Forearm_1", phantom_sos_adjustment: int = 0):
    path_manager = sp.PathManager()

    labels_shape = label_mask.shape
    settings = sp.Settings()
    settings[Tags.SIMULATION_PATH] = os.path.dirname(save_path)
    settings[Tags.VOLUME_NAME] = volume_name
    settings[Tags.RANDOM_SEED] = 1234
    settings[Tags.WAVELENGTHS] = wavelengths
    settings[Tags.SPACING_MM] = spacing
    settings[Tags.DIM_VOLUME_X_MM] = labels_shape[0] * spacing
    settings[Tags.DIM_VOLUME_Y_MM] = labels_shape[1] * spacing
    settings[Tags.DIM_VOLUME_Z_MM] = labels_shape[2] * spacing
    settings[Tags.GPU] = True

    settings.set_volume_creation_settings({
        Tags.INPUT_SEGMENTATION_VOLUME: label_mask,
        Tags.SEGMENTATION_CLASS_MAPPING: segmentation_class_mapping(forearm_nr,
                                                                    phantom_sos_adjustment=phantom_sos_adjustment),

    })

    settings.set_optical_settings({
        Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e8,
        Tags.OPTICAL_MODEL_BINARY_PATH: path_manager.get_mcx_binary_path(),
        Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50,
    })

    settings.set_reconstruction_settings(get_default_das_reconstruction_settings())
    settings.set_acoustic_settings(get_default_acoustic_settings(path_manager))

    pipeline = [
        sp.SegmentationBasedAdapter(settings),
        sp.MCXAdapter(settings),
        sp.KWaveAdapter(settings),
        sp.DelayAndSumAdapter(settings),
        sp.FieldOfViewCropping(settings),
    ]

    device = MSOTAcuityEcho(device_position_mm=np.array([
        settings[Tags.DIM_VOLUME_X_MM]/2,
        settings[Tags.DIM_VOLUME_Y_MM]/2,
        device_position]),
        field_of_view_extent_mm=np.array([-20, 20, 0, 0, 0, 20]))

    sp.simulate(pipeline, settings, device)
