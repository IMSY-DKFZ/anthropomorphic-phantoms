from simpa import Tags
import simpa as sp
import os
import numpy as np
from ap.simulations.pat.custom_msot_acuity import MSOTAcuityEcho
from ap.utils.dye_tissue_properties import get_vessel_tissue, get_background_tissue, get_bone_tissue, get_air_tissue


def get_default_acoustic_settings(path_manager):
    acoustic_settings = {
        Tags.ACOUSTIC_SIMULATION_3D: False,
        Tags.ACOUSTIC_MODEL_BINARY_PATH: path_manager.get_matlab_binary_path(),
        Tags.KWAVE_PROPERTY_ALPHA_POWER: 0.00,
        Tags.KWAVE_PROPERTY_SENSOR_RECORD: "p",
        Tags.KWAVE_PROPERTY_PMLInside: False,
        Tags.KWAVE_PROPERTY_PMLSize: [31, 32],
        Tags.KWAVE_PROPERTY_PMLAlpha: 1.5,
        Tags.KWAVE_PROPERTY_PlotPML: False,
        Tags.RECORDMOVIE: False,
        Tags.MOVIENAME: "visualization_log",
        Tags.ACOUSTIC_LOG_SCALE: True,
        Tags.KWAVE_PROPERTY_INITIAL_PRESSURE_SMOOTHING: False,
    }
    return acoustic_settings


def get_default_das_reconstruction_settings():
    return {
        Tags.RECONSTRUCTION_PERFORM_BANDPASS_FILTERING: True,
        Tags.TUKEY_WINDOW_ALPHA: 0,
        # Tags.BANDPASS_CUTOFF_LOWPASS_IN_HZ: int(1e8),
        # Tags.BANDPASS_CUTOFF_HIGHPASS_IN_HZ: int(100000),
        Tags.BANDPASS_CUTOFF_LOWPASS_IN_HZ: int(5500000),
        Tags.BANDPASS_CUTOFF_HIGHPASS_IN_HZ: int(50000),
        Tags.RECONSTRUCTION_BMODE_AFTER_RECONSTRUCTION: True,
        Tags.RECONSTRUCTION_BMODE_METHOD: Tags.RECONSTRUCTION_BMODE_METHOD_HILBERT_TRANSFORM,
        Tags.RECONSTRUCTION_APODIZATION_METHOD: Tags.RECONSTRUCTION_APODIZATION_BOX,
        Tags.RECONSTRUCTION_MODE: Tags.RECONSTRUCTION_MODE_PRESSURE,
        Tags.DATA_FIELD_SPEED_OF_SOUND: 1497.4,
        Tags.SPACING_MM: 0.1,
    }


def segmentation_class_mapping(forearm_nr, phantom_sos_adjustment=0):
    ret_dict = dict()
    ret_dict[1] = (sp.MolecularCompositionGenerator()
                   .append(sp.MOLECULE_LIBRARY.heavy_water())
                   .get_molecular_composition(1))
    ret_dict[2] = (sp.MolecularCompositionGenerator()
                   .append(sp.MOLECULE_LIBRARY.water())
                   .get_molecular_composition(2))
    ret_dict[3] = get_background_tissue(forearm_nr, phantom_sos_adjustment)
    for i in range(4, 9):
        ret_dict[i] = get_vessel_tissue(i, phantom_sos_adjustment)
    ret_dict[9] = (sp.MolecularCompositionGenerator()
                   .append(sp.MOLECULE_LIBRARY.mediprene())
                   .get_molecular_composition(2))
    ret_dict[10] = get_bone_tissue()
    ret_dict[11] = get_air_tissue()
    return ret_dict


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
