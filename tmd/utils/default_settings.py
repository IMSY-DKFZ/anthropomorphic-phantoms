from simpa import Tags

def get_default_acoustic_settings(path_manager):
    acoustic_settings = {
        Tags.ACOUSTIC_SIMULATION_3D: True,
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
        Tags.BANDPASS_CUTOFF_LOWPASS_IN_HZ: int(1e8),
        Tags.BANDPASS_CUTOFF_HIGHPASS_IN_HZ: int(100000),
        Tags.RECONSTRUCTION_BMODE_AFTER_RECONSTRUCTION: True,
        Tags.RECONSTRUCTION_BMODE_METHOD: Tags.RECONSTRUCTION_BMODE_METHOD_HILBERT_TRANSFORM,
        Tags.RECONSTRUCTION_APODIZATION_METHOD: Tags.RECONSTRUCTION_APODIZATION_BOX,
        Tags.RECONSTRUCTION_MODE: Tags.RECONSTRUCTION_MODE_PRESSURE,
        Tags.DATA_FIELD_SPEED_OF_SOUND: 1500,
        Tags.SPACING_MM: 0.1,
    }


def get_default_tr_reconstruction_settings(path_manager):
    return {
        Tags.RECONSTRUCTION_PERFORM_BANDPASS_FILTERING: True,
        Tags.ACOUSTIC_MODEL_BINARY_PATH: path_manager.get_matlab_binary_path(),
        # Tags.ACOUSTIC_SIMULATION_3D: True,
        Tags.KWAVE_PROPERTY_ALPHA_POWER: 1.05,
        Tags.TUKEY_WINDOW_ALPHA: 0.5,
        Tags.BANDPASS_CUTOFF_LOWPASS_IN_HZ: 8e6,
        Tags.BANDPASS_CUTOFF_HIGHPASS_IN_HZ: 0.1e4,
        Tags.RECONSTRUCTION_BMODE_AFTER_RECONSTRUCTION: True,
        Tags.RECONSTRUCTION_BMODE_METHOD: Tags.RECONSTRUCTION_BMODE_METHOD_HILBERT_TRANSFORM,
        Tags.RECONSTRUCTION_APODIZATION_METHOD: Tags.RECONSTRUCTION_APODIZATION_BOX,
        Tags.RECONSTRUCTION_MODE: Tags.RECONSTRUCTION_MODE_PRESSURE,
        Tags.KWAVE_PROPERTY_SENSOR_RECORD: "p",
        Tags.KWAVE_PROPERTY_PlotPML: False,
        Tags.RECORDMOVIE: False,
        Tags.MOVIENAME: "visualization_log",
        Tags.ACOUSTIC_LOG_SCALE: True,
        Tags.KWAVE_PROPERTY_PMLInside: False,
        Tags.KWAVE_PROPERTY_PMLSize: [31, 32],
        Tags.KWAVE_PROPERTY_PMLAlpha: 1.5,
        Tags.DATA_FIELD_SPEED_OF_SOUND: 1540,
        Tags.SPACING_MM: 0.1,
        Tags.MODEL_SENSOR_FREQUENCY_RESPONSE: False
    }

