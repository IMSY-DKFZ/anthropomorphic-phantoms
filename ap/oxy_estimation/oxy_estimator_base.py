import numpy as np
import os
import simpa as sp
from ap.utils.io_iad_results import load_iad_results


class OxyEstimator:
    """
    Base class for oxygenation estimators.

    Parameters:
    config (dict): Configuration dictionary for the estimator.
    """

    def __init__(self, config):
        self.config = config
        self.estimation_type = self.config.get('estimation_type', 'physiological')
        self.unmixing_wavelengths = self.config.get('unmixing_wavelengths', np.arange(700, 851, 10))
        self.path_to_measured_spectra = self.config.get('spectra_path', None)
        if self.path_to_measured_spectra is None:
            raise ValueError("Path to measured spectra must be provided.")
        self.spectral_basis = self.load_spectral_components()

    def estimate(self, input_data):
        """
        Estimates oxygenation from the input data.

        Parameters:
        input_data (np.ndarray): Multispectral data. Can be either 2D (samples x spectra)
                                 or 4D (samples x height x width x spectra).

        Returns:
        np.ndarray: Estimated oxygenation values with the same shape as input_data, excluding spectral dimension.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def load_spectral_components(self):
        """
        Load the spectral components based on the estimation type.
        """
        if self.estimation_type == 'physiological':
            return self.load_physiological_spectra()
        elif self.estimation_type == 'proxy':
            return self.load_proxy_spectra()
        else:
            raise ValueError("Invalid estimation type. Must be 'physiological' or 'proxy'.")

    def load_physiological_spectra(self):
        """
        Load the physiological spectral components (oxy- and deoxyhemoglobin).
        """
        hbo2_spectrum, hb_spectrum = sp.get_simpa_internal_absorption_spectra_by_names(
            [sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_OXYHEMOGLOBIN,
             sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_DEOXYHEMOGLOBIN]
        )
        wavelengths = hb_spectrum.wavelengths
        hb_spectrum, hbo2_spectrum = hb_spectrum.values, hbo2_spectrum.values
        hb_spectrum = np.interp(self.unmixing_wavelengths, wavelengths, hb_spectrum)
        hbo2_spectrum = np.interp(self.unmixing_wavelengths, wavelengths, hbo2_spectrum)
        return np.vstack([hbo2_spectrum, hb_spectrum]).T

    def load_proxy_spectra(self):
        """
        Load the proxy spectral components from measured data.
        """
        hb_abs_spectrum = load_iad_results(os.path.join(self.path_to_measured_spectra, "B90.npz"))["mua"]
        hb_abs_spectrum = np.interp(self.unmixing_wavelengths, np.arange(650, 950), hb_abs_spectrum)
        hbo2_abs_spectrum = load_iad_results(os.path.join(self.path_to_measured_spectra, "BIR3.npz"))["mua"]
        hbo2_abs_spectrum = np.interp(self.unmixing_wavelengths, np.arange(650, 950), hbo2_abs_spectrum)
        return np.vstack([hbo2_abs_spectrum, hb_abs_spectrum]).T

