import numpy as np
from scipy.optimize import nnls
import os
import simpa as sp
from tmd.utils.io_iad_results import load_iad_results
# np.seterr(all="raise")

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
            [sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_OXYHEMOGLOBIN, sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_DEOXYHEMOGLOBIN]
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
        base_path = "/home/kris/Data/Dye_project/Measured_Spectra"
        hb_abs_spectrum = load_iad_results(os.path.join(base_path, "B90.npz"))["mua"]
        hb_abs_spectrum = np.interp(self.unmixing_wavelengths, np.arange(650, 950), hb_abs_spectrum)
        hbo2_abs_spectrum = load_iad_results(os.path.join(base_path, "BIR.npz"))["mua"]
        hbo2_abs_spectrum = np.interp(self.unmixing_wavelengths, np.arange(650, 950), hbo2_abs_spectrum)
        return np.vstack([hbo2_abs_spectrum, hb_abs_spectrum]).T


class LinearUnmixingOxyEstimator(OxyEstimator):
    """
    Oxygenation estimator using linear unmixing with a non-negativity constraint (NNLS).

    Parameters:
    config (dict): Configuration dictionary. Should contain the following keys:
                   - 'spectral_basis': The known spectral components (basis matrix) to unmix.
                   - Other configurations can be added as needed.
    """

    def estimate(self, input_data):
        """
        Perform linear unmixing with NNLS to estimate oxygenation.

        Parameters:
        input_data (np.ndarray): The multispectral input data. Can be 1D or 3D.
                                 - 1D: (n_spectra,) for a single spectrum.
                                 - 3D: (n_spectra, height, width) for an image.

        Returns:
        np.ndarray: Estimated oxygenation map with the same shape as input_data,
                    excluding the spectral dimension.
        """
        if input_data.ndim == 1:
            # Single spectrum: (n_spectra,)
            return self._estimate_per_spectrum(input_data[np.newaxis, :])[0]
        elif input_data.ndim == 3:
            # Image-based input: (n_spectra, height, width)
            n_spectra, height, width = input_data.shape
            result = np.zeros((height, width))
            for h in range(height):
                for w in range(width):
                    pixel_spectra = input_data[:, h, w]
                    result[h, w] = self._estimate_per_spectrum(pixel_spectra[np.newaxis, :])[0]
            return result
        else:
            raise ValueError("Input data must be either 1D (single spectrum) or 3D (image data).")

    def _estimate_per_spectrum(self, sample_data):
        """
        Helper function to estimate oxygenation for pixel-based or single-spectrum data.

        Parameters:
        sample_data (np.ndarray): 2D array (n_samples, n_spectra) representing the data.

        Returns:
        np.ndarray: Estimated oxygenation per sample.
        """
        n_samples = sample_data.shape[0]
        oxy_estimates = np.zeros(n_samples)

        # Perform NNLS for each sample
        for i in range(n_samples):
            spectra = sample_data[i, :]
            # NNLS to solve: spectra ≈ spectral_basis * coefficients, with coefficients >= 0
            coeffs, _ = nnls(self.spectral_basis, spectra)
            # Assuming the first component corresponds to oxygenated hemoglobin
            # try:
            oxy_estimates[i] = coeffs[0] / coeffs.sum(axis=0)
            # except FloatingPointError:
            #     oxy_estimates[i] = np.nan
        return oxy_estimates


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import nrrd
    # Example usage
    config = {
        'estimation_type': 'proxy',
        "unmixing_wavelengths": np.arange(700, 851, 10)
    }
    estimator = LinearUnmixingOxyEstimator(config)
    path = "/home/kris/Data/Dye_project/PAT_Data/iThera_2_data/US_analysis/Study_25/Forearm_1.hdf5"
    reconstruction = sp.load_data_field(path, sp.Tags.DATA_FIELD_RECONSTRUCTED_DATA)
    wavelengths = np.arange(700, 851, 10)
    reconstruction_array = np.stack([np.rot90(reconstruction[str(wl)][:, :, ...], 3) for wl in wavelengths])
    labels, _ = nrrd.read("/home/kris/Data/Dye_project/PAT_Data/iThera_2_data/US_analysis/Study_25/Scan_25_pa-labels.nrrd")
    labels = np.squeeze(labels)
    wavelengths = np.arange(700, 851, 10)
    oxy_estimates = estimator.estimate(reconstruction_array)

    plt.subplot(1, 2, 1)
    plt.imshow(reconstruction_array[0, :, :])
    plt.imshow(labels.T, alpha=0.2)
    plt.subplot(1, 2, 2)
    plt.imshow(oxy_estimates)
    plt.show()
