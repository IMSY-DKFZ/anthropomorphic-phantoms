import numpy as np
import simpa as sp
from scipy.optimize import nnls
from ap.oxy_estimation.oxy_estimator_base import OxyEstimator


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
    import os
    plt.switch_backend("TkAgg")
    # Example usage
    base_path = "/path/to/publication_data"

    config = {
        "estimation_type": "proxy",
        "unmixing_wavelengths": np.arange(700, 851, 10),
        "spectra_path": os.path.join(base_path, "Measured_Spectra")
    }
    estimator = LinearUnmixingOxyEstimator(config)

    file_path = os.path.join(base_path, "PAT_Data", "Phantom_01", "Scan_25")

    reconstruction = sp.load_data_field(file_path + "_recon.hdf5", sp.Tags.DATA_FIELD_RECONSTRUCTED_DATA)
    wavelengths = np.arange(700, 851, 10)
    reconstruction_array = np.stack([np.rot90(reconstruction[str(wl)][:, :, ...], 3) for wl in wavelengths])
    labels, _ = nrrd.read(file_path + "_pa-labels.nrrd")
    labels = np.squeeze(labels)
    wavelengths = np.arange(700, 851, 10)
    oxy_estimates = estimator.estimate(reconstruction_array)

    device_pos = 152

    plt.subplot(1, 2, 1)
    plt.imshow(reconstruction_array[0, :, :])
    plt.imshow(labels.T[device_pos:device_pos+200, :], alpha=0.2)
    plt.subplot(1, 2, 2)
    plt.imshow(oxy_estimates)
    plt.show()
