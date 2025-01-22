import numpy as np
import os
from tmd.utils.io_iad_results import load_iad_results
from scipy.stats import linregress


measurements_path = "/home/kris/Data/Dye_project/Measured_Spectra"
abs_dict = {
    0: "B90",
    0.3: "B93",
    0.5: "B95",
    0.7: "B97",
    1: "BIR",
}


def correlate_spectrum(spectrum, wavelengths, oxy):
    """
    Correlate a given spectrum with target absorption spectra for different oxygenation levels.

    This function loads the absorption spectra for deoxygenated hemoglobin (B90) and oxygenated hemoglobin (BIR),
    interpolates them to match the provided wavelengths, and then correlates the given spectrum with the target
    absorption spectrum for the specified oxygenation level. The correlation is performed using linear regression.

    :param spectrum: The spectrum to be correlated.
    :type spectrum: numpy.ndarray
    :param wavelengths: The wavelengths corresponding to the spectrum.
    :type wavelengths: numpy.ndarray
    :param oxy: The oxygenation level (0, 0.3, 0.5, 0.7, or 1) for which the target spectrum is loaded.
    :type oxy: float

    :return: A tuple containing the slope, intercept, correlation coefficient (r_value), p-value, and standard error
             of the regression.
    :rtype: tuple
    """
    # Load and interpolate the absorption spectrum for deoxygenated hemoglobin (B90)
    hb_abs_spectrum = load_iad_results(os.path.join(measurements_path, "B90.npz"))["mua"]
    hb_abs_spectrum = np.interp(wavelengths, np.arange(650, 950), hb_abs_spectrum)

    # Load and interpolate the absorption spectrum for oxygenated hemoglobin (BIR)
    hbo2_abs_spectrum = load_iad_results(os.path.join(measurements_path, "BIR.npz"))["mua"]
    hbo2_abs_spectrum = np.interp(wavelengths, np.arange(650, 950), hbo2_abs_spectrum)

    # Load and interpolate the target absorption spectrum for the specified oxygenation level
    target_spectrum = load_iad_results(os.path.join(measurements_path, abs_dict[oxy] + ".npz"))["mua"]
    target_spectrum = np.interp(wavelengths, np.arange(650, 950), target_spectrum)

    # Load and interpolate the standard deviation of the target absorption spectrum
    target_std = load_iad_results(os.path.join(measurements_path, abs_dict[oxy] + ".npz"))["mua_std"]
    target_std = np.interp(wavelengths, np.arange(650, 950), target_std)

    # Perform linear regression between the target spectrum and the given spectrum
    slope, intercept, r_value, p_value, std_err = linregress(spectrum, target_spectrum)

    return slope, intercept, r_value, p_value, std_err, target_spectrum, target_std
