import os
import numpy as np
import torch
from collections import OrderedDict
from ap.utils.io_iad_results import load_iad_results


def get_measured_spectra(spectra_dir: str, unmixing_wavelengths: np.array):
    """
    Retrieve and interpolate measured absorption spectra from files in a directory.

    This function scans the specified directory for measured spectrum files,
    filters out files that contain any of the substrings "BF", "BI", or "BS",
    and further restricts the selection to files where the integer value extracted
    from characters at positions 1 to 3 of the filename is less than 50.
    For each valid file, the function performs the following steps:

    - Loads the absorption spectrum data from the file (expected to be stored under the key "mua"
      in an .npz file) using the ``load_iad_results`` function.
    - Interpolates the spectrum from its native wavelength grid (assumed to be 650 nm to 949 nm)
      to the wavelengths specified in ``unmixing_wavelengths``.
    - Stores the interpolated spectrum in an ordered dictionary with a key derived from the
      filename (the portion before the file extension).

    :param spectra_dir: Directory path containing the measured spectrum files.
    :type spectra_dir: str

    :param unmixing_wavelengths: A 1D NumPy array specifying the target wavelengths (in nm)
                                 to which each spectrum will be interpolated.
    :type unmixing_wavelengths: np.array

    :return: An OrderedDict mapping each spectrum's name (extracted from the filename without
             its extension) to its corresponding interpolated absorption spectrum (as a NumPy array).
    :rtype: collections.OrderedDict

    :raises ValueError: May be raised if the filename format is unexpected such that conversion
                        of the substring at positions 1 to 3 to an integer fails.
    """
    example_spectra = sorted(os.listdir(spectra_dir))
    example_spectra = [spectrum for spectrum in example_spectra if
                       ("BF" not in spectrum and "BI" not in spectrum and "BS" not in spectrum)
                       and int(spectrum[1:3]) < 50]

    chromophore_spectra_dict = OrderedDict()

    for dye_idx, example_spectrum in enumerate(example_spectra):
        spectrum_name = example_spectrum.split(".")[0]
        abs_spectrum = load_iad_results(os.path.join(spectra_dir, example_spectrum))["mua"]
        abs_spectrum = np.interp(unmixing_wavelengths, np.arange(650, 950), abs_spectrum)
        chromophore_spectra_dict[spectrum_name] = abs_spectrum

    return chromophore_spectra_dict
