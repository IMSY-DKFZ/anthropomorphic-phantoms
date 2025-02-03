import os
import numpy as np
import torch
from collections import OrderedDict
from ap.utils.io_iad_results import load_iad_results


def get_measured_spectra(spectra_dir: str, unmixing_wavelengths: np.array):
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
