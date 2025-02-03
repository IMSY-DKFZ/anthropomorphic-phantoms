import numpy as np
import simpa as sp
import os
from ap.utils.io_iad_results import load_iad_results
import torch
from ap.utils.generate_random_spectra import generate_tissue_spectra


def get_target_spectrum(target_spectrum_name: str, unmixing_wavelengths: np.array, dye_spectra_dir: str = ""):
    simpa_library = {
        "Hb": sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_DEOXYHEMOGLOBIN,
        "HbO2": sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_OXYHEMOGLOBIN,
        "Water": sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_WATER,
        "Fat": sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_FAT,
        "Melanin": sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_MELANIN,
        "Nickel_Sulphide": sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_NICKEL_SULPHIDE,
        "Copper_Sulphide": sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_COPPER_SULPHIDE
    }

    if target_spectrum_name in ["B90", "B93", "B95", "B97"]:
        spectrum = load_iad_results(os.path.join(dye_spectra_dir, target_spectrum_name + ".npz"))["mua"]
        abs_spectrum = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum)
        target_spectrum = torch.from_numpy(abs_spectrum).type(torch.float32)
    elif target_spectrum_name in simpa_library:
        spectrum = sp.get_simpa_internal_absorption_spectra_by_names(
            [simpa_library[target_spectrum_name]]
        )[0]
        wavelengths = spectrum.wavelengths
        spectrum = spectrum.values

        target_spectrum = np.interp(unmixing_wavelengths, wavelengths, spectrum)
    elif target_spectrum_name == "random":
        target_spectrum = generate_tissue_spectra(nr_of_spectra=1, wavelength_range=unmixing_wavelengths)
    else:
        raise ValueError("Unknown target spectrum name.")

    return target_spectrum
