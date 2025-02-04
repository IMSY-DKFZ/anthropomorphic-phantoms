import numpy as np
import simpa as sp
import os
from ap.utils.io_iad_results import load_iad_results
import torch
from ap.utils.generate_random_spectra import generate_tissue_spectra


def get_target_spectrum(target_spectrum_name: str, unmixing_wavelengths: np.array, dye_spectra_dir: str = ""):
    """
    Retrieve an absorption spectrum interpolated to the specified wavelengths.

    This function returns an absorption spectrum based on the provided
    ``target_spectrum_name`` and interpolates it to the wavelengths given in
    ``unmixing_wavelengths``. Depending on the value of ``target_spectrum_name``,
    the spectrum is obtained from one of three sources:

    - **Dye spectra files**: For ``target_spectrum_name`` in ``["B90", "B93", "B95", "B97"]``,
      the function loads a precomputed spectrum from a corresponding ``.npz`` file located in
      the directory specified by ``dye_spectra_dir``. The loaded spectrum is assumed to cover
      the wavelength range 650 nm to 949 nm and is interpolated to ``unmixing_wavelengths``.
      The resulting spectrum is returned as a PyTorch tensor of type ``torch.float32``.

    - **SIMPA library spectra**: For ``target_spectrum_name`` in
      ``["Hb", "HbO2", "Water", "Fat", "Melanin", "Nickel_Sulphide", "Copper_Sulphide"]``,
      the function retrieves the corresponding internal absorption spectrum via SIMPA and
      interpolates it from its native wavelength grid to ``unmixing_wavelengths``.
      The result is returned as a NumPy array.

    - **Randomly generated spectrum**: If ``target_spectrum_name`` is ``"random"``,
      a synthetic tissue absorption spectrum is generated using ``unmixing_wavelengths``
      as the wavelength range. The generated spectrum is returned in the format produced
      by ``generate_tissue_spectra``.

    :param target_spectrum_name: Identifier for the target spectrum. Supported values include:
        - ``"B90", "B93", "B95", "B97"``: Load spectra from ``.npz`` files located in ``dye_spectra_dir``.
        - ``"Hb", "HbO2", "Water", "Fat", "Melanin", "Nickel_Sulphide", "Copper_Sulphide"``:
          Retrieve the corresponding absorption spectrum from the SIMPA internal library.
        - ``"random"``: Generate a random tissue absorption spectrum.
    :type target_spectrum_name: str

    :param unmixing_wavelengths: A 1D NumPy array specifying the wavelengths (in nm) to which
                                 the retrieved spectrum should be interpolated.
    :type unmixing_wavelengths: np.array

    :param dye_spectra_dir: Directory path containing dye spectra ``.npz`` files.
                              This parameter is used only when ``target_spectrum_name`` is one
                              of ``"B90", "B93", "B95", "B97"``. Defaults to an empty string.
    :type dye_spectra_dir: str

    :return: The absorption spectrum interpolated to ``unmixing_wavelengths``. The return type depends on
             the source:
             - For dye spectra, a PyTorch tensor of type ``torch.float32``.
             - For SIMPA library spectra and the randomly generated spectrum, a NumPy array.
    :rtype: Union[torch.Tensor, np.array]

    :raises ValueError: If ``target_spectrum_name`` does not match any of the supported identifiers.
    """
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
        # Load spectrum from file and interpolate.
        spectrum = load_iad_results(os.path.join(dye_spectra_dir, target_spectrum_name + ".npz"))["mua"]
        abs_spectrum = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum)
        target_spectrum = torch.from_numpy(abs_spectrum).type(torch.float32)
    elif target_spectrum_name in simpa_library:
        # Retrieve SIMPA internal absorption spectrum and interpolate.
        spectrum = sp.get_simpa_internal_absorption_spectra_by_names(
            [simpa_library[target_spectrum_name]]
        )[0]
        wavelengths = spectrum.wavelengths
        spectrum = spectrum.values

        target_spectrum = np.interp(unmixing_wavelengths, wavelengths, spectrum)
    elif target_spectrum_name == "random":
        # Generate a random tissue absorption spectrum.
        target_spectrum = generate_tissue_spectra(nr_of_spectra=1, wavelength_range=unmixing_wavelengths)
    else:
        raise ValueError("Unknown target spectrum name.")

    return target_spectrum
