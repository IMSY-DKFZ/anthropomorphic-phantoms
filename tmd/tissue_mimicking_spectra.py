from tmd.linear_unimxing import linear_spectral_unmixing
from tmd.utils.io_iad_results import load_iad_results
from tmd.dye_analysis import DyeColors, DyeNames
from collections import OrderedDict
import matplotlib.pyplot as plt
import simpa as sp
import numpy as np
import os
from itertools import combinations

unmixing_wavelengths = np.arange(700, 850)
target_spectrum_name = "Hb"

hbo2_spectrum, hb_spectrum = sp.get_simpa_internal_absorption_spectra_by_names(
    [sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_OXYHEMOGLOBIN, sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_DEOXYHEMOGLOBIN]
)
wavelengths = hb_spectrum.wavelengths
hb_spectrum, hbo2_spectrum = hb_spectrum.values, hbo2_spectrum.values

target_spectra = {
    "Hb": hb_spectrum,
    "HbO2": hbo2_spectrum,
}

target_spectrum = np.interp(unmixing_wavelengths, wavelengths, target_spectra[target_spectrum_name])

dye_spectra_dir = "/home/kris/Work/Data/TMD/DyeSpectra/Measured_Spectra"
example_spectra = os.listdir(dye_spectra_dir)

chromophore_spectra_dict = OrderedDict()
for example_spectrum in example_spectra:
    abs_spectrum = load_iad_results(os.path.join(dye_spectra_dir, example_spectrum))["mua"]
    chromophore_spectra_dict[example_spectrum.split(".")[0]] = np.interp(unmixing_wavelengths, np.arange(650, 950), abs_spectrum)

combinations = list(combinations(list(chromophore_spectra_dict.keys()), 22))

for permutation in combinations:
    perm_dict = OrderedDict({key: chromophore_spectra_dict[key] for key in permutation})
    result = linear_spectral_unmixing(target_spectrum, perm_dict, non_negativity_constraint=True, weighted_optimization=True)

    # print(result)
    # result /= np.linalg.norm(result, ord=1)
    # spec = 4 * chromophore_spectra_dict["B10"] + chromophore_spectra_dict["B23"]
    # plt.semilogy(spec)
    # plt.show()
    print(result)

    endmember_spectra = list(perm_dict.values())

    nr_of_nonzero_endmembers = np.count_nonzero(result)
    resulting_spectrum = 0
    plt_idx = 1
    plt.figure(figsize=(12, 8))
    for c_idx, concentration in enumerate(result):
        if concentration == 0:
            continue
        plt.subplot(2, 1, 2)
        mixed_spectrum = concentration * endmember_spectra[c_idx]
        endmember_name = list(perm_dict.keys())[c_idx]
        plt.semilogy(unmixing_wavelengths, mixed_spectrum, label=f"{endmember_name} ({DyeNames[endmember_name]}), c={concentration:.1f}", color=DyeColors[endmember_name])
        plt.legend()
        resulting_spectrum += mixed_spectrum
        plt.title(f"Endmembers")
        plt.ylabel("Absorption coefficient $\mu_a'$ [$cm^{-1}$]")
        plt.xlabel("Wavelength [nm]")
        plt.xticks(range(700, 901, 25))
        plt_idx += 1

    plt.subplot(2, 1, 1)
    plt.title(f"Target spectrum {target_spectrum_name}")
    if target_spectrum_name == "Hb":
        c = "blue"
    else:
        c = "red"
    plt.semilogy(unmixing_wavelengths, target_spectrum, label=target_spectrum_name, color=c)
    plt.semilogy(unmixing_wavelengths, resulting_spectrum, label="Mixed spectrum", color="green")
    plt.ylabel("Absorption coefficient $\mu_a'$ [$cm^{-1}$]")
    plt.xlabel("Wavelength [nm]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"/home/kris/Work/Data/TMD/Plots/Mixing_result_{target_spectrum_name}_850.png")
