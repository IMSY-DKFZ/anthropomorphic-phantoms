from tmd.linear_unimxing import linear_spectral_unmixing
from tmd.load_iad_results import load_iad_results
from collections import OrderedDict
import matplotlib.pyplot as plt
import simpa as sp
import numpy as np
import os
from sympy.utilities.iterables import multiset_permutations

unmixing_wavelengths = np.arange(650, 950)

hbo2_spectrum, hb_spectrum = sp.get_simpa_internal_absorption_spectra_by_names(
    [sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_OXYHEMOGLOBIN, sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_DEOXYHEMOGLOBIN]
)
wavelengths = hb_spectrum.wavelengths
hb_spectrum, hbo2_spectrum = hb_spectrum.values, hbo2_spectrum.values
target_spectrum = 0.1*np.interp(unmixing_wavelengths, wavelengths, hb_spectrum)

dye_spectra_dir = r"C:\Users\adm-dreherk\Documents\Cambridge\Dye Measurements\Example_spectra"
example_spectra = os.listdir(dye_spectra_dir)[:5]

chromophore_spectra_dict = OrderedDict()
for example_spectrum in example_spectra:
    abs_spectrum = load_iad_results(os.path.join(dye_spectra_dir, example_spectrum))["mua"]
    chromophore_spectra_dict[example_spectrum.split(".")[0]] = abs_spectrum

permutations = multiset_permutations(chromophore_spectra_dict.keys(), size=4)

for permutation in permutations:
    perm_dict = OrderedDict({key: chromophore_spectra_dict[key] for key in permutation})
    result = linear_spectral_unmixing(target_spectrum, perm_dict, non_negativity_constraint=False)

    print(result)
    result /= np.linalg.norm(result, ord=1)
    print(result)

    endmember_spectra = list(perm_dict.values())

    resulting_spectrum = 0
    for c_idx, concentration in enumerate(result):
        resulting_spectrum += concentration * endmember_spectra[c_idx]

    plt.subplot(2, 4, 1)
    plt.semilogy(endmember_spectra[0])
    plt.subplot(2, 4, 2)
    plt.semilogy(endmember_spectra[1])
    plt.subplot(2, 4, 3)
    plt.semilogy(endmember_spectra[2])
    plt.subplot(2, 4, 4)
    plt.semilogy(endmember_spectra[3])
    plt.subplot(2, 2, 3)
    plt.semilogy(target_spectrum)
    plt.title("Target spectrum")
    plt.subplot(2, 2, 4)
    plt.title("Resulting spectrum")
    plt.semilogy(resulting_spectrum)
    plt.show()

