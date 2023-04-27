import numpy as np
import scipy.linalg as linalg
from scipy.optimize import nnls
from collections import OrderedDict


def linear_spectral_unmixing(data_to_unmix: np.array,
                             endmembers_dictionary: OrderedDict,
                             non_negativity_constraint: bool = True,
                             return_type="array"):
    """Runs linear unmixing for a spectrum with specific endmember options

    :param data_to_unmix: target data
    :param endmembers_dictionary: OrderedDict of the endmembers that should be in the data to unmix
    :param non_negativity_constraint: flag to indicate whether a concentration can be negative
    :param return_type: either 'dict' or 'array'
    :return: either dictionary with endmember names as keys or an array in the same order as the endmembers dict
    """

    absorption_matrix = np.stack(list(endmembers_dictionary.values())).T

    if non_negativity_constraint:
        output, ris = nnls(absorption_matrix, data_to_unmix)
    else:
        pseudo_inverse_absorption_matrix = linalg.pinv(absorption_matrix)
        output = np.matmul(pseudo_inverse_absorption_matrix, data_to_unmix)

    if return_type == "dict":
        unmixing_result = dict()
        for e_idx, endmember in enumerate(endmembers_dictionary):
            unmixing_result[endmember] = output[e_idx]
    else:
        unmixing_result = output
    return unmixing_result


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import simpa as sp

    unmixing_wavelengths = np.arange(700, 901)

    hbo2_spectrum, hb_spectrum = sp.get_simpa_internal_absorption_spectra_by_names(
        [sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_OXYHEMOGLOBIN, sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_DEOXYHEMOGLOBIN]
    )

    wavelengths = hb_spectrum.wavelengths

    hb_spectrum, hbo2_spectrum = hb_spectrum.values, hbo2_spectrum.values

    hb_spectrum = np.interp(unmixing_wavelengths, wavelengths, hb_spectrum)
    hbo2_spectrum = np.interp(unmixing_wavelengths, wavelengths, hbo2_spectrum)

    mixed_blood = (0.3 * hbo2_spectrum + 0.7 * hb_spectrum) * (np.random.random(size=np.shape(hb_spectrum)) * 0.4 + 1)
    chromophore_spectra_dict = OrderedDict()
    chromophore_spectra_dict["hbO2"] = hbo2_spectrum
    chromophore_spectra_dict["hb"] = hb_spectrum

    result = linear_spectral_unmixing(mixed_blood, chromophore_spectra_dict)
    print(result)
    result /= np.linalg.norm(result, ord=1)
    print(result)

    plt.subplot(2, 2, 1)
    plt.semilogy(hbo2_spectrum)
    plt.title("HbO$_2$")
    plt.subplot(2, 2, 2)
    plt.semilogy(hb_spectrum)
    plt.title("Hb")
    plt.subplot(2, 2, 3)
    plt.title("Mixed blood, ratio 0.3:0.7")
    plt.semilogy(mixed_blood)
    plt.subplot(2, 2, 4)
    plt.title(f"Unmixing result, ratio {result[0]:.2}:{result[1]:.2}")
    plt.semilogy(result[0] * hbo2_spectrum + result[1] * hb_spectrum)
    plt.show()
