import numpy as np
import scipy.linalg as linalg
from scipy.optimize import nnls
from collections import OrderedDict
import simpa as sp
from ap.utils.io_iad_results import load_iad_results
import os


def linear_spectral_unmixing(data_to_unmix: np.array,
                             endmembers_dictionary: OrderedDict,
                             non_negativity_constraint: bool = True,
                             return_type="array",
                             weighted_optimization: bool = False,
                             return_so2: bool = False):
    """Runs linear unmixing for a spectrum or an image with specific endmember options

    :param data_to_unmix: target data, expects np array with shapes: 1D spectrum: (nr_wavelengths) or 2D multispectral
    image: (nr_wavelengths, Nx, Ny).
    :param endmembers_dictionary: OrderedDict of the endmembers that should be in the data to unmix
    :param non_negativity_constraint: flag to indicate whether a concentration can be negative
    :param return_type: either 'dict' or 'array'.
    :param return_so2: If set to True and return_type="array", then only a np array will be returned with so2 values.
    :param weighted_optimization:
    :return: either dictionary with endmember names as keys or an array in the same order as the endmembers dict
    """

    data_shape = data_to_unmix.shape
    reshape = True if len(data_shape) > 1 else False

    if weighted_optimization:
        data_to_unmix /= np.max(data_to_unmix)

    absorption_matrix = np.stack(list(endmembers_dictionary.values())).T

    if non_negativity_constraint:
        if reshape:
            output = list()
            data_to_unmix = np.reshape(data_to_unmix, (data_shape[0], -1))
            for i in range(np.prod(data_shape[1:])):
                nnls_out, ris = nnls(absorption_matrix, data_to_unmix[:, i])
                output.append(nnls_out)
            output = np.swapaxes(output, axis1=0, axis2=1)
        else:
            output, ris = nnls(absorption_matrix, data_to_unmix)

    else:
        pseudo_inverse_absorption_matrix = linalg.pinv(absorption_matrix)
        output = np.matmul(pseudo_inverse_absorption_matrix, data_to_unmix)

    if return_type == "dict":
        unmixing_result = dict()
        for e_idx, endmember in enumerate(endmembers_dictionary):
            if not reshape:
                unmixing_result[endmember] = output[e_idx]
            else:
                unmixing_result[endmember] = np.reshape(output[e_idx, :], (data_shape[1:]))
        if return_so2:
            unmixing_result["sO2"] = unmixing_result["hbO2"] / (unmixing_result["hbO2"] + unmixing_result["hb"])
    else:
        unmixing_result = np.reshape(output, (len(endmembers_dictionary), *data_shape[1:]))
    return unmixing_result


def unmix_so2(data_to_unmix, wavelengths: np.ndarray = np.arange(700, 851, 10)):
    hbo2_spectrum, hb_spectrum = sp.get_simpa_internal_absorption_spectra_by_names(
        [sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_OXYHEMOGLOBIN,
         sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_DEOXYHEMOGLOBIN]
    )

    wavelengths_ = hb_spectrum.wavelengths

    hb_spectrum, hbo2_spectrum = hb_spectrum.values, hbo2_spectrum.values

    hb_spectrum = np.interp(wavelengths, wavelengths_, hb_spectrum)
    hbo2_spectrum = np.interp(wavelengths, wavelengths_, hbo2_spectrum)

    chromophore_spectra_dict = OrderedDict()
    chromophore_spectra_dict["hbO2"] = hbo2_spectrum
    chromophore_spectra_dict["hb"] = hb_spectrum

    so2 = linear_spectral_unmixing(data_to_unmix, chromophore_spectra_dict, return_type="dict", return_so2=True)["sO2"]

    if not isinstance(so2, float):
        nan_idx = np.isnan(so2)
        so2[nan_idx] = 0

    return so2


def unmix_so2_proxy(data_to_unmix, wavelengths: np.ndarray = np.arange(700, 851, 10)):
    base_path = "/path/to/publication_data/Measured_Spectra"
    hb_abs_spectrum = load_iad_results(os.path.join(base_path, "B90.npz"))["mua"]
    hb_abs_spectrum = np.interp(wavelengths, np.arange(650, 950), hb_abs_spectrum)

    hbo2_abs_spectrum = load_iad_results(os.path.join(base_path, "BIR.npz"))["mua"]
    hbo2_abs_spectrum = np.interp(wavelengths, np.arange(650, 950), hbo2_abs_spectrum)

    chromophore_spectra_dict = OrderedDict()
    chromophore_spectra_dict["hbO2"] = hbo2_abs_spectrum
    chromophore_spectra_dict["hb"] = hb_abs_spectrum

    so2 = linear_spectral_unmixing(data_to_unmix, chromophore_spectra_dict, return_type="dict", return_so2=True)["sO2"]

    if not isinstance(so2, float):
        nan_idx = np.isnan(so2)
        so2[nan_idx] = 0

    return so2


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    two_dim = True

    unmixing_wavelengths = np.arange(700, 901, 10)

    hbo2_spectrum, hb_spectrum = sp.get_simpa_internal_absorption_spectra_by_names(
        [sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_OXYHEMOGLOBIN, sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_DEOXYHEMOGLOBIN]
    )

    wavelengths = hb_spectrum.wavelengths

    hb_spectrum, hbo2_spectrum = hb_spectrum.values, hbo2_spectrum.values

    hb_spectrum = np.interp(unmixing_wavelengths, wavelengths, hb_spectrum)
    hbo2_spectrum = np.interp(unmixing_wavelengths, wavelengths, hbo2_spectrum)

    mixed_blood = (0.3 * hbo2_spectrum + 0.7 * hb_spectrum)# * (np.random.random(size=np.shape(hb_spectrum)) * 0.4 + 1)
    chromophore_spectra_dict = OrderedDict()
    chromophore_spectra_dict["hbO2"] = hbo2_spectrum
    chromophore_spectra_dict["hb"] = hb_spectrum

    if two_dim:
        mixed_blood_2d = np.ones([21, 50, 50])
        for wl_idx, i in enumerate(mixed_blood):
            mixed_blood_2d[wl_idx, ...] *= i

        mixed_blood = mixed_blood_2d

        result = linear_spectral_unmixing(mixed_blood, chromophore_spectra_dict, return_type="dict", return_so2=True)["sO2"]

        plt.imshow(result)
        plt.show()

    else:
        result = linear_spectral_unmixing(mixed_blood, chromophore_spectra_dict)

        print(result)
        result /= np.linalg.norm(result, ord=1, axis=0)
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

