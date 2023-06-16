import numpy as np
import matplotlib.pyplot as plt
import os


def load_iad_results(file_path: str) -> dict:
    """
    Loads the results from the iad programme and returns a dictionary with the results.

    :param file_path: path to the results of the iad algorithm
    :return: absorption and scattering data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist!")

    results = np.load(file_path, allow_pickle=True)
    results_dict = dict(results)
    return results_dict


def plot_iad_results(file_path: str):
    """
    Plots the absorption and scattering coefficients calculated by the iad programme.

    :param file_path: path to the results of the iad algorithm
    :return:
    """
    data_dict = load_iad_results(file_path=file_path)

    wavelengths = data_dict["wavelengths"]
    mua = data_dict["mua"]
    mua_std = data_dict["mua_std"]
    mus = data_dict["mus"]
    mus_std = data_dict["mus_std"]

    plt.subplot(1, 2, 1)
    plt.plot(wavelengths, mua, color="red")
    plt.title("Optical absorption [$cm^{-1}$]", color="red")
    plt.fill_between(wavelengths, mua, mua + mua_std, color="red", alpha=0.5)
    plt.fill_between(wavelengths, mua, mua - mua_std, color="red", alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(wavelengths, mus, color="blue")
    plt.title("Optical scattering [$cm^{-1}$]", color="blue")
    plt.fill_between(wavelengths, mus, mus + mus_std, color="blue", alpha=0.5)
    plt.fill_between(wavelengths, mus, mus - mus_std, color="blue", alpha=0.5)

    plt.show()


if __name__ == "__main__":
    for i in range(1, 19):
        path = fr"C:\Users\adm-dreherk\Documents\Cambridge\Dye Measurements\Example_spectra\example_spectrum_{i}.npz"
        plot_iad_results(path)
