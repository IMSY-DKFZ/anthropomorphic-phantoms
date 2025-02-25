import numpy as np
import pandas as pd
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


def load_total_refl_and_transmission(folder_path: str) -> dict:
    """
    Loads the reflectance and transmission data from the iad programme and returns a dictionary with the results.

    :param folder_path: path to the results of the iad algorithm
    :return: reflectance and transmission data
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The file {folder_path} does not exist!")

    sample_folders = os.listdir(folder_path)
    assert len(sample_folders) == 2, f"Expected two folders in {folder_path}, found {len(sample_folders)}"

    results_dict = dict()
    for sample_folder in sample_folders:
        refl, trans = list(), list()
        sample_folder_path = os.path.join(folder_path, sample_folder)
        measurement_folders = [dirname for dirname in os.listdir(sample_folder_path)
                               if os.path.isdir(os.path.join(sample_folder_path, dirname))]

        assert len(measurement_folders) == 6, (f"Expected six measurement folders in {sample_folder_path}, "
                                               f"found {len(measurement_folders)}")

        for measurement_folder in measurement_folders:
            measurement_file_path = os.path.join(sample_folder_path, measurement_folder, "1.rxt")
            sample_results = pd.read_csv(measurement_file_path, sep=" ", header=None, skiprows=1)
            wavelengths = np.array(sample_results[0])
            refl.append(np.array(sample_results[1]))
            trans.append(np.array(sample_results[2]))
        results_dict[sample_folder] = {"reflectance_mean": np.mean(np.array(refl), axis=0),
                                       "reflectance_std": np.std(np.array(refl), axis=0),
                                       "transmission_mean": np.mean(np.array(trans), axis=0),
                                       "transmission_std": np.std(np.array(trans), axis=0)
                                       }

    refl_means = np.array([results_dict[sample_folder]["reflectance_mean"] for sample_folder in sample_folders])
    refl_stds = np.array([results_dict[sample_folder]["reflectance_std"] for sample_folder in sample_folders])
    trans_means = np.array([results_dict[sample_folder]["transmission_mean"] for sample_folder in sample_folders])
    trans_stds = np.array([results_dict[sample_folder]["transmission_std"] for sample_folder in sample_folders])

    return {"reflectance_mean": np.mean(refl_means, axis=0),
            "reflectance_std": np.mean(refl_stds, axis=0),
            "transmission_mean": np.mean(trans_means, axis=0),
            "transmission_std": np.mean(trans_stds, axis=0),
            "wavelengths": wavelengths
            }


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
    path = ""
    meas_dict = load_total_refl_and_transmission(path)
    transmission = meas_dict["transmission_mean"]
    transmission_std = meas_dict["transmission_std"]
    reflectance = meas_dict["reflectance_mean"]
    reflectance_std = meas_dict["reflectance_std"]
    wavelengths = meas_dict["wavelengths"]

    plt.plot(wavelengths, reflectance, label="Reflectance")
    plt.fill_between(wavelengths, reflectance - reflectance_std, reflectance + reflectance_std, alpha=0.5)
    plt.plot(wavelengths, transmission, label="Transmission")
    plt.fill_between(wavelengths, transmission - transmission_std, transmission + transmission_std, alpha=0.5)
    plt.legend()
    plt.show()
