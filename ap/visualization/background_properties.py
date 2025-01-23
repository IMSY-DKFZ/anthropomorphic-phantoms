from ap.utils.io_iad_results import load_iad_results
from ap.dye_analysis import DyeColors, DyeNames
import matplotlib.pyplot as plt
import simpa as sp
import numpy as np
import os

unmixing_wavelengths = np.arange(700, 855, 10)

hbo2_spectrum, hb_spectrum = sp.get_simpa_internal_absorption_spectra_by_names(
    [sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_OXYHEMOGLOBIN, sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_DEOXYHEMOGLOBIN]
)
wavelengths = hb_spectrum.wavelengths
hb_spectrum, hbo2_spectrum = hb_spectrum.values, hbo2_spectrum.values

hb_spectrum = np.interp(unmixing_wavelengths, wavelengths, hb_spectrum)
hbo2_spectrum = np.interp(unmixing_wavelengths, wavelengths, hbo2_spectrum)

dye_spectra_dir = "/home/kris/Data/Dye_project/Measured_Spectra"

background_spectra = {
    "1": {"oxy": 0, "bvf": 2.5},
    "2": {"oxy": 50, "bvf": 2.5},
    "3": {"oxy": 100, "bvf": 2.5},
    "4": {"oxy": 100, "bvf": 4},
    "5": {"oxy": 50, "bvf": 4},
    "6": {"oxy": 0, "bvf": 4},
    "7": {"oxy": 100, "bvf": 1},
    "8": {"oxy": 50, "bvf": 1},
    "9": {"oxy": 0, "bvf": 1},
    "10A": {"oxy": 100, "bvf": 0.5},
    "10B": {"oxy": 0, "bvf": 5},
    "10C": {"oxy": 70, "bvf": 3},
}

color_dict = {
    "color": {
        0: "blue",
        50: DyeColors["B95"],
        70: DyeColors["B97"],
        100: "red",
    },
    "alpha": {
        0.5: 0.25,
        1: 0.5,
        2.5: 0.75,
        3: 0.9,
        4: 1,
        5: 1,
    }
}

for f_idx, (fore_nr, fore_dict) in enumerate(background_spectra.items()):
    absorption_spectrum = load_iad_results(os.path.join(dye_spectra_dir, f"BF{fore_nr}.npz"))["mua"]
    absorption_spectrum = np.interp(unmixing_wavelengths, np.arange(650, 950), absorption_spectrum)

    absorption_std = load_iad_results(os.path.join(dye_spectra_dir, f"BF{fore_nr}.npz"))["mua_std"]
    absorption_std = np.interp(unmixing_wavelengths, np.arange(650, 950), absorption_std)

    scatter_spectrum = load_iad_results(os.path.join(dye_spectra_dir, f"BF{fore_nr}.npz"))["mus"]
    scatter_spectrum = np.interp(unmixing_wavelengths, np.arange(650, 950), scatter_spectrum)

    scatter_std = load_iad_results(os.path.join(dye_spectra_dir, f"BF{fore_nr}.npz"))["mus_std"]
    scatter_std = np.interp(unmixing_wavelengths, np.arange(650, 950), scatter_std)

    if f_idx in [0, 3, 6, 9]:
        plt.figure(figsize=(10, 4))
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)
        ax1.set_title("Absorption")
        ax2.set_title("Scattering (g=0.7)")
        ax1.set_xlabel("Wavelength [nm]")
        ax2.set_xlabel("Wavelength [nm]")
        ax1.set_ylabel("Absorption coefficient [cm⁻¹]")
        ax2.set_ylabel("Scattering coefficient [cm⁻¹]")

    ax1.plot(unmixing_wavelengths, absorption_spectrum,
             label=f"Forearm {fore_nr}: bvf: {fore_dict['bvf']}%, oxy: {fore_dict['oxy']}%",
             color=color_dict["color"][fore_dict["oxy"]],
             alpha=color_dict["alpha"][fore_dict["bvf"]])
    ax1.fill_between(unmixing_wavelengths, absorption_spectrum - absorption_std, absorption_spectrum + absorption_std,
                     color=color_dict["color"][fore_dict["oxy"]],
                     alpha=color_dict["alpha"][fore_dict["bvf"]]/2)

    ax2.plot(unmixing_wavelengths, scatter_spectrum,
             label=f"Forearm {fore_nr}: bvf: {fore_dict['bvf']}%, oxy: {fore_dict['oxy']}%",
             color=color_dict["color"][fore_dict["oxy"]],
             alpha=color_dict["alpha"][fore_dict["bvf"]])
    ax2.fill_between(unmixing_wavelengths, scatter_spectrum - scatter_std, scatter_spectrum + scatter_std,
                     color=color_dict["color"][fore_dict["oxy"]],
                     alpha=color_dict["alpha"][fore_dict["bvf"]]/2)

    if f_idx in [2, 5, 8, 11]:
        ax1.legend(fancybox=True, framealpha=0)
        ax2.legend(fancybox=True, framealpha=0)
        plt.savefig(f"/home/kris/Data/Dye_project/Plots/forearms_{f_idx-1}_{f_idx+1}.png",
                    dpi=400, transparent=False)
        plt.close()
