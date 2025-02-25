from ap.utils.io_iad_results import load_iad_results
from ap.dye_analysis import DyeColors
import matplotlib.pyplot as plt
import simpa as sp
import numpy as np
import os
plt.rcParams.update({'font.size': 12,
                     "font.family": "serif"})

try:
    run_by_bash: bool = bool(os.environ["RUN_BY_BASH"])
    print("This runner script is invoked in a bash script!")
except KeyError:
    run_by_bash: bool = False

if run_by_bash:
    base_path = os.environ["BASE_PATH"]
else:
    # In case the script is run from an IDE, the base path has to be set manually
    base_path = ""

dye_spectra_dir = os.path.join(base_path, "Measured_Spectra")
unmixing_wavelengths = np.arange(700, 855, 10)

hbo2_spectrum, hb_spectrum = sp.get_simpa_internal_absorption_spectra_by_names(
    [sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_OXYHEMOGLOBIN, sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_DEOXYHEMOGLOBIN]
)
wavelengths = hb_spectrum.wavelengths
hb_spectrum, hbo2_spectrum = hb_spectrum.values, hbo2_spectrum.values

hb_spectrum = np.interp(unmixing_wavelengths, wavelengths, hb_spectrum)
hbo2_spectrum = np.interp(unmixing_wavelengths, wavelengths, hbo2_spectrum)

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
        fig = plt.figure(figsize=(11, 4))
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)
        ax1.set_xlabel(r"Wavelength [nm]"
                       "\n"
                       "(a)")
        ax2.set_xlabel("Wavelength [nm]"
                       "\n"
                       "(b)"
                       )
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
             # label=f"Forearm {fore_nr}: bvf: {fore_dict['bvf']}%, oxy: {fore_dict['oxy']}%",
             color=color_dict["color"][fore_dict["oxy"]],
             alpha=color_dict["alpha"][fore_dict["bvf"]])
    ax2.fill_between(unmixing_wavelengths, scatter_spectrum - scatter_std, scatter_spectrum + scatter_std,
                     color=color_dict["color"][fore_dict["oxy"]],
                     alpha=color_dict["alpha"][fore_dict["bvf"]]/2)

    if f_idx in [2, 5, 8, 11]:
        fig.legend(loc='upper center', ncol=3, frameon=False, fontsize="small", fancybox=True, bbox_to_anchor=(0.5, 1.01))
        # ax2.legend(fancybox=True, framealpha=0)
        save_path = os.path.join(base_path, "Paper_Results", "Plots", f"forearms_{f_idx-1}_{f_idx+1}.pdf")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # plt.tight_layout()
        plt.savefig(save_path,
                    dpi=400, bbox_inches="tight", pad_inches=0, transparent=False)
        plt.close()
