import os
import glob
from ap.dye_analysis import DyeColors, DyeNames
from ap.utils.io_iad_results import load_iad_results
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 12,
                     "font.family": "serif"})

base_path = "/home/kris/Data/Dye_project/publication_data"
spectrum_files = sorted(glob.glob(os.path.join(base_path, "Measured_Spectra", "*.npz")))

excluded_phantoms = [f"BF{i}" for i in range(1, 10)] + ["BF10A", "BF10B", "BF10C", "BIR", "B90", "B93", "B95", "B97"]

for coefficient, desc, name in zip(["mua", "mus"], ["mu_a", "mu_s"], ["Absorption", "Scattering"]):
    if coefficient == "mua":
        plt.figure(figsize=(4, 5))
    else:
        plt.figure(figsize=(6, 5))
    for file in spectrum_files:
        phantom_name = os.path.basename(file).split(".")[0]
        data_dict = load_iad_results(file_path=file)

        wavelengths = data_dict["wavelengths"]
        wl_indices = np.where((wavelengths >= 700) & (wavelengths <= 850))
        wavelengths = wavelengths[wl_indices]
        mua = data_dict[coefficient][wl_indices]
        mua_std = data_dict[coefficient + "_std"][wl_indices]
        g = data_dict["g"]

        print(phantom_name)
        try:
            if int(phantom_name[1:]) >= 46:
                continue
        except ValueError:
            pass
        if any(substring in phantom_name for substring in ["BF", "BJ", "BI", "B9", "BR", "BS", "B40", "B41", "B31", "39", "B38"]):
            continue

        linestyle = "-"
        alpha = 0.5
        if phantom_name not in ["B43", "B30"]:
            linestyle = "--"
            alpha = 0.3

        ax = plt.gca()
        ax.set_yscale("log", base=10)

        plt.plot(wavelengths, mua, color=DyeColors[phantom_name], label=f"{DyeNames[phantom_name]}",
                 linestyle=linestyle)
        plt.fill_between(wavelengths, mua, mua + mua_std, color=DyeColors[phantom_name], alpha=alpha)
        plt.fill_between(wavelengths, mua, mua - mua_std, color=DyeColors[phantom_name], alpha=alpha)
        plt.ylabel(f"{name} coefficient $\{desc}$ [$cm^{{-1}}$]")
        plt.xlabel("Wavelength [nm]")
        if coefficient == "mus":
            plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=8, fancybox=True, frameon=True, framealpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(base_path, "Paper_Results", "Plots", f"All_Spectra_{name}.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    # plt.show()