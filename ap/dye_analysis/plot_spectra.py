import os
import glob
from ap.dye_analysis import DyeColors, DyeNames
from ap.utils.io_iad_results import load_iad_results
import matplotlib.pyplot as plt
import numpy as np
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

spectrum_files = sorted(glob.glob(os.path.join(base_path, "Measured_Spectra", "*.npz")))

excluded_phantoms = [f"BF{i}" for i in range(1, 10)] + ["BF10A", "BF10B", "BF10C", "BIR", "B90", "B93", "B95", "B97"]
yellow_reds = ["B07", "B11", "B16", "B20", "B21", "B22", "B24", "B25", "B12", "B33"]
darks = ["B10", "B14", "B18", "B27", "B32", "B37", "B23", "B42"]
strongs = ["B05", "B06", "B09", "B15", "B17", "B19", "B30", "B43"]

groups = {"yellow_reds": yellow_reds,
          "darks": darks,
          "strongs": strongs}

for name, ls in groups.items():
    print(name, len(ls))

assert len(set(yellow_reds + darks + strongs)) == len(yellow_reds) + len(darks) + len(strongs)

draft_version = True

for group_name, group in groups.items():
    if draft_version:
        fig = plt.figure(figsize=(10, 5))
    else:
        fig = plt.figure(figsize=(6, 5))
    for file in spectrum_files:
        phantom_name = os.path.basename(file).split(".")[0]
        if phantom_name not in group:
            print(phantom_name)
            continue
        for coefficient, desc, name in zip(["mua", "mus"], ["mu_a", "mu_s"], ["Absorption", "Scattering"]):
            if draft_version and coefficient == "mua":
                plt.subplot(1, 2, 1)
            elif draft_version and coefficient == "mus":
                plt.subplot(1, 2, 2)

            data_dict = load_iad_results(file_path=file)

            wavelengths = data_dict["wavelengths"]
            wl_indices = np.where((wavelengths >= 700) & (wavelengths <= 850))
            wavelengths = wavelengths[wl_indices]
            mua = data_dict[coefficient][wl_indices]
            mua_std = data_dict[coefficient + "_std"][wl_indices]
            g = data_dict["g"]

            try:
                if int(phantom_name[1:]) >= 46:
                    continue
            except ValueError:
                pass
            if any(substring in phantom_name for substring in ["BF", "BJ", "BI", "B9", "BR", "BS", "B40", "B41", "B31", "39", "B38"]):
                continue

            linestyle = "-"
            alpha = 0.3
            # if phantom_name not in ["B43", "B30"]:
            #     linestyle = "--"
            #     alpha = 0.3

            ax = plt.gca()
            if group_name not in ["yellow_reds", "darks"]:
                ax.set_yscale("log", base=10)

            plt.plot(wavelengths, mua, color=DyeColors[phantom_name], label=f"{DyeNames[phantom_name]}",
                     linestyle=linestyle)
            plt.fill_between(wavelengths, mua, mua + mua_std, color=DyeColors[phantom_name], alpha=alpha)
            plt.fill_between(wavelengths, mua, mua - mua_std, color=DyeColors[phantom_name], alpha=alpha)
            plt.ylabel(f"{name} coefficient $\{desc}$ [$cm^{{-1}}$]")
            sub_plot = "(b)" if coefficient == "mus" else "(a)"
            plt.xlabel(r"Wavelength [nm]"
                       "\n"
                       + sub_plot
                       )
            if coefficient == "mus":
                plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=8, fancybox=True, frameon=False, framealpha=0.5)

        plt.tight_layout()
        if draft_version:
            save_path = os.path.join(base_path, "Paper_Results", "Plots", f"All_Spectra_draft_{group_name}.pdf")
        else:
            save_path = os.path.join(base_path, "Paper_Results", "Plots", f"All_Spectra_{name}_{group_name}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if not draft_version:
            plt.savefig(save_path,
                    dpi=300, bbox_inches="tight")
    plt.savefig(save_path,
                dpi=300, bbox_inches="tight")
    plt.close()
    # plt.show()