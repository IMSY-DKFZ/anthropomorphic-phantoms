from ap.utils.io_iad_results import load_iad_results, load_total_refl_and_transmission
from ap.dye_analysis import DyeColors
import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams.update({'font.size': 12,
                     "font.family": "serif"})
unmixing_wavelengths = np.arange(700, 855, 10)

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

oxy_data = load_iad_results(os.path.join(dye_spectra_dir, f"BIR3.npz"))
oxy_absorption_spectrum = np.interp(unmixing_wavelengths, np.arange(650, 950), oxy_data["mua"])
oxy_absorption_std = np.interp(unmixing_wavelengths, np.arange(650, 950), oxy_data["mua_std"])
oxy_scatter_spectrum = np.interp(unmixing_wavelengths, np.arange(650, 950), oxy_data["mus"])
oxy_scatter_std = np.interp(unmixing_wavelengths, np.arange(650, 950), oxy_data["mus_std"])

deoxy_data = load_iad_results(os.path.join(dye_spectra_dir, "B90.npz"))
deoxy_absorption_spectrum = np.interp(unmixing_wavelengths, np.arange(650, 950), deoxy_data["mua"])
deoxy_absorption_std = np.interp(unmixing_wavelengths, np.arange(650, 950), deoxy_data["mua_std"])
deoxy_scatter_spectrum = np.interp(unmixing_wavelengths, np.arange(650, 950), deoxy_data["mus"])
deoxy_scatter_std = np.interp(unmixing_wavelengths, np.arange(650, 950), deoxy_data["mus_std"])

data_97 = load_iad_results(os.path.join(dye_spectra_dir, "B97.npz"))
abs_spectrum_B97 = np.interp(unmixing_wavelengths, np.arange(650, 950), data_97["mua"])
abs_spectrum_B97_std = np.interp(unmixing_wavelengths, np.arange(650, 950), data_97["mua_std"])
scatter_spectrum_B97 = np.interp(unmixing_wavelengths, np.arange(650, 950), data_97["mus"])
scatter_spectrum_B97_std = np.interp(unmixing_wavelengths, np.arange(650, 950), data_97["mus_std"])

data_95 = load_iad_results(os.path.join(dye_spectra_dir, "B95.npz"))
abs_spectrum_B95 = np.interp(unmixing_wavelengths, np.arange(650, 950), data_95["mua"])
abs_spectrum_B95_std = np.interp(unmixing_wavelengths, np.arange(650, 950), data_95["mua_std"])
scatter_spectrum_B95 = np.interp(unmixing_wavelengths, np.arange(650, 950), data_95["mus"])
scatter_spectrum_B95_std = np.interp(unmixing_wavelengths, np.arange(650, 950), data_95["mus_std"])

data_93 = load_iad_results(os.path.join(dye_spectra_dir, "B93.npz"))
abs_spectrum_B93 = np.interp(unmixing_wavelengths, np.arange(650, 950), data_93["mua"])
abs_spectrum_B93_std = np.interp(unmixing_wavelengths, np.arange(650, 950), data_93["mua_std"])
scatter_spectrum_B93 = np.interp(unmixing_wavelengths, np.arange(650, 950), data_93["mus"])
scatter_spectrum_B93_std = np.interp(unmixing_wavelengths, np.arange(650, 950), data_93["mus_std"])


fig = plt.figure(figsize=(10, 3.6))

plt.subplot(1, 2, 1)

plt.plot(unmixing_wavelengths, deoxy_absorption_spectrum, label="0% sO$_2$: Mix 90:10", color="blue")
plt.fill_between(unmixing_wavelengths, deoxy_absorption_spectrum - deoxy_absorption_std,
                 deoxy_absorption_spectrum + deoxy_absorption_std, color="blue", alpha=0.1)

plt.plot(unmixing_wavelengths, abs_spectrum_B93, label="30.7% sO$_2$: Mix 93:7", color=DyeColors["B93"],
         linestyle="--")
plt.fill_between(unmixing_wavelengths, abs_spectrum_B93 - abs_spectrum_B93_std,
                 abs_spectrum_B93 + abs_spectrum_B93_std, color=DyeColors["B93"], alpha=0.1)

plt.plot(unmixing_wavelengths, abs_spectrum_B95, label="52.4% sO$_2$: Mix 95:5", color=DyeColors["B95"],
         linestyle="--")
plt.fill_between(unmixing_wavelengths, abs_spectrum_B95 - abs_spectrum_B95_std,
                 abs_spectrum_B95 + abs_spectrum_B95_std, color=DyeColors["B95"], alpha=0.1)

plt.plot(unmixing_wavelengths, abs_spectrum_B97, label="67.4% sO$_2$: Mix 97:3", color=DyeColors["B97"],
         linestyle="--")
plt.fill_between(unmixing_wavelengths, abs_spectrum_B97 - abs_spectrum_B97_std,
                 abs_spectrum_B97 + abs_spectrum_B97_std, color=DyeColors["B97"], alpha=0.1)

plt.plot(unmixing_wavelengths, oxy_absorption_spectrum, label="100% sO$_2$: HbO$_{2}$ dye", color="r")
plt.fill_between(unmixing_wavelengths, oxy_absorption_spectrum - oxy_absorption_std,
                 oxy_absorption_spectrum + oxy_absorption_std, color="r", alpha=0.1)

plt.ylabel("Absorption coefficient $\mu_a$ [$cm^{-1}$]")
plt.xlabel(r"Wavelength [nm]"
           "\n"
           "(a)")
fig.legend(loc='upper center', ncol=5, frameon=False, fontsize="small", fancybox=True, bbox_to_anchor=(0.5, 1.04))

plt.subplot(1, 2, 2)

plt.plot(unmixing_wavelengths, deoxy_scatter_spectrum, label="0% sO$_2$: Mix 90:10", color="blue")
plt.fill_between(unmixing_wavelengths, deoxy_scatter_spectrum - deoxy_scatter_std,
                 deoxy_scatter_spectrum + deoxy_scatter_std, color="blue", alpha=0.1)

plt.plot(unmixing_wavelengths, scatter_spectrum_B93, label="30.7% sO$_2$: Mix 93:7", color=DyeColors["B93"],
         linestyle="--")
plt.fill_between(unmixing_wavelengths, scatter_spectrum_B93 - scatter_spectrum_B93_std,
                 scatter_spectrum_B93 + scatter_spectrum_B93_std, color=DyeColors["B93"], alpha=0.1)

plt.plot(unmixing_wavelengths, scatter_spectrum_B95, label="52.4% sO$_2$: Mix 95:5", color=DyeColors["B95"],
         linestyle="--")
plt.fill_between(unmixing_wavelengths, scatter_spectrum_B95 - scatter_spectrum_B95_std,
                 scatter_spectrum_B95 + scatter_spectrum_B95_std, color=DyeColors["B95"], alpha=0.1)

plt.plot(unmixing_wavelengths, scatter_spectrum_B97, label="67.4% sO$_2$: Mix 97:3", color=DyeColors["B97"],
         linestyle="--")
plt.fill_between(unmixing_wavelengths, scatter_spectrum_B97 - scatter_spectrum_B97_std,
                 scatter_spectrum_B97 + scatter_spectrum_B97_std, color=DyeColors["B97"], alpha=0.1)

plt.plot(unmixing_wavelengths, oxy_scatter_spectrum, label="100 sO$_2$: HbO$_{2}$ dye", color="r")
plt.fill_between(unmixing_wavelengths, oxy_scatter_spectrum - oxy_scatter_std,
                 oxy_scatter_spectrum + oxy_scatter_std, color="r", alpha=0.1)

plt.ylabel("Scattering coefficient $\mu_s$ [$cm^{-1}$]")
plt.xlabel(r"Wavelength [nm]"
            "\n"
           "(b)")

plt.tight_layout()
save_path = os.path.join(base_path, "Paper_Results", "Plots", "oxy_levels.pdf")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path,
            dpi=400, bbox_inches="tight", pad_inches=0, transparent=False)
# plt.show()
plt.close()

fig = plt.figure(figsize=(10, 3))
for plt_idx, (oxy_level_name, label, color) in enumerate(zip(["90", "93", "95", "97", "100"],
                                        ["0% sO$_2$: Mix 90:10", "30.7 sO$_2$: Mix 93:7", "52.4% sO$_2$: Mix 95:5",
                                         "67.4% sO$_2$: Mix 97:3", "100% sO$_2$: HbO$_{2}$ dye"],
                                        ["blue", DyeColors["B93"], DyeColors["B95"], DyeColors["B97"], "red"])):

    data = load_total_refl_and_transmission(os.path.join(dye_spectra_dir, oxy_level_name))
    wavelengths = data["wavelengths"]
    transmission = np.interp(unmixing_wavelengths, wavelengths, data["transmission_mean"])
    transmission_std = np.interp(unmixing_wavelengths, wavelengths, data["transmission_std"])
    reflectance = np.interp(unmixing_wavelengths, wavelengths, data["reflectance_mean"])
    reflectance_std = np.interp(unmixing_wavelengths, wavelengths, data["reflectance_std"])

    plt.subplot(1, 2, 1)
    plt.plot(unmixing_wavelengths, reflectance, label=label, color=color, linestyle="--" if 0 < plt_idx < 4 else "-")
    plt.fill_between(unmixing_wavelengths, reflectance - reflectance_std, reflectance + reflectance_std,
                     color=color, alpha=0.1)
    plt.ylabel("Reflectance [a.u.]")
    plt.xlabel("Wavelength [nm]")
    if plt_idx == 4:
        fig.legend(loc='upper center', ncol=5, frameon=False, fontsize="small", fancybox=True, bbox_to_anchor=(0.5, 1.04))

    plt.subplot(1, 2, 2)
    plt.plot(unmixing_wavelengths, transmission, color=color, linestyle="--" if 0 < plt_idx < 4 else "-")
    plt.fill_between(unmixing_wavelengths, transmission - transmission_std, transmission + transmission_std,
                     color=color, alpha=0.1)
    plt.ylabel("Transmission [a.u.]")
    plt.xlabel("Wavelength [nm]")


plt.tight_layout()
save_path = os.path.join(base_path, "Paper_Results", "Plots", "oxy_levels_measurements.pdf")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path,
            dpi=400, transparent=False, pad_inches=0, bbox_inches="tight")
# plt.show()
plt.close()


