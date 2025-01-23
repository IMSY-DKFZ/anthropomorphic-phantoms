from ap.utils.io_iad_results import load_iad_results
from ap.dye_analysis import DyeColors, DyeNames
import matplotlib.pyplot as plt
import simpa as sp
import numpy as np
import os
plt.rcParams.update({'font.size': 12,
                     "font.family": "serif"})
unmixing_wavelengths = np.arange(700, 855, 10)

dye_spectra_dir = "/home/kris/Data/Dye_project/Measured_Spectra"

oxy_data = load_iad_results(os.path.join(dye_spectra_dir, f"BIR3.npz"))
oxy_absorption_spectrum = np.interp(unmixing_wavelengths, np.arange(650, 950), oxy_data["mua"])
oxy_absorption_std = np.interp(unmixing_wavelengths, np.arange(650, 950), oxy_data["mua_std"])
oxy_scatter_spectrum = np.interp(unmixing_wavelengths, np.arange(650, 950), oxy_data["mus"])
oxy_scatter_std = np.interp(unmixing_wavelengths, np.arange(650, 950), oxy_data["mus_std"])

deoxy_data_orig = load_iad_results(os.path.join(dye_spectra_dir, f"BS0.npz"))
deoxy_absorption_spectrum_orig = np.interp(unmixing_wavelengths, np.arange(650, 950), deoxy_data_orig["mua"])
deoxy_absorption_std_orig = np.interp(unmixing_wavelengths, np.arange(650, 950), deoxy_data_orig["mua_std"])

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


fig = plt.figure(figsize=(10, 5))

# plt.plot(unmixing_wavelengths, deoxy_absorption_spectrum_orig, label="Hb-dye (Spectrasense-765)", color="teal")
# plt.fill_between(unmixing_wavelengths, deoxy_absorption_spectrum_orig - deoxy_absorption_std,
#                  deoxy_absorption_spectrum_orig + deoxy_absorption_std, color="teal", alpha=0.1)

plt.subplot(1, 2, 1)

plt.plot(unmixing_wavelengths, deoxy_absorption_spectrum, label="0% oxy: Mix 90:10", color="blue")
plt.fill_between(unmixing_wavelengths, deoxy_absorption_spectrum - deoxy_absorption_std,
                 deoxy_absorption_spectrum + deoxy_absorption_std, color="blue", alpha=0.1)

plt.plot(unmixing_wavelengths, abs_spectrum_B93, label="30.7% oxy: Mix 93:7", color=DyeColors["B93"],
         linestyle="--")
plt.fill_between(unmixing_wavelengths, abs_spectrum_B93 - abs_spectrum_B93_std,
                 abs_spectrum_B93 + abs_spectrum_B93_std, color=DyeColors["B93"], alpha=0.1)

plt.plot(unmixing_wavelengths, abs_spectrum_B95, label="52.4% oxy: Mix 95:5", color=DyeColors["B95"],
         linestyle="--")
plt.fill_between(unmixing_wavelengths, abs_spectrum_B95 - abs_spectrum_B95_std,
                 abs_spectrum_B95 + abs_spectrum_B95_std, color=DyeColors["B95"], alpha=0.1)

plt.plot(unmixing_wavelengths, abs_spectrum_B97, label="67.4% oxy: Mix 97:3", color=DyeColors["B97"],
         linestyle="--")
plt.fill_between(unmixing_wavelengths, abs_spectrum_B97 - abs_spectrum_B97_std,
                 abs_spectrum_B97 + abs_spectrum_B97_std, color=DyeColors["B97"], alpha=0.1)

plt.plot(unmixing_wavelengths, oxy_absorption_spectrum, label="100% oxy: HbO2 dye", color="r")
plt.fill_between(unmixing_wavelengths, oxy_absorption_spectrum - oxy_absorption_std,
                 oxy_absorption_spectrum + oxy_absorption_std, color="r", alpha=0.1)

plt.ylabel("Absorption coefficient $\mu_a$ [$cm^{-1}$]")
plt.xlabel("Wavelength [nm]")
fig.legend(loc='upper center', ncol=5, frameon=False, fontsize="small", fancybox=True, bbox_to_anchor=(0.5, 1.04))

plt.subplot(1, 2, 2)

plt.plot(unmixing_wavelengths, deoxy_scatter_spectrum, label="0% oxy: Mix 90:10", color="blue")
plt.fill_between(unmixing_wavelengths, deoxy_scatter_spectrum - deoxy_scatter_std,
                 deoxy_scatter_spectrum + deoxy_scatter_std, color="blue", alpha=0.1)

plt.plot(unmixing_wavelengths, scatter_spectrum_B93, label="30.7% oxy: Mix 93:7", color=DyeColors["B93"],
         linestyle="--")
plt.fill_between(unmixing_wavelengths, scatter_spectrum_B93 - scatter_spectrum_B93_std,
                 scatter_spectrum_B93 + scatter_spectrum_B93_std, color=DyeColors["B93"], alpha=0.1)

plt.plot(unmixing_wavelengths, scatter_spectrum_B95, label="52.4% oxy: Mix 95:5", color=DyeColors["B95"],
         linestyle="--")
plt.fill_between(unmixing_wavelengths, scatter_spectrum_B95 - scatter_spectrum_B95_std,
                 scatter_spectrum_B95 + scatter_spectrum_B95_std, color=DyeColors["B95"], alpha=0.1)

plt.plot(unmixing_wavelengths, scatter_spectrum_B97, label="67.4% oxy: Mix 97:3", color=DyeColors["B97"],
         linestyle="--")
plt.fill_between(unmixing_wavelengths, scatter_spectrum_B97 - scatter_spectrum_B97_std,
                 scatter_spectrum_B97 + scatter_spectrum_B97_std, color=DyeColors["B97"], alpha=0.1)

plt.plot(unmixing_wavelengths, oxy_scatter_spectrum, label="100% oxy: HbO2 dye", color="r")
plt.fill_between(unmixing_wavelengths, oxy_scatter_spectrum - oxy_scatter_std,
                 oxy_scatter_spectrum + oxy_scatter_std, color="r", alpha=0.1)

plt.ylabel("Scattering coefficient $\mu_a$ [$cm^{-1}$]")
plt.xlabel("Wavelength [nm]")
# plt.legend(fancybox=True, framealpha=0, loc="upper right")

# plt.plot(unmixing_wavelengths, oxy_scatter_spectrum, label="IR-1061", color=DyeColors["B30"])
# plt.fill_between(unmixing_wavelengths, oxy_scatter_spectrum - oxy_scatter_std,
#                  oxy_scatter_spectrum + oxy_scatter_std, color=DyeColors["B30"], alpha=0.25)
#
# plt.ylabel("Scattering coefficient $\mu_s$ [$cm^{-1}$]")
# plt.xlabel("Wavelength [nm]")
# plt.legend(fancybox=True, framealpha=0)
plt.tight_layout()
plt.savefig(f"/home/kris/Data/Dye_project/Plots/oxy_levels.png", dpi=400, transparent=False, bbox_inches="tight")
# plt.show()
plt.close()


