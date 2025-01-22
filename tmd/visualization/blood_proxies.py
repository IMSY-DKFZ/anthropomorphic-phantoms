from tmd.utils.io_iad_results import load_iad_results
from tmd.dye_analysis import DyeColors, DyeNames
import matplotlib.pyplot as plt
import simpa as sp
import numpy as np
import os

unmixing_wavelengths = np.arange(700, 855, 1)

hbo2_spectrum, hb_spectrum = sp.get_simpa_internal_absorption_spectra_by_names(
    [sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_OXYHEMOGLOBIN, sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_DEOXYHEMOGLOBIN]
)

wavelengths = hb_spectrum.wavelengths
hb_spectrum, hbo2_spectrum = hb_spectrum.values, hbo2_spectrum.values

hb_spectrum = np.interp(unmixing_wavelengths, wavelengths, hb_spectrum)
hbo2_spectrum = np.interp(unmixing_wavelengths, wavelengths, hbo2_spectrum)
blood_scatter = sp.ScatteringSpectrumLibrary().get_spectrum_by_name("blood_scattering")
blood_scatter = np.interp(unmixing_wavelengths, blood_scatter.wavelengths, blood_scatter.values)

dye_spectra_dir = "/home/kris/Data/Dye_project/Measured_Spectra"

oxy_data = load_iad_results(os.path.join(dye_spectra_dir, f"BIR3.npz"))
oxy_absorption_spectrum = np.interp(unmixing_wavelengths, np.arange(650, 950), oxy_data["mua"])
oxy_absorption_std = np.interp(unmixing_wavelengths, np.arange(650, 950), oxy_data["mua_std"])
oxy_scatter_spectrum = np.interp(unmixing_wavelengths, np.arange(650, 950), oxy_data["mus"])
oxy_scatter_std = np.interp(unmixing_wavelengths, np.arange(650, 950), oxy_data["mus_std"])

deoxy_data = load_iad_results(os.path.join(dye_spectra_dir, f"BS0.npz"))
deoxy_absorption_spectrum = np.interp(unmixing_wavelengths, np.arange(650, 950), deoxy_data["mua"])
deoxy_absorption_std = np.interp(unmixing_wavelengths, np.arange(650, 950), deoxy_data["mua_std"])
deoxy_scatter_spectrum = np.interp(unmixing_wavelengths, np.arange(650, 950), deoxy_data["mus"])
deoxy_scatter_std = np.interp(unmixing_wavelengths, np.arange(650, 950), deoxy_data["mus_std"])



plt.figure(figsize=(5, 4))
plt.plot(unmixing_wavelengths, hbo2_spectrum, label="HbO2", color="red")#, linestyle="--")
# plt.plot(unmixing_wavelengths, oxy_absorption_spectrum, label="HbO2 dye (IR-1061)", color="red")
# plt.fill_between(unmixing_wavelengths, oxy_absorption_spectrum - oxy_absorption_std,
#                  oxy_absorption_spectrum + oxy_absorption_std, color="red", alpha=0.1)

plt.plot(unmixing_wavelengths, hb_spectrum, label="Hb", color="teal")#, linestyle="--")
# plt.plot(unmixing_wavelengths, deoxy_absorption_spectrum, label="Hb-dye (Spectrasense-765)", color="teal")
# plt.fill_between(unmixing_wavelengths, deoxy_absorption_spectrum - deoxy_absorption_std,
#                  deoxy_absorption_spectrum + deoxy_absorption_std, color="teal", alpha=0.1)

plt.ylabel("Absorption coefficient $\mu_a$ [$cm^{-1}$]")
plt.xlabel("Wavelength [nm]")
plt.legend(fancybox=True, framealpha=0)

# plt.subplot(1, 2, 2)
# plt.plot(unmixing_wavelengths, blood_scatter, label="Blood", color="red")
# plt.plot(unmixing_wavelengths, oxy_scatter_spectrum, label="IR-1061", color=DyeColors["B30"])
# plt.fill_between(unmixing_wavelengths, oxy_scatter_spectrum - oxy_scatter_std,
#                  oxy_scatter_spectrum + oxy_scatter_std, color=DyeColors["B30"], alpha=0.25)
# plt.plot(unmixing_wavelengths, deoxy_scatter_spectrum, label="Spectrasense-765", color="teal")
# plt.fill_between(unmixing_wavelengths, deoxy_scatter_spectrum - deoxy_scatter_std,
#                  deoxy_scatter_spectrum + deoxy_scatter_std, color="teal", alpha=0.25)
#
# plt.ylabel("Scattering coefficient $\mu_s$ [$cm^{-1}$]")
# plt.xlabel("Wavelength [nm]")
# plt.legend(fancybox=True, framealpha=0)

plt.tight_layout()
plt.savefig(f"/home/kris/Data/Dye_project/Plots/blood.png", dpi=400, transparent=False)
plt.show()
plt.close()


