from ap.utils.io_iad_results import load_iad_results
from ap.dye_analysis import DyeColors, DyeNames
from ap.data.load_icg_absorption import load_icg
from ap.data.load_methylene_blue_absorption import load_mb
from collections import OrderedDict
import matplotlib.pyplot as plt
import simpa as sp
import numpy as np
import os

unmixing_wavelengths = np.arange(700, 850, 10)

hbo2_spectrum, hb_spectrum = sp.get_simpa_internal_absorption_spectra_by_names(
    [sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_OXYHEMOGLOBIN, sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_DEOXYHEMOGLOBIN]
)
wavelengths = hb_spectrum.wavelengths
hb_spectrum, hbo2_spectrum = hb_spectrum.values, hbo2_spectrum.values

hb_spectrum = np.interp(unmixing_wavelengths, wavelengths, hb_spectrum)
hbo2_spectrum = np.interp(unmixing_wavelengths, wavelengths, hbo2_spectrum)

dye_spectra_dir = "/home/kris/Data/Dye_project/Measured_Spectra"

spectrum_B30 = load_iad_results(os.path.join(dye_spectra_dir, "B60.npz"))["mua"]
spectrum_B30 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_B30)
# factor_B30_oxy = 4.985
factor_B30_oxy = 1
factor_B30_deoxy = 1

spectrum_B30_oxy = spectrum_B30 * factor_B30_oxy
spectrum_B30_deoxy = 0#spectrum_B30 * factor_B30_deoxy

spectrum_B15 = load_iad_results(os.path.join(dye_spectra_dir, "B50.npz"))["mua"]
spectrum_B15 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_B15)
factor_B15 = 0.783
spectrum_B15 *= factor_B15

spectrum_B43 = load_iad_results(os.path.join(dye_spectra_dir, "B51.npz"))["mua"]
spectrum_B43 = np.interp(unmixing_wavelengths, np.arange(650, 950), spectrum_B43)
factor_B43 = 1.075
spectrum_B43 *= factor_B43

print(np.mean((spectrum_B43 + spectrum_B15)/spectrum_B30))
print(np.std((spectrum_B43 + spectrum_B15)/spectrum_B30))

plt.figure(figsize=(7, 5))
plt.subplot(2, 1, 1)
plt.plot(unmixing_wavelengths, hb_spectrum, label="Hb", color="blue")
plt.plot(unmixing_wavelengths, spectrum_B30, label="Hb-proxy", color="green")
# plt.plot(unmixing_wavelengths, spectrum_B15 + spectrum_B43, label="Mixture", color="green", linestyle="--")
plt.ylim([1, 10])
plt.ylabel("Absorption coefficient $\mu_a$ [$cm^{-1}$]")
plt.xlabel("Wavelength [nm]")
plt.legend()
# plt.plot(unmixing_wavelengths, hbo2_spectrum, label="HbO2", color="red")
plt.subplot(2, 1, 2)
# plt.plot(unmixing_wavelengths, spectrum_B30_oxy, label="HbO2-proxy", color="red", linestyle="--")
# plt.plot(unmixing_wavelengths, spectrum_B30_deoxy + spectrum_B43 + spectrum_B15, label="Hb-proxy", color="blue", linestyle="--")
# plt.plot(unmixing_wavelengths, spectrum_B30_deoxy, label=f"B30 (IR-1061) c={factor_B30_deoxy}")
plt.plot(unmixing_wavelengths, spectrum_B15, label=f"B15 (Ultramarine) c={factor_B15}")
plt.plot(unmixing_wavelengths, spectrum_B43, label=f"B43 (Spectrasense IR765) c={factor_B43}")

plt.ylabel("Absorption coefficient $\mu_a$ [$cm^{-1}$]")
plt.xlabel("Wavelength [nm]")
plt.legend()
plt.tight_layout()
plt.savefig(f"/home/kris/Data/Dye_project/Plots/Hb_endmembers.png", dpi=400)
plt.close()


