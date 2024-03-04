import os
import glob
from tmd.dye_analysis import DyeColors, DyeNames, Reds, Yellows, Blues, Brights, Darks
from tmd.utils.io_iad_results import load_iad_results
import matplotlib.pyplot as plt
import simpa as sp
import numpy as np
# plt.style.use('bmh')

dye_base_dir = "/home/kris/Work/Data/TMD/DyeSpectra/Measured_Spectra"
spectrum_files = glob.glob(os.path.join(dye_base_dir, "*.npz"))
plt.figure(figsize=(12, 8))
# hbo2_spectrum, hb_spectrum = sp.get_simpa_internal_absorption_spectra_by_names(
#     [sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_OXYHEMOGLOBIN, sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_DEOXYHEMOGLOBIN]
# )
# wavelengths = hb_spectrum.wavelengths
# hb_spectrum, hbo2_spectrum = hb_spectrum.values, hbo2_spectrum.values
#
# target_spectrum = np.interp(np.arange(650, 951), wavelengths, hb_spectrum)
# plt.subplot(1, 2, 1)
# plt.semilogy(np.arange(650, 951), target_spectrum, color="blue", label="Hb")
for file in spectrum_files:
    phantom_name = os.path.basename(file).split(".")[0]
    data_dict = load_iad_results(file_path=file)

    wavelengths = data_dict["wavelengths"]
    mua = data_dict["mua"]
    mua_std = data_dict["mua_std"]
    mus = data_dict["mus"]
    mus_std = data_dict["mus_std"]
    g = data_dict["g"]

    if phantom_name not in ["B43"]:
        continue

    linestyle = "-"
    if phantom_name in ["BJG", "B30"]:
        linestyle = "--"

    plt.subplot(1, 2, 1)
    ax = plt.gca()
    # ax.set_facecolor("lightgrey")
    plt.plot(wavelengths, mua, color=DyeColors[phantom_name], label=f"{phantom_name} ({DyeNames[phantom_name]})",
                 linestyle=linestyle)
    plt.title("Optical absorption")
    plt.fill_between(wavelengths, mua, mua + mua_std, color=DyeColors[phantom_name], alpha=0.5)
    plt.fill_between(wavelengths, mua, mua - mua_std, color=DyeColors[phantom_name], alpha=0.5)
    plt.ylabel("Absorption coefficient $\mu_a$ [$cm^{-1}$]")
    plt.xlabel("Wavelength [nm]")
    plt.legend()

    plt.subplot(1, 2, 2)
    ax = plt.gca()
    # ax.set_facecolor("lightgrey")
    plt.plot(wavelengths, mus, color=DyeColors[phantom_name], label=f"{phantom_name} ({DyeNames[phantom_name]})",
                 linestyle=linestyle)
    plt.title(f"Optical scattering, g={g:.1}")
    plt.fill_between(wavelengths, mus, mus + mus_std, color=DyeColors[phantom_name], alpha=0.5)
    plt.fill_between(wavelengths, mus, mus - mus_std, color=DyeColors[phantom_name], alpha=0.5)
    plt.ylabel("Reduced scattering coefficient $\mu_s'$ [$cm^{{-1}}$]")
    plt.xlabel("Wavelength [nm]")
    plt.legend()

plt.tight_layout()
plt.savefig("/home/kris/Work/Data/TMD/Plots/Spectrasense.png")
