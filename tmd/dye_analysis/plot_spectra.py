import os
import glob
from tmd.dye_analysis import DyeColors, DyeNames, Reds, Yellows, Blues, Brights, Darks
from tmd.utils.io_iad_results import load_iad_results
import matplotlib.pyplot as plt
# plt.style.use('bmh')

dye_base_dir = "/home/kris/Work/Data/TMD/DyeSpectra/Measured_Spectra"
spectrum_files = glob.glob(os.path.join(dye_base_dir, "*.npz"))
plt.figure(figsize=(12, 8))
for file in spectrum_files:
    phantom_name = os.path.basename(file).split(".")[0]
    data_dict = load_iad_results(file_path=file)

    wavelengths = data_dict["wavelengths"]
    mua = data_dict["mua"]
    mua_std = data_dict["mua_std"]
    mus = data_dict["mus"]
    mus_std = data_dict["mus_std"]
    g = data_dict["g"]

    if phantom_name not in ["B06", "B09", "B19"]:
        continue

    linestyle = "-"
    if phantom_name in ["BJG", "B30"]:
        linestyle = "--"

    plt.subplot(1, 2, 1)
    ax = plt.gca()
    # ax.set_facecolor("lightgrey")
    plt.semilogy(wavelengths, mua, color=DyeColors[phantom_name], label=f"{phantom_name} ({DyeNames[phantom_name]})",
                 linestyle=linestyle)
    plt.title("Optical absorption")
    plt.fill_between(wavelengths, mua, mua + mua_std, color=DyeColors[phantom_name], alpha=0.5)
    plt.fill_between(wavelengths, mua, mua - mua_std, color=DyeColors[phantom_name], alpha=0.5)
    plt.ylabel("Absorption coefficient $\mu_a'$ [$cm^{-1}$]")
    plt.xlabel("Wavelength [nm]")
    plt.legend()

    plt.subplot(1, 2, 2)
    ax = plt.gca()
    # ax.set_facecolor("lightgrey")
    plt.semilogy(wavelengths, mus, color=DyeColors[phantom_name], label=f"{phantom_name} ({DyeNames[phantom_name]})",
                 linestyle=linestyle)
    plt.title(f"Optical scattering, g={g:.1}")
    plt.fill_between(wavelengths, mus, mus + mus_std, color=DyeColors[phantom_name], alpha=0.5)
    plt.fill_between(wavelengths, mus, mus - mus_std, color=DyeColors[phantom_name], alpha=0.5)
    plt.ylabel("Reduced scattering coefficient $\mu_s$ [$cm^{{-1}}$]")
    plt.xlabel("Wavelength [nm]")
    plt.legend()

plt.tight_layout()
plt.savefig("/home/kris/Work/Data/TMD/Plots/weird_dyes.png")
