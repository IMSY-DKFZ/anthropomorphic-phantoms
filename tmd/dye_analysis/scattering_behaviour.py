import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tmd.dye_analysis import DyeColors, DyeNames, Reds, Yellows, Blues, Brights, Darks

RAW_PATH = "/home/kris/Work/Data/TMD/DyeSpectra/Scattering_meas"

Dyes = ["B09", "B06", "B19"]

if __name__ == "__main__":
    plt.figure(figsize=(12, 8))
    for phantom_name in Dyes:
        out_files = glob.glob(os.path.join(RAW_PATH, phantom_name + "*", "*", "1.txt"))

        abs_spectra = list()
        scat_spectra = list()

        for out_file in out_files:
            wavelength = np.loadtxt(out_file, comments='#', usecols=0)

            mu_a_est = np.loadtxt(out_file, comments='#', usecols=5)
            mu_sp_est = np.loadtxt(out_file, comments='#', usecols=6)
            err_code = np.loadtxt(out_file, dtype=np.dtype(str), comments=None, usecols=9)

            abs_spectra.append(mu_a_est[err_code == "*"] * 10)
            scat_spectra.append(mu_sp_est[err_code == "*"] * 100 * 0.3)

        mua = np.nanmean(np.array(abs_spectra), axis=0)
        mua_std = np.nanstd(np.array(abs_spectra), axis=0)
        mus = np.nanmean(np.array(scat_spectra), axis=0)
        mus_std = np.nanstd(np.array(scat_spectra), axis=0)
        wavelengths = wavelength[err_code == "*"]

        plt.subplot(1, 2, 1)
        ax = plt.gca()
        plt.semilogy(wavelengths, mua, color=DyeColors[phantom_name],
                     label=f"{phantom_name} ({DyeNames[phantom_name]})")
        plt.title("Optical absorption")
        plt.fill_between(wavelengths, mua, mua + mua_std, color=DyeColors[phantom_name], alpha=0.5)
        plt.fill_between(wavelengths, mua, mua - mua_std, color=DyeColors[phantom_name], alpha=0.5)
        plt.ylabel("Absorption coefficient $\mu_a'$ [$cm^{-1}$]")
        plt.xlabel("Wavelength [nm]")
        plt.legend()

        plt.subplot(1, 2, 2)
        ax = plt.gca()
        plt.semilogy(wavelengths, mus, color=DyeColors[phantom_name],
                     label=f"{phantom_name} ({DyeNames[phantom_name]})")
        plt.title(f"Optical scattering, g=0.7")
        plt.fill_between(wavelengths, mus, mus + mus_std, color=DyeColors[phantom_name], alpha=0.5)
        plt.fill_between(wavelengths, mus, mus - mus_std, color=DyeColors[phantom_name], alpha=0.5)
        plt.ylabel("Reduced scattering coefficient $\mu_s$ [$cm^{{-1}}$]")
        plt.xlabel("Wavelength [nm]")
        plt.legend()

    plt.tight_layout()
    plt.savefig("/home/kris/Work/Data/TMD/Plots/Scattering.png")
