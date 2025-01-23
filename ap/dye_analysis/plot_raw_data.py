import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from ap.dye_analysis import DyeColors, DyeNames, Reds, Yellows, Blues, Brights, Darks

RAW_PATH = "/home/kris/Work/Data/TMD/DyeSpectra/Scattering_meas/B19A/1_T_IncBeam.txt"
RERFL_PATH = "/home/kris/Work/Data/TMD/DyeSpectra/Scattering_meas/B19A/1_R_RefInPlace.txt"
t_path = "/home/kris/Work/Data/TMD/DyeSpectra/Scattering_meas/B19A/1/1.txt"

wavelengths = np.loadtxt(RAW_PATH, comments='#', usecols=0)
transmission = np.loadtxt(RAW_PATH, comments='#', usecols=1)
refl = np.loadtxt(RERFL_PATH, comments='#', usecols=1)
M_T_meas = np.loadtxt(t_path, comments='#', usecols=3)
M_R_meas = np.loadtxt(t_path, comments='#', usecols=1)

plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.title("Reference")
plt.semilogy(wavelengths, transmission, label="Transmission incident beam")
plt.semilogy(wavelengths, refl, label="Reflectance white standard")
plt.ylabel(r'$Intensity \; \mathrm{[a.u.]}$')
plt.xlabel("Wavelength [nm]")
plt.legend()
plt.subplot(1, 2, 2)
plt.title("Transmission and reflectance")
plt.semilogy(wavelengths, M_T_meas, label="Transmission B19")
plt.semilogy(wavelengths, M_R_meas, label="Reflectance B19")
plt.ylabel(r'$Intensity \; \mathrm{[a.u.]}$')
plt.xlabel("Wavelength [nm]")
plt.legend()
plt.tight_layout()
plt.savefig("/home/kris/Work/Data/TMD/Plots/raw_data.png")

