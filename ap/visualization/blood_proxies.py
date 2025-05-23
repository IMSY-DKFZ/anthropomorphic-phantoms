from ap.utils.io_iad_results import load_iad_results
import matplotlib.pyplot as plt
import simpa as sp
import numpy as np
import os
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

dye_spectra_dir = os.path.join(base_path, "Measured_Spectra")

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

print(f"MAE of Hb proxy: {np.mean(np.abs(hb_spectrum - deoxy_absorption_spectrum))}")
print(f"MAE of HbO2 proxy: {np.mean(np.abs(hbo2_spectrum - oxy_absorption_spectrum))}")

fig = plt.figure(figsize=(8, 3))
plt.plot(unmixing_wavelengths, hbo2_spectrum, label="HbO$_{2}$", color="red", linestyle="--")
plt.plot(unmixing_wavelengths, oxy_absorption_spectrum, label="HbO$_{2}$ dye (IR-1061)", color="red")
plt.fill_between(unmixing_wavelengths, oxy_absorption_spectrum - oxy_absorption_std,
                 oxy_absorption_spectrum + oxy_absorption_std, color="red", alpha=0.4)

plt.plot(unmixing_wavelengths, hb_spectrum, label="Hb", color="teal", linestyle="--")
plt.plot(unmixing_wavelengths, deoxy_absorption_spectrum, label="Hb-dye (Spectrasense-765)", color="teal")
plt.fill_between(unmixing_wavelengths, deoxy_absorption_spectrum - deoxy_absorption_std,
                 deoxy_absorption_spectrum + deoxy_absorption_std, color="teal", alpha=0.4)

plt.ylabel("Absorption coefficient $\mu_a$ [$cm^{-1}$]")
plt.xlabel("Wavelength [nm]")
fig.legend(loc='upper center', ncol=4, frameon=False, fontsize="small", fancybox=True, bbox_to_anchor=(0.5, 1.01))
save_path = os.path.join(base_path, "Paper_Results", "Plots", "blood_abs.pdf")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path,
            dpi=400, bbox_inches="tight", pad_inches=0, transparent=False)
# plt.show()
plt.close()

# plt.subplot(1, 2, 2)
fig = plt.figure(figsize=(5, 4))
# plt.plot(unmixing_wavelengths, blood_scatter, label="Blood", color="red", linestyle="--")
plt.plot(unmixing_wavelengths, oxy_scatter_spectrum, label="HbO$_{2}$ dye (IR-1061)", color="red")
plt.fill_between(unmixing_wavelengths, oxy_scatter_spectrum - oxy_scatter_std,
                 oxy_scatter_spectrum + oxy_scatter_std, color="red", alpha=0.4)
plt.plot(unmixing_wavelengths, deoxy_scatter_spectrum, label="Spectrasense-765", color="teal")
plt.fill_between(unmixing_wavelengths, deoxy_scatter_spectrum - deoxy_scatter_std,
                 deoxy_scatter_spectrum + deoxy_scatter_std, color="teal", alpha=0.4)

plt.ylabel("Scattering coefficient $\mu_s$ [$cm^{-1}$]")
plt.xlabel("Wavelength [nm]")
fig.legend(loc='upper center', ncol=4, frameon=False, fontsize="small", fancybox=True, bbox_to_anchor=(0.5, 1.01))
# plt.legend(fancybox=True, framealpha=0)

save_path = os.path.join(base_path, "Paper_Results", "Plots", "blood_scat.pdf")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path,
            dpi=400, bbox_inches="tight", pad_inches=0, transparent=False)
# plt.show()
plt.close()

