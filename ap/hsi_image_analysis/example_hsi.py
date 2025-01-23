import os
import glob
import nrrd
import htc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from htc import Config, DataPath, DatasetImage, LabelMapping, settings, settings_atlas, tivita_wavelengths
from ap.linear_unimxing import unmix_so2_proxy
from ap.utils.io_iad_results import load_iad_results

from scipy.stats import linregress
import json

examples_images = {
    1: {"oxy": 0.5, "path": "2024_02_20_15_28_13"},
    2: {"oxy": 0.3, "path": f"2024_02_20_15_44_02"},
    3: {"oxy": 0, "path": "2024_02_20_16_12_35"},
    4: {"oxy": 0.7, "path": "2024_02_20_16_24_05"},
    5: {"oxy": 1, "path": "2024_02_20_16_50_30"},
    # 6: {"oxy": 1, "path": "2024_02_20_17_18_56"},
    # 6: {"oxy": 1, "path": "2024_02_20_15_58_27"},
}

measurements_path = "/home/kris/Data/Dye_project/Measured_Spectra"

abs_dict = {
    0: "B90",
    0.3: "B93",
    0.5: "B95",
    0.7: "B97",
    1: "BIR",
}

base_path = "/home/kris/Data/Dye_project/HSI_Data/"

for forearm_nr, forearm_specs in examples_images.items():
    path = os.path.join(base_path, forearm_specs["path"])
    htc_data = DataPath(path)
    rgb = htc_data.read_rgb_reconstructed()
    hsi = htc_data.read_cube()
    # training_labels = np.zeros(rgb.shape[:2], dtype=np.uint8)
    # training_labels[173:176, 202:208] = 3
    # nrrd.write(os.path.join(path, f"{forearm_specs['path']}-roi.nrrd"), training_labels)
    # nrrd.write(os.path.join(path, f"{forearm_specs['path']}-hsi.nrrd"), hsi)
    # exit()
    training_labels, _ = nrrd.read(os.path.join(path, f"{forearm_specs['path']}-roi.nrrd"))
    labels, _ = nrrd.read(os.path.join(path, f"{forearm_specs['path']}-labels.nrrd"))
    data_shape = labels.shape

    if len(training_labels.shape) == 3:
        training_labels = training_labels[:, :, 0]

    fig, axes = plt.subplots(2, 1, figsize=(8, 7))
    axes[0].imshow(rgb)
    axes[0].imshow(labels, alpha=0.2)
    axes[0].contour(training_labels)
    axes[0].axes.xaxis.set_visible(False)
    axes[0].axes.yaxis.set_visible(False)

    axes[0].set_title('Image and ROI boundaries')
    vessel = - np.log10(hsi[training_labels == 3])
    vessel_spectrum = np.mean(vessel, axis=0)
    vessel_std = np.std(vessel, axis=0)
    hsi_wavelengths = np.arange(500, 1000, 5)
    wavelengths = np.arange(700, 851, 10)
    absorption = np.interp(wavelengths, hsi_wavelengths, vessel_spectrum)
    absorption_std = np.interp(wavelengths, hsi_wavelengths, vessel_std)
    # absorption = - np.log10(vessel_spectrum)
    # absorption_std = - np.log10(vessel_std)

    hb_abs_spectrum = load_iad_results(os.path.join(measurements_path, "B90.npz"))["mua"]
    hb_abs_spectrum = np.interp(wavelengths, np.arange(650, 950), hb_abs_spectrum)

    hbo2_abs_spectrum = load_iad_results(os.path.join(measurements_path, "BIR.npz"))["mua"]
    hbo2_abs_spectrum = np.interp(wavelengths, np.arange(650, 950), hbo2_abs_spectrum)

    target_spectrum = load_iad_results(os.path.join(measurements_path, abs_dict[forearm_specs["oxy"]] + ".npz"))["mua"]
    target_spectrum = np.interp(wavelengths, np.arange(650, 950), target_spectrum)

    target_std = load_iad_results(os.path.join(measurements_path, abs_dict[forearm_specs["oxy"]] + ".npz"))["mua_std"]
    target_std = np.interp(wavelengths, np.arange(650, 950), target_std)

    slope, intercept, r_value, p_value, std_err = linregress(absorption, target_spectrum)

    json.dump({"slope": slope, "intercept": intercept, "r_value": r_value, "p_value": p_value, "std_err": std_err},
              open(path + ".json", "w"))
    print(slope, intercept, r_value, p_value, std_err)

    # axes[1].plot(wavelengths, target_spectrum, label="Measured material absorption", color="blue")
    # axes[1].fill_between(wavelengths, target_spectrum - target_std,
    #                      target_spectrum + target_std,
    #                      alpha=0.2, color="blue")
    # axes[1].plot(wavelengths, intercept + slope * absorption, color="green",
    #              label=f"Correlation (R={r_value:.2f}, p-value={p_value:.2f})")
    # axes[1].set_title(f"Spectrum with unmixed oxy: {unmix_so2_proxy(intercept + slope * absorption):.2f}")
    # axes[1].set_xlabel(f"Wavelength [nm]")
    # axes[1].set_ylabel(f"Absorption Coefficient [$cm^{-1}$]")
    # plt.legend()

    axes[1].set_title(
        f"Target spectrum (oxy={int(100 * forearm_specs['oxy']):d}%) with unmixed oxy: "
        f"{unmix_so2_proxy(absorption * slope + intercept, wavelengths=wavelengths):.2f} %")
    axes[1].set_xlabel(f"HS signal adapted with Lambert-Beer approx. [a.u.]")
    axes[1].set_ylabel(f"Absorption coefficient [$cm^{-1}$]")
    axes[1].scatter(absorption, target_spectrum, color="green", label=f"Measured HS signal")
    axes[1].plot(sorted(absorption), np.array(sorted(absorption)) * slope + intercept, color="black",
             label=f"Correlation (R={r_value:.2f}, p-value={p_value:.2f})")

    ins = axes[1].inset_axes((0.7, 0.2, 0.2, 0.2))
    ins.plot(wavelengths, absorption*slope + intercept, color="green")
    ins.plot(wavelengths, target_spectrum, label="Measured material absorption", color="blue")
    ins.fill_between(wavelengths, target_spectrum - target_std,
                     target_spectrum + target_std,
                     alpha=0.2, color="blue")
    ins.set_title("HSI approx. for $\mu_a$")
    ins.set_ylabel(f"$\mu_a$ [$cm^{-1}$]")
    ins.set_xlabel("Wavelength [nm]")

    lgd = plt.legend(loc="upper left")
    ax = lgd.axes
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D(xdata=[0], ydata=[0], color="blue"))
    labels.append("Measured material absorption")
    lgd._legend_box = None
    lgd._init_legend_box(handles, labels)
    lgd._set_loc(lgd._loc)
    lgd.set_title(lgd.get_title().get_text())

    fig.tight_layout()

    # plt.show()
    # exit()
    plt.savefig(os.path.join("/home/kris/Pictures/Phantom_Paper_Figures/", f"HSI_spectrum_correlation_oxy_{int(100*forearm_specs['oxy']):0d}.png"), dpi=300)
    plt.close()
