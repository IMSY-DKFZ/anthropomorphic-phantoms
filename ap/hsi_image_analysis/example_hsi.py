import os
import nrrd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from htc import DataPath
from ap.linear_unimxing import unmix_so2_proxy
from ap.utils.correlate_spectrum_to_oxies import correlate_spectrum
import json
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

measurements_path = os.path.join(base_path, "Measured_Spectra")

examples_images = {
    1: {"oxy": 0.5, "path": "2024_02_20_15_28_13"},
    2: {"oxy": 0.3, "path": "2024_02_20_15_44_02"},
    3: {"oxy": 0, "path": "2024_02_20_16_12_35"},
    4: {"oxy": 0.7, "path": "2024_02_20_16_24_05"},
    5: {"oxy": 1, "path": "2024_02_20_16_50_30"},
}

for forearm_nr, forearm_specs in examples_images.items():
    path = os.path.join(base_path, "HSI_Data", forearm_specs["path"])
    htc_data = DataPath(path)
    rgb = htc_data.read_rgb_reconstructed()
    hsi = htc_data.read_cube()

    training_labels, _ = nrrd.read(os.path.join(path, f"{forearm_specs['path']}-roi.nrrd"))
    labels, _ = nrrd.read(os.path.join(path, f"{forearm_specs['path']}-labels.nrrd"))
    data_shape = labels.shape

    if len(training_labels.shape) == 3:
        training_labels = training_labels[:, :, 0]

    if forearm_nr == 1:
        fig, axes = plt.subplots(1, 1, figsize=(4, 3))
        axes = (axes, )
    else:
        fig, axes = plt.subplots(2, 1, figsize=(8, 7))
        axes[0].set_title('Image and ROI boundaries')

    axes[0].imshow(rgb)
    axes[0].imshow(labels, alpha=0.2)
    axes[0].contour(training_labels)
    axes[0].axes.xaxis.set_visible(False)
    axes[0].axes.yaxis.set_visible(False)

    vessel = - np.log10(hsi[training_labels == 3])
    vessel_spectrum = np.mean(vessel, axis=0)
    vessel_std = np.std(vessel, axis=0)
    hsi_wavelengths = np.arange(500, 1000, 5)
    wavelengths = np.arange(700, 851, 10)
    absorption = np.interp(wavelengths, hsi_wavelengths, vessel_spectrum)
    absorption_std = np.interp(wavelengths, hsi_wavelengths, vessel_std)

    slope, intercept, r_value, p_value, std_err, target_spectrum, target_std = correlate_spectrum(
        absorption,
        wavelengths=wavelengths,
        oxy=forearm_specs["oxy"],
        measurements_path=measurements_path)

    json_path = os.path.join(base_path, "Paper_Results", "HSI_Measurement_Correlation",
                             f"HSI_spectrum_correlation_oxy_{int(100*forearm_specs['oxy']):0d}.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    json.dump({"slope": slope, "intercept": intercept, "r_value": r_value, "p_value": p_value, "std_err": std_err},
              open(json_path, "w"), indent=4)
    print(f"Slope: {slope:.2f}, Intercept: {intercept:.2f}, R-value: {r_value:.2f}, "
          f"p-value: {p_value:.2f}, Std Error: {std_err:.2f}")

    p_value_for_legend = f"p-value={p_value:.2f}" if p_value > 0.01 else f"p-value<0.01"
    unmixed_result = unmix_so2_proxy(
        absorption * slope + intercept, wavelengths=wavelengths, path_to_spectra=measurements_path)

    if forearm_nr == 1:
        save_path = os.path.join(base_path, "Paper_Results", "HSI_Measurement_Correlation",
                                 f"HSI_example_image.pdf")
        plt.savefig(save_path,
                    bbox_inches="tight", pad_inches=0, dpi=400)
        plt.close()
        fig, axes = plt.subplots(1, 1, figsize=(8, 4))
        axes = (None, axes)

    axes[1].set_title(
        f"Target spectrum (sO$_2$: {int(100 * forearm_specs['oxy']):d}%) with unmixed sO$_2$: "
        f"{int(100 * unmixed_result):d} %")
    axes[1].set_xlabel(f"HS signal adapted with Lambert-Beer approx. [a.u.]")
    axes[1].set_ylabel("Absorption coefficient [$cm^{-1}$]")
    axes[1].scatter(absorption, target_spectrum, color="#337538", label=f"Measured HS signal")
    axes[1].plot(sorted(absorption), np.array(sorted(absorption)) * slope + intercept, color="black",
                 label=f"Correlation (R={r_value:.2f}, {p_value_for_legend})")

    ins = axes[1].inset_axes((0.7, 0.2, 0.2, 0.2))
    ins.plot(wavelengths, absorption*slope + intercept, color="#337538")
    ins.plot(wavelengths, target_spectrum, label="Measured material absorption", color="#3266FF")
    ins.fill_between(wavelengths, target_spectrum - target_std,
                     target_spectrum + target_std,
                     alpha=0.2, color="#3266FF")
    ins.set_title("HSI approx. for $\mu_a$")
    ins.set_ylabel("$\mu_a$ [$cm^{-1}$]")
    ins.set_xlabel("Wavelength [nm]")

    lgd = plt.legend(loc="upper left")
    ax = lgd.axes
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D(xdata=[0], ydata=[0], color="#3266FF"))
    labels.append("Measured material absorption")
    lgd._legend_box = None
    lgd._init_legend_box(handles, labels)
    lgd._set_loc(lgd._loc)
    lgd.set_title(lgd.get_title().get_text())

    fig.tight_layout()

    save_path = os.path.join(base_path, "Paper_Results", "HSI_Measurement_Correlation",
                             f"HSI_spectrum_correlation_oxy_{int(100*forearm_specs['oxy']):0d}.pdf")
    plt.savefig(save_path,
                bbox_inches="tight", pad_inches=0, dpi=400)
    plt.close()
