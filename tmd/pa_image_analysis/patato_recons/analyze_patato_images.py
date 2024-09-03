import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os
import patato as pat
from tmd.utils.io_iad_results import load_iad_results
from tmd.linear_unimxing import unmix_so2_proxy


recon_method = "mb"

base_path = f"/home/kris/Data/Dye_project/PAT_Data/Example_reconstructions_patato_{recon_method}/"
# examples_images = {
#     1: {"oxy": 0.5, "path": os.path.join("Study_6", "Scan_22.npz")},
#     # 2: {"oxy": 0.3, "path": os.path.join("Study_7", "Scan_11.npz")},
#     # 3: {"oxy": 0, "path": os.path.join("Study_8", "Scan_17.npz")},
#     # 4: {"oxy": 0.7, "path": os.path.join("Study_9", "Scan_4.npz")},
#     # 5: {"oxy": 1, "path": os.path.join("Study_10", "Scan_8.npz")},
#     # 6: {"oxy": 1, "path": os.path.join("Study_11", "Scan_19.npz")},
# }

examples_images = {
    # 1: {"oxy": 0.5, "path": os.path.join("Study_17", "Scan_1.npz")},
    3: {"oxy": 0, "path": os.path.join("Study_18", "Scan_1.npz")},
    # 5: {"oxy": 1, "path": os.path.join("Study_19", "Scan_1.npz")},
    # 6: {"oxy": 1, "path": os.path.join("Study_20", "Scan_1.npz")},
}

wavelengths = np.arange(700, 851, 10)

for forearm_nr, forearm_specs in examples_images.items():
    path = os.path.join(base_path, forearm_specs["path"])
    pa_data = np.load(path, allow_pickle=True)

    reconstruction_array = pa_data["recon"]
    reconstruction_array = np.rot90(reconstruction_array[:16], 2, axes=(1, 2))

    training_labels = np.zeros(reconstruction_array.shape[1:], dtype=np.uint8)
    # training_labels[129:130, 215:217] = 3   # for image nr 3
    # training_labels[122:124, 152:154] = 3   # for image nr 4
    # training_labels[135:136, 176:179] = 3 for image nr 6
    # training_labels[139:140, 251:257] = 3   # for example image nr 5
    training_labels[131:132, 184:186] = 3
    # For image nr. 6: training_labels[135:137, 175:185] = 3
    # nrrd.write(os.path.join(path, f"{forearm_specs['path']}-roi.nrrd"), training_labels)
    # # training_labels, _ = nrrd.read(os.path.join(path, f"{forearm_specs['path']}-roi.nrrd"))
    #
    fig = plt.figure(figsize=(10, 5))
    plt.subplot(2, 2, 1)
    im = plt.imshow(reconstruction_array[10])
    plt.contour(training_labels)
    plt.title('Image, mask and ROI boundaries')
    plt.colorbar(im)
    plt.subplot(2, 2, 3)
    plt.title('Unmixing result')
    im = plt.imshow(unmix_so2_proxy(reconstruction_array, wavelengths))
    plt.contour(training_labels)
    plt.colorbar(im)
    vessel = reconstruction_array[:, training_labels == 3]
    print(vessel)
    vessel_norm = np.linalg.norm(vessel, axis=0, ord=2)
    vessel_spectrum = vessel/vessel_norm[np.newaxis, :]
    vessel_spectrum = np.squeeze(vessel_spectrum)
    if len(vessel_spectrum.shape) > 1:
        vessel_spectrum = np.mean(vessel_spectrum, axis=1)

    hb_abs_spectrum = load_iad_results(os.path.join("/home/kris/Data/Dye_project/Measured_Spectra", "B90.npz"))["mua"]
    hb_abs_spectrum = np.interp(wavelengths, np.arange(650, 950), hb_abs_spectrum)

    hbo2_abs_spectrum = load_iad_results(os.path.join("/home/kris/Data/Dye_project/Measured_Spectra", "BIR.npz"))["mua"]
    hbo2_abs_spectrum = np.interp(wavelengths, np.arange(650, 950), hbo2_abs_spectrum)

    target_spectrum = forearm_specs["oxy"] * hbo2_abs_spectrum + (1 - forearm_specs["oxy"]) * hb_abs_spectrum
    slope, intercept, r_value, p_value, std_err = linregress(vessel_spectrum, target_spectrum)
    print(slope, intercept, r_value, p_value, std_err)

    plt.subplot(1, 2, 2)
    plt.plot(wavelengths, target_spectrum, label="Measured material absorption")
    plt.plot(wavelengths, intercept + slope * vessel_spectrum, color="green",
             label=f"Correlation (R={r_value:.2f}, p-value={p_value:.2f})")
    plt.title(f"Target Spectrum (oxy={int(100*forearm_specs['oxy']):d}%) with unmixed oxy: {unmix_so2_proxy(intercept + slope * vessel_spectrum, wavelengths=wavelengths):.2f}")
    plt.xlabel(f"Wavelength [nm]")
    plt.ylabel(f"Absorption Coefficient [$cm^{-1}$]")
    plt.legend(loc="upper left")

    ax = plt.gca()
    ins = ax.inset_axes((0.7, 0.1, 0.2, 0.2))
    ins.plot(wavelengths, vessel_spectrum, label="PA spectrum of ROI")
    ins.set_title("PA spectrum of ROI")
    ins.set_ylabel("PA signal [a.u.]")

    plt.suptitle(f"Patato reconstruction {recon_method}")
    fig.tight_layout()

    plt.show()
    # # plt.savefig(os.path.join(path, f"{forearm_specs['path']}-seg.png"), dpi=300)
    plt.close()