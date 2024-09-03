import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os
import simpa as sp
from tmd.utils.io_iad_results import load_iad_results
from tmd.linear_unimxing import unmix_so2_proxy
import nrrd
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D

recon_method = "das"
# base_path = f"/home/kris/Data/Dye_project/PAT_Data/Example_reconstructions_{recon_method}/"
base_path = f"/home/kris/Data/Dye_project/PAT_Data/iThera_2_data/Reconstructions_das/"

measurements_path = "/home/kris/Data/Dye_project/Measured_Spectra"

abs_dict = {
    0: "B90",
    0.3: "B93",
    0.5: "B95",
    0.7: "B97",
    1: "BIR",
}

# examples_images = {
#     # 1: {"oxy": 0.5, "path": os.path.join("Study_6", "Scan_22.hdf5")},
#     # 2: {"oxy": 0.3, "path": os.path.join("Study_7", "Scan_11.hdf5")},
#     3: {"oxy": 0, "path": os.path.join("Study_8", "Scan_17.hdf5")},
#     # 4: {"oxy": 0.7, "path": os.path.join("Study_9", "Scan_4.hdf5")},
#     # 5: {"oxy": 1, "path": os.path.join("Study_10", "Scan_8.hdf5")},
#     # 6: {"oxy": 1, "path": os.path.join("Study_11", "Scan_19.hdf5")},
# }

# examples_images = {
#     # 1: {"oxy": 0.5, "path": os.path.join("Study_17", "Scan_1.hdf5")},
#     3: {"oxy": 0, "path": os.path.join("Study_18", "Scan_1.hdf5")},
#     # 5: {"oxy": 1, "path": os.path.join("Study_19", "Scan_1.hdf5")},
# }

# examples_images = {
#     # 1: {"oxy": 0.5, "path": os.path.join("Study_17", "Forearm_1.hdf5")},
#     3: {"oxy": 0, "path": os.path.join("Study_18", "Forearm_3.hdf5")},
#     # 6: {"oxy": 1, "path": os.path.join("Study_19", "Forearm_6.hdf5")},
# }

# ithera images
# examples_images = {
#     1: {"oxy": 0.5, "path": os.path.join("Study_61", "Scan_24_recon.hdf5")},
#     2: {"oxy": 0.3, "path": os.path.join("Study_62", "Scan_11_recon.hdf5")},
#     3: {"oxy": 0, "path": os.path.join("Study_63", "Scan_17_recon.hdf5")},
#     4: {"oxy": 0.7, "path": os.path.join("Study_64", "Scan_5_recon.hdf5")},
#     # 5: {"oxy": 1, "path": os.path.join("Study_19", "Scan_1_recon.hdf5")},
#     6: {"oxy": 1, "path": os.path.join("Study_66", "Scan_20_recon.hdf5")},
# }

# ithera2 images
examples_images = {
    1: {"oxy": 0.5, "path": os.path.join("Study_25", "Scan_25_recon.hdf5")},
    2: {"oxy": 0.3, "path": os.path.join("Study_26", "Scan_12_recon.hdf5")},
    3: {"oxy": 0, "path": os.path.join("Study_27", "Scan_18_recon.hdf5")},
    4: {"oxy": 0.7, "path": os.path.join("Study_28", "Scan_6_recon.hdf5")},
    5: {"oxy": 1, "path": os.path.join("Study_31", "Scan_9_recon.hdf5")},
    # 6: {"oxy": 1, "path": os.path.join("Study_32", "Scan_21_recon.hdf5")},
}

for forearm_nr, forearm_specs in examples_images.items():
    path = os.path.join(base_path, forearm_specs["path"])
    wavelengths = sp.load_data_field(path, sp.Tags.SETTINGS)[sp.Tags.WAVELENGTHS][:]
    # print(wavelengths)
    training_labels = np.rot90(sp.load_data_field(path, sp.Tags.DATA_FIELD_SEGMENTATION), 3)
    reconstruction = sp.load_data_field(path, sp.Tags.DATA_FIELD_RECONSTRUCTED_DATA)
    reconstruction_array = np.stack([np.rot90(reconstruction[str(wl)][:, :, ...], 3) for wl in wavelengths])
    # print(reconstruction_array.shape)

    # absorption = sp.load_data_field(path, sp.Tags.DATA_FIELD_ABSORPTION_PER_CM)
    # absorption_array = np.stack([np.rot90(absorption[str(wl)], 3) for wl in wavelengths])

    # plt.subplot(3, 1, 1)
    # plt.imshow(reconstruction_array[4])
    # plt.subplot(3, 1, 2)
    # plt.imshow(reconstruction_array[5])
    # plt.subplot(3, 1, 3)
    # plt.imshow(reconstruction_array[5] - reconstruction_array[4])
    # plt.show()
    # training_labels, _ = nrrd.read(path.replace(".hdf5", "_image-labels.nrrd"))
    # training_labels = training_labels[0]

    # training_labels = np.zeros(reconstruction_array.shape[1:], dtype=np.uint8)
    # print(training_labels.shape)
    # training_labels[42:45, 195:198] = 3     # for example image 5
    # training_labels[61:63, 255:257] = 3     # for example image 5
    # training_labels[53:55, 204:206] = 3     # for example image 3
    # training_labels[41:45, 194:197] = 3     # for sim image 1
    # training_labels[39:41, 201:203] = 3     # for sim image 3
    # training_labels[42:45, 244:247] = 3     # for sim image 5

    # training_labels[65:68, 190:193] = 3  # for ithera image 3: 17
    # training_labels[65:69, 145:149] = 3  # for ithera image 3: 16
    # training_labels[65:75, 183:207] = 3  # for ithera image 3: 18   sos 1500
    # training_labels[62:65, 189:200] = 3  # for ithera image 3: 18   sos 1500
    # training_labels[63:77, 139:161] = 3  # for ithera image 1: 24   sos 1500
    # training_labels[59:63, 144:156] = 3  # for ithera image 1: 24   sos 1500
    # training_labels[63:69, 266:271] = 3  # for ithera image 4
    # nrrd.write(os.path.join(path, f"{forearm_specs['path']}-roi.nrrd"), training_labels)
    # # training_labels, _ = nrrd.read(os.path.join(path, f"{forearm_specs['path']}-roi.nrrd"))

    # nrrd.write(path.split('.')[0] + "_image.nrrd", reconstruction_array)
    #
    fig = plt.figure(figsize=(8, 7))
    plt.subplot(2, 1, 1)
    im = plt.imshow(reconstruction_array[0], vmin=0, vmax=43.3)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    scalebar = ScaleBar(0.1, "mm")
    ax.add_artist(scalebar)
    plt.contour(training_labels)
    plt.title('Image, mask and ROI boundaries')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    plt.colorbar(im, cax=cax, orientation="vertical")
    # plt.tight_layout()
    # plt.show()
    # exit()


    vessel = reconstruction_array[:, training_labels == 1]
    # abs_vessel = absorption_array[:, training_labels == 3]
    # abs_vessel = np.squeeze(abs_vessel)
    # abs_vessel = np.mean(abs_vessel, axis=1)
    print(np.mean(vessel, axis=0).shape)
    print(vessel.shape)
    vessel = vessel - np.mean(vessel, axis=0)
    vessel_norm = np.linalg.norm(vessel, axis=0, ord=1)
    vessel_spectrum = vessel / vessel_norm[np.newaxis, :]
    # vessel_spectrum = np.squeeze(vessel)
    if len(vessel_spectrum.shape) > 1:
        vessel_std = np.std(vessel_spectrum, axis=1)
        vessel_spectrum = np.mean(vessel_spectrum, axis=1)
    hb_abs_spectrum = load_iad_results(os.path.join(measurements_path, "B90.npz"))["mua"]
    hb_abs_spectrum = np.interp(wavelengths, np.arange(650, 950), hb_abs_spectrum)

    hbo2_abs_spectrum = load_iad_results(os.path.join(measurements_path, "BIR.npz"))["mua"]
    hbo2_abs_spectrum = np.interp(wavelengths, np.arange(650, 950), hbo2_abs_spectrum)

    target_spectrum = load_iad_results(os.path.join(measurements_path, abs_dict[forearm_specs["oxy"]] + ".npz"))["mua"]
    target_spectrum = np.interp(wavelengths, np.arange(650, 950), target_spectrum)

    target_std = load_iad_results(os.path.join(measurements_path, abs_dict[forearm_specs["oxy"]] + ".npz"))["mua_std"]
    target_std = np.interp(wavelengths, np.arange(650, 950), target_std)

    slope, intercept, r_value, p_value, std_err = linregress(target_spectrum, vessel_spectrum)
    print(slope, intercept, r_value, p_value, std_err)

    # plt.subplot(2, 2, 3)
    # plt.title('Unmixing result')
    # im = plt.imshow(unmix_so2_proxy(intercept + slope * reconstruction_array, wavelengths))
    # # plt.imshow(absorption_array[10], alpha=0.2)
    # plt.contour(training_labels)
    # plt.colorbar(im)

    # print("vessel spectrum", intercept + slope * vessel_spectrum)

    plt.subplot(2, 1, 2)

    plt.title(
        f"Target Spectrum (oxy={int(100 * forearm_specs['oxy']):d}%) with unmixed oxy: "
        f"{unmix_so2_proxy((vessel_spectrum - intercept)/slope, wavelengths=wavelengths):.2f} % and "
        f"{unmix_so2_proxy(vessel_spectrum, wavelengths=wavelengths):.2f} % (uncorrelated)")
    plt.xlabel(f"Absorption Coefficient [$cm^{-1}$]")
    plt.ylabel(f"Photoacoustic Signal [a.u.]")
    plt.errorbar(target_spectrum, (vessel_spectrum - intercept)/slope, xerr=target_std, yerr=vessel_std, color="green", barsabove=True, ls="none", fmt="o",
                 label=f"Measured PA signal")
    plt.plot(target_spectrum, target_spectrum, color="black",
             label=f"Correlation (R={r_value:.2f}, p-value={p_value:.2f})")

    ax = plt.gca()
    ins = ax.inset_axes((0.7, 0.2, 0.2, 0.2))
    ins.plot(wavelengths, (vessel_spectrum - intercept)/slope, color="green")
    ins.plot(wavelengths, target_spectrum, label="Measured material absorption", color="blue")
    ins.fill_between(wavelengths, target_spectrum - target_std,
                     target_spectrum + target_std,
                     alpha=0.2, color="blue")
    ins.set_title("PA spectrum of ROI")
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
    # plt.savefig(os.path.join("/home/kris/Pictures/Phantom_Paper_Figures/", f"PAT_spectrum_correlation_oxy_{int(100*forearm_specs['oxy']):0d}.png"), dpi=300)
    plt.close()
