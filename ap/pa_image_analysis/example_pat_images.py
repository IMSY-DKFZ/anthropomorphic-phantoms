import numpy as np
import matplotlib.pyplot as plt
import os
import simpa as sp
from ap.utils.correlate_spectrum_to_oxies import correlate_spectrum
from ap.utils.maximum_x_percent_values import top_x_percent_indices
from ap.utils.generate_phantom_file_paths import generate_file_paths
from ap.linear_unimxing import unmix_so2_proxy
import nrrd
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from typing import Union
import json

recon_method = "das"
# base_path = f"/home/kris/Data/Dye_project/PAT_Data/Example_reconstructions_{recon_method}/"
base_path = f"/home/kris/Data/Dye_project/PAT_Data/iThera_2_data/Reconstructions_das/"

measurements_path = "/home/kris/Data/Dye_project/Measured_Spectra"

roi: int = 5
normalize: int = 1
average_phantom: int = 0

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
    1: {"oxy": 0.5, "path": os.path.join("Study_25", "Scan_25_recon1.hdf5")},
    2: {"oxy": 0.3, "path": os.path.join("Study_26", "Scan_12_recon1.hdf5")},
    3: {"oxy": 0, "path": os.path.join("Study_27", "Scan_19_recon1.hdf5")},
    4: {"oxy": 0.7, "path": os.path.join("Study_28", "Scan_5_recon1.hdf5")},
    5: {"oxy": 1, "path": os.path.join("Study_31", "Scan_9_recon1.hdf5")},
    # 6: {"oxy": 1, "path": os.path.join("Study_32", "Scan_21_recon1.hdf5")},
}

oxy_dict = {
    0: 4,
    0.3: 5,
    0.5: 6,
    0.7: 7,
    1: 8
}

for forearm_nr, forearm_specs in examples_images.items():
    print(forearm_nr)
    main_phantom_path = os.path.join(base_path, forearm_specs["path"])
    if average_phantom:
        path_list = generate_file_paths(main_phantom_path)
    else:
        path_list = [main_phantom_path]
    spectra_to_unmix = list()
    for p_idx, path in enumerate(path_list):
        wavelengths = sp.load_data_field(path, sp.Tags.SETTINGS)[sp.Tags.WAVELENGTHS][1:]
        # print(wavelengths)
        # training_labels = np.rot90(sp.load_data_field(path, sp.Tags.DATA_FIELD_SEGMENTATION), 3)
        z_det_pos = 152
        training_labels = np.rot90(nrrd.read(path.replace("Reconstructions_das", "US_analysis").replace("recon1.hdf5", "pa-labels.nrrd"))[0], 3)
        training_labels = np.squeeze(np.fliplr(training_labels)[z_det_pos:z_det_pos + 200, :])
        label = oxy_dict[forearm_specs["oxy"]]
        training_labels[training_labels != label] = 0
        training_labels[training_labels == label] = 1

        reconstruction = sp.load_data_field(path, sp.Tags.DATA_FIELD_RECONSTRUCTED_DATA)
        reconstruction_array = np.stack([np.rot90(reconstruction[str(wl)][:, :, ...], 3) for wl in wavelengths])

        fig = plt.figure(figsize=(8, 7))
        plt.subplot(2, 1, 1)
        im = plt.imshow(reconstruction_array[0], vmin=0, vmax=43.3)
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        scalebar = ScaleBar(0.1, "mm")
        ax.add_artist(scalebar)
        plt.contour(training_labels)
        plt.title('Image and ROI boundaries')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        plt.colorbar(im, cax=cax, orientation="vertical")

        if roi in [100, 0]:
            vessel = reconstruction_array[:, training_labels == 1]
        else:
            indices = top_x_percent_indices(reconstruction_array[10], training_labels, roi)
            # print("Indices of the top 20% maximum values:", indices)

            seg_array = np.zeros_like(reconstruction_array[10])
            maximum_value_pixels = list()
            for idx in indices:
                maximum_value_pixels.append(reconstruction_array[:, idx[0], idx[1]])
                seg_array[idx[0], idx[1]] = 1

            vessel = np.array(maximum_value_pixels)
            vessel = np.moveaxis(vessel, 0, 1)
        print(np.mean(vessel, axis=0).shape)
        print(vessel.shape)
        if normalize:
            # vessel = vessel - np.mean(vessel, axis=0)
            vessel_norm = np.linalg.norm(vessel, axis=0, ord=1)
            vessel_spectrum = vessel / vessel_norm[np.newaxis, :]
        else:
            vessel_spectrum = np.squeeze(vessel)
        if len(vessel_spectrum.shape) > 1:
            vessel_std = np.std(vessel_spectrum, axis=1)
            vessel_spectrum = np.mean(vessel_spectrum, axis=1)

        slope, intercept, r_value, p_value, std_err, target_spectrum, target_std = correlate_spectrum(
            vessel_spectrum,
            wavelengths=wavelengths,
            oxy=forearm_specs["oxy"])

        json.dump({"slope": slope, "intercept": intercept, "r_value": r_value, "p_value": p_value, "std_err": std_err},
                  open(path.replace("hdf5", "json"), "w"))

        plt.subplot(2, 1, 2)

        plt.title(
            f"Target spectrum (oxy={int(100 * forearm_specs['oxy']):d}%) with unmixed oxy: "
            f"{unmix_so2_proxy(vessel_spectrum * slope + intercept, wavelengths=wavelengths):.2f} % and "
            f"{unmix_so2_proxy(vessel_spectrum, wavelengths=wavelengths):.2f} % (uncorrelated)")
        plt.ylabel("Absorption coefficient [$cm^{-1}$]")
        plt.xlabel("Photoacoustic signal [a.u.]")
        plt.scatter(vessel_spectrum, target_spectrum, color="green", label=f"Measured PA signal")
        plt.plot(vessel_spectrum, vessel_spectrum * slope + intercept, color="black",
                 label=f"Correlation (R={r_value:.2f}, p-value={p_value:.2f})")

        ax = plt.gca()
        ins = ax.inset_axes((0.7, 0.2, 0.2, 0.2))
        ins.plot(wavelengths, vessel_spectrum * slope + intercept, color="green")
        ins.plot(wavelengths, target_spectrum, label="Measured material absorption", color="blue")
        ins.fill_between(wavelengths, target_spectrum - target_std,
                         target_spectrum + target_std,
                         alpha=0.2, color="blue")
        ins.set_title("PAT approx of $\mu_a$")
        ins.set_ylabel("$\mu_a$ [$cm^{-1}$]")
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
        plt.savefig(os.path.join("/home/kris/Pictures/Phantom_Paper_Figures/", f"PAT_spectrum_correlation_oxy_{int(100*forearm_specs['oxy']):0d}_p{p_idx}.png"), dpi=300)
        plt.close()
