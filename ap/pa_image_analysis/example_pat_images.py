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
import json

recon_method = "das"

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

roi: int = 5
normalize: int = 1
average_phantom: int = 0

examples_images = {
    1: {"oxy": 0.5, "path": os.path.join("Phantom_01", "Scan_25_recon.hdf5")},
    2: {"oxy": 0.3, "path": os.path.join("Phantom_02", "Scan_12_recon.hdf5")},
    3: {"oxy": 0, "path": os.path.join("Phantom_03", "Scan_19_recon.hdf5")},
    4: {"oxy": 0.7, "path": os.path.join("Phantom_04", "Scan_5_recon.hdf5")},
    5: {"oxy": 1, "path": os.path.join("Phantom_05", "Scan_9_recon.hdf5")},
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
    main_phantom_path = os.path.join(base_path, "PAT_Data", forearm_specs["path"])
    if average_phantom:
        path_list = generate_file_paths(main_phantom_path)
    else:
        path_list = [main_phantom_path]
    spectra_to_unmix = list()
    for p_idx, path in enumerate(path_list):
        wavelengths = sp.load_data_field(path, sp.Tags.SETTINGS)[sp.Tags.WAVELENGTHS][1:]
        z_det_pos = 152
        training_labels = np.rot90(nrrd.read(path.replace("recon.hdf5", "pa-labels.nrrd"))[0], 3)
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
            oxy=forearm_specs["oxy"],
            measurements_path=measurements_path)

        json_path = os.path.join(base_path, "Paper_Results", "PAT_Measurement_Correlation",
                                 f"PAT_spectrum_correlation_oxy_{int(100*forearm_specs['oxy']):0d}_p{p_idx}.json")
        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        json.dump({"slope": slope, "intercept": intercept, "r_value": r_value, "p_value": p_value, "std_err": std_err},
                  open(json_path, "w"))

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

        save_path = os.path.join(base_path, "Paper_Results", "PAT_Measurement_Correlation",
                                 f"PAT_spectrum_correlation_oxy_{int(100*forearm_specs['oxy']):0d}_p{p_idx}.png")
        plt.savefig(save_path,
                    bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close()
