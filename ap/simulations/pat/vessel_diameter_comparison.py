import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage import measure
import nrrd

plt.rcParams.update({'font.size': 12,
                     "font.family": "serif"})


def get_centroid_loc_and_orientation(con_props):
    prop = con_props[0]
    y0, x0 = prop.centroid
    orientation = prop.orientation
    x1 = x0 + np.cos(orientation) * 0.5 * prop.axis_minor_length
    y1 = y0 - np.sin(orientation) * 0.5 * prop.axis_minor_length
    x2 = x0 - np.sin(orientation) * 0.5 * prop.axis_major_length
    y2 = y0 - np.cos(orientation) * 0.5 * prop.axis_major_length

    # mean_d = (prop.axis_major_length + prop.axis_minor_length) / 2
    mean_d = np.sqrt(prop.area / np.pi) * 2
    return x0, y0, x1, y1, x2, y2, mean_d


def check_for_vessel_diameter_changes(simulation_path: str, forearm_nr: str,
                                      comparison_dict: dict, save_fig: bool = False,
                                      labels_path: str = "",
                                      results_path: str = ""):

    if isinstance(comparison_dict["short"], str):
        comparison_dict["short"] = [comparison_dict["short"]]
    else:
        assert isinstance(comparison_dict["short"], list), "comparison_dict['short'] must be string or list of strings"

    orig_recon, _ = nrrd.read(os.path.join(os.path.dirname(labels_path), f"{forearm_nr}.nrrd"))
    orig_recon = np.rot90(np.squeeze(orig_recon), 3)

    segmentation, _ = nrrd.read(os.path.join(os.path.dirname(labels_path), f"{forearm_nr}-labels.nrrd"))
    segmentation = np.rot90(np.squeeze(segmentation), 3).astype(int)

    connected_labels = measure.label(segmentation, background=0)
    con_props = measure.regionprops(connected_labels)

    x0, y0, x1, y1, x2, y2, mean_d = get_centroid_loc_and_orientation(con_props)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    img = plt.imshow(orig_recon)
    plt.imshow(segmentation, alpha=0.3)
    plt.plot(x0, y0, 'r.')
    plt.plot((x0, x1), (y0, y1), '-r', linewidth=2.5, label=f"Diameter (D): {mean_d*0.1:.2f} mm")
    plt.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
    plt.legend(loc="lower left")

    plt.title(f"Simulation without {comparison_dict['description']} ($sos_{{original}}$ = 1470" + r" $\frac{m}{s}$)")
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    scalebar = ScaleBar(0.1, "mm")
    ax.add_artist(scalebar)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    plt.colorbar(img, cax=cax, orientation="vertical")
    # plt.close()
    # plt.show()
    # exit()


    for sos_idx, sos_adjustment in enumerate(comparison_dict["short"]):
        sos_recon, _ = nrrd.read(os.path.join(os.path.dirname(labels_path), f"{forearm_nr}_{sos_adjustment}.nrrd"))
        sos_recon = np.rot90(np.squeeze(sos_recon), 3)

        sos_segmentation, _ = nrrd.read(os.path.join(os.path.dirname(labels_path), f"{forearm_nr}_{sos_adjustment}-labels.nrrd"))
        sos_segmentation = np.rot90(np.squeeze(sos_segmentation), 3).astype(int)

        connected_labels = measure.label(sos_segmentation, background=0)
        con_props = measure.regionprops(connected_labels)
        sos_segmentation = sos_segmentation[con_props[0].slice]

        connected_labels_crop = measure.label(sos_segmentation, background=0)
        con_props_crop = measure.regionprops(connected_labels_crop)

        x0, y0, x1, y1, x2, y2, mean_d = get_centroid_loc_and_orientation(con_props_crop)
        sos_recon = sos_recon[con_props[0].slice]

        plt.subplot(2, len(comparison_dict["short"]), sos_idx + 5)
        img = plt.imshow(sos_recon)
        plt.imshow(sos_segmentation, alpha=0.3)
        plt.plot(x0, y0, 'r.')
        plt.plot((x0, x1), (y0, y1), '-r', linewidth=2.5, label=f"D = {mean_d * 0.1:.2f} mm")
        plt.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
        plt.legend(loc="lower left")
        title_string = sos_adjustment[3:]
        title_string = " + " + title_string if title_string[0] != "-" else title_string.replace("-", " - ")
        plt.title(f"$sos_{{original}}${title_string}" + r" $\frac{m}{s}$")
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        scalebar = ScaleBar(0.1, "mm")
        ax.add_artist(scalebar)
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="2%", pad=0.05)
        # plt.colorbar(img, cax=cax, orientation="vertical")

    if save_fig:
        plt.savefig(os.path.join(results_path, f"{forearm_nr}_vessel_comparison.png"),
                    bbox_inches="tight", pad_inches=0)
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    from tabulate import tabulate
    diameters = [3, 4, 5, 6]
    sos = [1470 + adj for adj in range(-100, 101, 50)]

    diameter_shifts = [[s] + [d / s * 1497.4 for d in diameters] for s in sos]

    print(tabulate(diameter_shifts, headers=[str(d) for d in diameters], tablefmt="fancy_grid"))

