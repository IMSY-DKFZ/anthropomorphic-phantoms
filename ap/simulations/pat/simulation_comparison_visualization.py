import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage import measure

import simpa as sp
from simpa import Tags

from ap.utils.dye_tissue_properties import seg_dict
from ap.dye_analysis import DyeColors
from ap.utils.maximum_x_percent_values import top_x_percent_indices
plt.rcParams.update({'font.size': 12,
                     "font.family": "serif"})


def visualize_comparison(simulation_path: str, forearm_nr: str, wavelengths: np.ndarray,
                         comparison_dict: dict, save_fig: bool = False, results_path: str = ""):

    if isinstance(comparison_dict["short"], str):
        comparison_dict["short"] = [comparison_dict["short"]]
    else:
        assert isinstance(comparison_dict["short"], list), "comparison_dict['short'] must be string or list of strings"
    for short_description in comparison_dict["short"]:
        compare_recon = sp.load_data_field(os.path.join(os.path.dirname(simulation_path),
                                                        f"{forearm_nr + '_' + short_description}.hdf5"),
                                           Tags.DATA_FIELD_RECONSTRUCTED_DATA)
        compare_recon_array = np.stack([np.rot90(compare_recon[str(wl)][:, :, ...], 3) for wl in wavelengths])[:, :-9, :]

        orig_recon = sp.load_data_field(os.path.join(os.path.dirname(simulation_path), f"{forearm_nr}.hdf5"),
                                        Tags.DATA_FIELD_RECONSTRUCTED_DATA)
        orig_recon_array = np.stack([np.rot90(orig_recon[str(wl)][:, :, ...], 3) for wl in wavelengths])[:, :-9, :]

        segmentation = np.rot90(sp.load_data_field(os.path.join(os.path.dirname(simulation_path), f"{forearm_nr}.hdf5"),
                                                   Tags.DATA_FIELD_SEGMENTATION), 3)[9:, :].astype(int)

        segmentation[segmentation < 4] = 0
        segmentation[segmentation >= 9] = 0
        connected_labels = measure.label(segmentation, background=0)
        con_props = measure.regionprops(connected_labels)
        seg_props = measure.regionprops(segmentation)

        seg_props_bboxes = [prop.bbox for prop in seg_props]

        if len(con_props) != len(seg_props):
            double_instance_components = list()
            for con_prop in con_props:
                if con_prop.bbox not in seg_props_bboxes:
                    double_instance_components.append(con_prop)
                else:
                    continue

            if double_instance_components[0].centroid[0] > double_instance_components[1].centroid[0]:
                con_props.remove(double_instance_components[0])
                segmentation[connected_labels == double_instance_components[0].label] = 0
            else:
                con_props.remove(double_instance_components[1])
                segmentation[connected_labels == double_instance_components[1].label] = 0

        # fig, ax = plt.subplots(1, 2)
        # plt.axis('off')
        # ax[0].imshow(segmentation)
        # centroids = np.zeros(shape=(len(np.unique(connected_labels)), 2))  # Access the coordinates of centroids
        # for i, con_prop in enumerate(con_props):
        #     # if len(np.unique(segmentation[prop.slice])) > 1:
        #
        #     my_centroid = con_prop.centroid
        #     centroids[i, :] = my_centroid
        #     ax[0].plot(my_centroid[1], my_centroid[0], 'r.')
        #
        # ax[1].imshow(segmentation)
        #
        # # print(centroids)
        # # fig.savefig('out.png', bbox_inches='tight', pad_inches=0)
        # plt.show()
        # exit()


        fig = plt.figure(figsize=(9, 7))
        ax1 = plt.subplot(2, 2, 1)
        img = plt.imshow(np.fliplr(orig_recon_array[0]))
        plt.title(f"Simulation without {comparison_dict['description']}")
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        scalebar = ScaleBar(0.1, "mm")
        ax.add_artist(scalebar)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        plt.colorbar(img, cax=cax, orientation="vertical")
        plt.xlabel("[a.u.]")

        ax2 = plt.subplot(2, 2, 2)
        plt.imshow(np.fliplr(compare_recon_array[0]))
        plt.title(f"Simulation with {comparison_dict['description']}")
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        scalebar = ScaleBar(0.1, "mm")
        ax.add_artist(scalebar)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        plt.colorbar(img, cax=cax, orientation="vertical")
        plt.xlabel("[a.u.]")


        plt.subplot(2, 1, 2)
        plt.ylabel("Normalized spectrum [a.u.]")
        plt.xlabel("Wavelength [nm]")
        for vessel_label in range(4, 9):
            for simulation, sim_name in zip([orig_recon_array, compare_recon_array],
                                            [f"without {comparison_dict['description']}",
                                             f"with {comparison_dict['description']}"]):
                vessel_label_mask = np.zeros_like(segmentation)
                vessel_label_mask[segmentation != vessel_label] = 0
                vessel_label_mask[segmentation == vessel_label] = 1
                if (vessel_label_mask == 0).all():
                    break

                if vessel_label == 4:
                    plot_color = "b"
                elif vessel_label == 8:
                    plot_color = "r"
                else:
                    dye_color_string = "B9" + str(int(10 * seg_dict[vessel_label]))
                    plot_color = DyeColors[dye_color_string]
                for ax_idx, axis in enumerate([ax1, ax2]):
                    CS = axis.contour(np.fliplr(vessel_label_mask), colors=plot_color, alpha=0.5, linewidths=0.5,
                                      linestyles="--" if ax_idx == 1 else "-")
                    if ax_idx == 1:
                        for coll in CS.collections:
                            coll.set_linestyle((0, (8, 5)))

                indices = top_x_percent_indices(simulation[0], vessel_label_mask, 5)

                maximum_value_pixels = list()
                for idx in indices:
                    maximum_value_pixels.append(simulation[:, idx[0], idx[1]])
                vessel = np.array(maximum_value_pixels)
                vessel = np.moveaxis(vessel, 0, 1)

                vessel_norm = np.linalg.norm(vessel, axis=0, ord=1)
                vessel_spectrum = vessel / vessel_norm[np.newaxis, :]

                vessel_std = np.std(vessel_spectrum, axis=1)
                vessel_spectrum = np.mean(vessel_spectrum, axis=1)

                plt.plot(wavelengths, vessel_spectrum, label=f"Vessel oxy {100 * seg_dict[vessel_label]}% {sim_name}",
                         linestyle="-" if "out" in sim_name else "--",
                         color=plot_color)
                # plt.fill_between(wavelengths, vessel_spectrum - vessel_std,
                #                  vessel_spectrum + vessel_std,
                #                  alpha=0.2)
        plt.legend()
        plt.tight_layout()
        if save_fig:
            plt.savefig(os.path.join(results_path, f"{forearm_nr}_{short_description}.pdf"),
                        bbox_inches='tight', pad_inches=0, dpi=400)
        else:
            plt.show()

        plt.close()
