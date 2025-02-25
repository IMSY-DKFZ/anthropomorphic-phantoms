import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import json
from scipy import stats
from skimage import measure
plt.rcParams.update({'font.size': 12,
                     "font.family": "serif"})
plt.switch_backend("TkAgg")


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


def calculate_mean_std_ci(arr, mask=None, aggregate_over_structures=True):
    if mask is not None:
        if isinstance(mask, tuple):
            int_mask = mask[0].astype(int)
            int_mask[mask[1] == 1] = 2
            mask = int_mask

        if aggregate_over_structures:
            connected_components = measure.label(mask, background=0)
            small_components = np.where(np.bincount(connected_components.ravel()) < 10)[0]
            for small_component in small_components:
                connected_components[connected_components == small_component] = 0

            nr_conn = np.max(connected_components)

            means = [np.nanmean(arr[connected_components == i]) for i in range(1, nr_conn + 1)]
            stds = [np.nanstd(arr[connected_components == i]) for i in range(1, nr_conn + 1)]
            cis = [stats.bootstrap((arr[connected_components == i],),
                                   statistic=np.nanmean,
                                   confidence_level=0.95,
                                   method="percentile").confidence_interval for i in range(1, nr_conn + 1)]

            mean, std, conf_int = np.mean(means), np.mean(stds), [np.mean([ci.low for ci in cis]),
                                                                  np.mean([ci.high for ci in cis])]

        else:
            arr = arr[mask]

            mean, std = np.nanmean(arr), np.nanstd(arr)
            ci = stats.bootstrap((arr,), statistic=np.nanmean, confidence_level=0.95, method="percentile")
            conf_int = [ci.confidence_interval.low, ci.confidence_interval.high]
    else:
        mean, std = np.nanmean(arr), np.nanstd(arr)
        ci = stats.bootstrap((arr,), statistic=np.nanmean, confidence_level=0.95, method="percentile")
        conf_int = [ci.confidence_interval.low, ci.confidence_interval.high]

    return mean, std, conf_int


def mask_arrays(arr, mask):
    if mask is not None:
        arr[mask == 0] = np.nan
        return arr
    return arr


def calculate_error_along_depth(data, mask, aggregate_over_structures=True):
    data_shape = data.shape
    depth_axis = np.argmax(data_shape)

    if mask is not None:
        if isinstance(mask, tuple):
            int_mask = mask[0].astype(int)
            int_mask[mask[1] == 1] = 2
            mask = int_mask

        if aggregate_over_structures:
            connected_components = measure.label(mask, background=0)
            small_components = np.where(np.bincount(connected_components.ravel()) < 10)[0]
            for small_component in small_components:
                connected_components[connected_components == small_component] = 0

            nr_conn = np.max(connected_components)

            depth_error_mean = np.zeros((nr_conn, data_shape[np.abs(depth_axis - 1)]))
            depth_error_std = np.zeros((nr_conn, data_shape[np.abs(depth_axis - 1)]))

            for comp in range(1, nr_conn + 1):
                masking_array = np.zeros_like(mask)
                masking_array[connected_components == comp] = 1
                depth_error_mean[comp-1] = np.nanmean(mask_arrays(data.copy(), masking_array), axis=depth_axis)
                depth_error_std[comp-1] = np.nanstd(mask_arrays(data.copy(), masking_array), axis=depth_axis)

            depth_error_mean = np.nanmean(depth_error_mean, axis=0)
            depth_error_std = np.nanmean(depth_error_std, axis=0)
        else:
            depth_error_mean = np.nanmean(mask_arrays(data, mask), axis=depth_axis)
            depth_error_std = np.nanstd(mask_arrays(data, mask), axis=depth_axis)
    else:
        depth_error_mean = np.nanmean(data, axis=depth_axis)
        depth_error_std = np.nanstd(data, axis=depth_axis)
    return depth_error_mean, depth_error_std


if __name__ == "__main__":
    for image_modality in ["pat", "hsi"]:
        results_folder = os.path.join(base_path, "Paper_Results", "Oxy_Results")
        forearm_list = glob.glob(os.path.join(results_folder,
                                              f"{image_modality}_forearm_*.npz"))

        all_tissue_errors = []
        all_vessel_errors = []
        all_cal_tissue_errors = []
        all_cal_vessel_errors = []
        all_tissue_fluence_errors = []
        all_vessel_fluence_errors = []

        all_depth_tissue_errors = []
        all_depth_vessel_errors = []
        all_depth_cal_tissue_errors = []
        all_depth_cal_vessel_errors = []
        all_depth_tissue_fluence_errors = []
        all_depth_vessel_fluence_errors = []

        all_depth_tissue_errors_std = []
        all_depth_vessel_errors_std = []
        all_depth_cal_tissue_errors_std = []
        all_depth_cal_vessel_errors_std = []
        all_depth_tissue_fluence_errors_std = []
        all_depth_vessel_fluence_errors_std = []

        for forearm in forearm_list:
            name = f"Forearm {forearm.split('_')[-1].split('.')[0]}"
            print(name)
            data = np.load(forearm)

            tissue_mask = data["tissue"]
            vessel_mask = data["vessels"]

            tissue_error_mean, tissue_error_std, tissue_error_ci = calculate_mean_std_ci(
                data["oxy_error"],
                (tissue_mask, vessel_mask))
            vessel_error_mean, vessel_error_std, vessel_error_ci = calculate_mean_std_ci(
                data["oxy_error"], vessel_mask)
            cal_tissue_error_mean, cal_tissue_error_std, cal_tissue_error_ci = calculate_mean_std_ci(
                data["cal_oxy_error"], (tissue_mask, vessel_mask))
            cal_vessel_error_mean, cal_vessel_error_std, cal_vessel_error_ci = calculate_mean_std_ci(
                data["cal_oxy_error"], vessel_mask)

            all_tissue_errors.append((tissue_error_mean, tissue_error_std, tissue_error_ci))
            all_vessel_errors.append((vessel_error_mean, vessel_error_std, vessel_error_ci))
            all_cal_tissue_errors.append((cal_tissue_error_mean, cal_tissue_error_std, cal_tissue_error_ci))
            all_cal_vessel_errors.append((cal_vessel_error_mean, cal_vessel_error_std, cal_vessel_error_ci))

            if image_modality == "pat":
                tissue_fluence_error_mean, tissue_fluence_error_std, tissue_fluence_error_ci = calculate_mean_std_ci(
                    data["fluence_corr_error"],
                    (tissue_mask, vessel_mask))
                vessel_fluence_error_mean, vessel_fluence_error_std, vessel_fluence_error_ci = calculate_mean_std_ci(
                    data["fluence_corr_error"],
                    vessel_mask)

                all_tissue_fluence_errors.append(
                    (tissue_fluence_error_mean, tissue_fluence_error_std, tissue_fluence_error_ci)
                )
                all_vessel_fluence_errors.append(
                    (vessel_fluence_error_mean, vessel_fluence_error_std, vessel_fluence_error_ci)
                )

                depth_tissue_error, depth_tissue_error_std = calculate_error_along_depth(
                    data["oxy_error"],
                    (tissue_mask, vessel_mask))

                depth_vessel_error, depth_vessel_error_std = calculate_error_along_depth(
                    data["oxy_error"],
                    vessel_mask)

                depth_cal_tissue_error, depth_cal_tissue_error_std = calculate_error_along_depth(
                    data["cal_oxy_error"],
                    (tissue_mask, vessel_mask))

                depth_cal_vessel_error, depth_cal_vessel_error_std = calculate_error_along_depth(
                    data["cal_oxy_error"],
                    vessel_mask)

                depth_tissue_fluence_error, depth_tissue_fluence_error_std = calculate_error_along_depth(
                    data["fluence_corr_error"],
                    (tissue_mask, vessel_mask))

                depth_vessel_fluence_error, depth_vessel_fluence_error_std = calculate_error_along_depth(
                    data["fluence_corr_error"],
                    vessel_mask)

                all_depth_tissue_errors.append(depth_tissue_error)
                all_depth_vessel_errors.append(depth_vessel_error)
                all_depth_cal_tissue_errors.append(depth_cal_tissue_error)
                all_depth_cal_vessel_errors.append(depth_cal_vessel_error)
                all_depth_tissue_fluence_errors.append(depth_tissue_fluence_error)
                all_depth_vessel_fluence_errors.append(depth_vessel_fluence_error)

                all_depth_tissue_errors_std.append(depth_tissue_error_std)
                all_depth_vessel_errors_std.append(depth_vessel_error_std)
                all_depth_cal_tissue_errors_std.append(depth_cal_tissue_error_std)
                all_depth_cal_vessel_errors_std.append(depth_cal_vessel_error_std)
                all_depth_tissue_fluence_errors_std.append(depth_tissue_fluence_error_std)
                all_depth_vessel_fluence_errors_std.append(depth_vessel_fluence_error_std)

        # Aggregate means and standard deviations
        tissue_error_means = [x[0] for x in all_tissue_errors]
        tissue_error_stds = [x[1] for x in all_tissue_errors]
        tissue_error_cis = [x[2] for x in all_tissue_errors]
        vessel_error_means = [x[0] for x in all_vessel_errors]
        vessel_error_stds = [x[1] for x in all_vessel_errors]
        vessel_error_cis = [x[2] for x in all_vessel_errors]

        cal_tissue_error_means = [x[0] for x in all_cal_tissue_errors]
        cal_tissue_error_stds = [x[1] for x in all_cal_tissue_errors]
        cal_tissue_error_cis = [x[2] for x in all_cal_tissue_errors]
        cal_vessel_error_means = [x[0] for x in all_cal_vessel_errors]
        cal_vessel_error_stds = [x[1] for x in all_cal_vessel_errors]
        cal_vessel_error_cis = [x[2] for x in all_cal_vessel_errors]

        # Calculate overall mean
        overall_tissue_error_mean = np.mean(tissue_error_means)
        overall_vessel_error_mean = np.mean(vessel_error_means)
        overall_cal_tissue_error_mean = np.mean(cal_tissue_error_means)
        overall_cal_vessel_error_mean = np.mean(cal_vessel_error_means)

        # Calculate overall std regularly
        overall_tissue_error_std = np.mean(tissue_error_stds)
        overall_vessel_error_std = np.mean(vessel_error_stds)
        overall_cal_tissue_error_std = np.mean(cal_tissue_error_stds)
        overall_cal_vessel_error_std = np.mean(cal_vessel_error_stds)

        # Calculate overall cis regularly
        overall_tissue_error_ci = (np.mean([ci[0] for ci in tissue_error_cis]),
                                   np.mean([ci[1] for ci in tissue_error_cis]))
        overall_vessel_error_ci = (np.mean([ci[0] for ci in vessel_error_cis]),
                                   np.mean([ci[1] for ci in vessel_error_cis]))
        overall_cal_tissue_error_ci = (np.mean([ci[0] for ci in cal_tissue_error_cis]),
                                       np.mean([ci[1] for ci in cal_tissue_error_cis]))
        overall_cal_vessel_error_ci = (np.mean([ci[0] for ci in cal_vessel_error_cis]),
                                       np.mean([ci[1] for ci in cal_vessel_error_cis]))

        results = {
            "Entire Phantom LSU Error Mean": overall_tissue_error_mean,
            "Entire Phantom LSU Error Std": overall_tissue_error_std,
            "Entire Phantom LSU Error CI": overall_tissue_error_ci,
            "Vessels-only LSU Error Mean": overall_vessel_error_mean,
            "Vessels-only LSU Error Std": overall_vessel_error_std,
            "Vessels-only LSU Error CI": overall_vessel_error_ci,
            "Entire Phantom Cal. LSU Error Mean": overall_cal_tissue_error_mean,
            "Entire Phantom Cal. LSU Error Std": overall_cal_tissue_error_std,
            "Entire Phantom Cal. LSU Error CI": overall_cal_tissue_error_ci,
            "Vessels-only Cal. LSU Error Mean": overall_cal_vessel_error_mean,
            "Vessels-only Cal. LSU Error Std": overall_cal_vessel_error_std,
            "Vessels-only Cal. LSU Error CI": overall_cal_vessel_error_ci
        }

        if image_modality == "hsi":
            print(results)

        else:
            tissue_fluence_error_means = [x[0] for x in all_tissue_fluence_errors]
            tissue_fluence_error_stds = [x[1] for x in all_tissue_fluence_errors]
            tissue_fluence_error_cis = [x[2] for x in all_tissue_fluence_errors]
            vessel_fluence_error_means = [x[0] for x in all_vessel_fluence_errors]
            vessel_fluence_error_stds = [x[1] for x in all_vessel_fluence_errors]
            vessel_fluence_error_cis = [x[2] for x in all_vessel_fluence_errors]

            overall_tissue_fluence_error_mean = np.mean(tissue_fluence_error_means)
            overall_vessel_fluence_error_mean = np.mean(vessel_fluence_error_means)

            overall_tissue_fluence_error_std = np.mean(tissue_fluence_error_stds)
            overall_vessel_fluence_error_std = np.mean(vessel_fluence_error_stds)

            overall_tissue_fluence_error_ci = (np.mean([ci[0] for ci in tissue_fluence_error_cis]),
                                               np.mean([ci[1] for ci in tissue_fluence_error_cis]))
            overall_vessel_fluence_error_ci = (np.mean([ci[0] for ci in vessel_fluence_error_cis]),
                                               np.mean([ci[1] for ci in vessel_fluence_error_cis]))

            results["Entire Phantom Fluence Comp. Error Mean"] = overall_tissue_fluence_error_mean
            results["Entire Phantom Fluence Comp. Error Std"] = overall_tissue_fluence_error_std
            results["Entire Phantom Fluence Comp. Error CI"] = overall_tissue_fluence_error_ci
            results["Vessels-only Fluence Comp. Error Mean"] = overall_vessel_fluence_error_mean
            results["Vessels-only Fluence Comp. Error Std"] = overall_vessel_fluence_error_std
            results["Vessels-only Fluence Comp. Error CI"] = overall_vessel_fluence_error_ci

            print(results)

            # Calculate hierarchical std

            # Convert lists to arrays for easier manipulation
            all_depth_tissue_errors = np.array(all_depth_tissue_errors)
            all_depth_vessel_errors = np.array(all_depth_vessel_errors)
            all_depth_cal_tissue_errors = np.array(all_depth_cal_tissue_errors)
            all_depth_cal_vessel_errors = np.array(all_depth_cal_vessel_errors)
            all_depth_tissue_fluence_errors = np.array(all_depth_tissue_fluence_errors)
            all_depth_vessel_fluence_errors = np.array(all_depth_vessel_fluence_errors)

            all_depth_tissue_errors_std = np.array(all_depth_tissue_errors_std)
            all_depth_vessel_errors_std = np.array(all_depth_vessel_errors_std)
            all_depth_cal_tissue_errors_std = np.array(all_depth_cal_tissue_errors_std)
            all_depth_cal_vessel_errors_std = np.array(all_depth_cal_vessel_errors_std)
            all_depth_tissue_fluence_errors_std = np.array(all_depth_tissue_fluence_errors_std)
            all_depth_vessel_fluence_errors_std = np.array(all_depth_vessel_fluence_errors_std)


            # Calculate mean and std across all forearms for each depth
            mean_depth_tissue_error = np.nanmean(all_depth_tissue_errors, axis=0) * 100
            std_depth_tissue_error = np.nanmean(all_depth_tissue_errors_std, axis=0) * 100
            mean_depth_vessel_error = np.nanmean(all_depth_vessel_errors, axis=0) * 100
            std_depth_vessel_error = np.nanmean(all_depth_vessel_errors_std, axis=0) * 100

            mean_depth_cal_tissue_error = np.nanmean(all_depth_cal_tissue_errors, axis=0) * 100
            std_depth_cal_tissue_error = np.nanmean(all_depth_cal_tissue_errors_std, axis=0) * 100
            mean_depth_cal_vessel_error = np.nanmean(all_depth_cal_vessel_errors, axis=0) * 100
            std_depth_cal_vessel_error = np.nanmean(all_depth_cal_vessel_errors_std, axis=0) * 100

            mean_depth_tissue_fluence_error = np.nanmean(all_depth_tissue_fluence_errors, axis=0) * 100
            std_depth_tissue_fluence_error = np.nanmean(all_depth_tissue_fluence_errors_std, axis=0) * 100
            mean_depth_vessel_fluence_error = np.nanmean(all_depth_vessel_fluence_errors, axis=0) * 100
            std_depth_vessel_fluence_error = np.nanmean(all_depth_vessel_fluence_errors_std, axis=0) * 100

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

            depths = np.arange(mean_depth_tissue_error.shape[0]) * 0.1  # Depth range

            # Subplot 1: Whole Tissue Errors
            ax1.plot(depths, mean_depth_tissue_error, label='LSU', color='blue')
            ax1.fill_between(depths, mean_depth_tissue_error - std_depth_tissue_error,
                             mean_depth_tissue_error + std_depth_tissue_error, color='blue', alpha=0.3)
            ax1.plot(depths, mean_depth_cal_tissue_error, label='Calibrated LSU', color='green')
            ax1.fill_between(depths, mean_depth_cal_tissue_error - std_depth_cal_tissue_error,
                             mean_depth_cal_tissue_error + std_depth_cal_tissue_error, color='green', alpha=0.3)
            ax1.plot(depths, mean_depth_tissue_fluence_error, label='Fluence compensated', color='violet')
            ax1.fill_between(depths, mean_depth_tissue_fluence_error - std_depth_tissue_fluence_error,
                             mean_depth_tissue_fluence_error + std_depth_tissue_fluence_error, color='violet', alpha=0.3)
            ax1.set_ylabel("Error (phantom) [%]")
            fig.legend(loc='upper center', ncol=5, frameon=False, fontsize="small", fancybox=True, bbox_to_anchor=(0.5, 1.04))
            ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # Remove x-ticks

            # Subplot 2: Vessel Errors
            ax2.plot(depths, mean_depth_vessel_error, label='Oxy Error (Vessels)', color='blue')
            ax2.fill_between(depths, mean_depth_vessel_error - std_depth_vessel_error,
                             mean_depth_vessel_error + std_depth_vessel_error, color='blue', alpha=0.3)
            ax2.plot(depths, mean_depth_cal_vessel_error, label='Cal Oxy Error (Vessels)', color='green')
            ax2.fill_between(depths, mean_depth_cal_vessel_error - std_depth_cal_vessel_error,
                             mean_depth_cal_vessel_error + std_depth_cal_vessel_error, color='green', alpha=0.3)
            ax2.plot(depths, mean_depth_vessel_fluence_error, label='Fluence Error (Vessels)', color='violet')
            ax2.fill_between(depths, mean_depth_vessel_fluence_error - std_depth_vessel_fluence_error,
                             mean_depth_vessel_fluence_error + std_depth_vessel_fluence_error, color='violet', alpha=0.3)
            ax2.set_ylabel("Error (vessels-only) [%]")
            ax2.yaxis.set_label_coords(-0.06, 0.5)
            # ax2.legend()

            # Shared x-axis label for the bottom plot
            ax2.set_xlabel("Depth [mm]")

            # Adjust layout for better spacing
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.2)  # Adjust this value as needed for spacing between subplots
            # plt.show()
            plt.savefig(os.path.join(results_folder, f"{image_modality}_depth_error.pdf"), dpi=300,
                        bbox_inches='tight')
            plt.close()

        with open(os.path.join(results_folder, f"{image_modality}_results.json"), "w") as json_file:
            json.dump(results, json_file, indent=4)
