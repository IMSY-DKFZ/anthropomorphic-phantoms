import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import json
plt.rcParams.update({'font.size': 12,
                     "font.family": "serif"})

def calculate_mean_std(data, mask=None):
    if mask is not None:
        data = data[mask]
    return np.nanmean(data), np.nanstd(data)


def hierarchical_std(means, stds, overall_mean):
    N = len(means)
    variance = np.sum([std**2 + (mean - overall_mean)**2 for mean, std in zip(means, stds)]) / N
    return np.sqrt(variance)


def mask_arrays(arr, mask):
    if mask is not None:
        arr[mask == 0] = np.nan
        return arr
    return arr


image_modality = "PAT"


if __name__ == "__main__":
    base_path = f"/home/kris/Data/Dye_project/{image_modality}_Data/Oxy_Results/"
    forearm_list = glob.glob(os.path.join(base_path, "forearm_*.npz"))

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

    for forearm in forearm_list:
        name = f"Forearm {forearm.split('_')[-1].split('.')[0]}"
        print(name)
        data = np.load(forearm)

        tissue_mask = data["tissue"]
        vessel_mask = data["vessels"]

        tissue_error_mean, tissue_error_std = calculate_mean_std(data["oxy_error"], tissue_mask)
        vessel_error_mean, vessel_error_std = calculate_mean_std(data["oxy_error"], vessel_mask)
        cal_tissue_error_mean, cal_tissue_error_std = calculate_mean_std(data["cal_oxy_error"], tissue_mask)
        cal_vessel_error_mean, cal_vessel_error_std = calculate_mean_std(data["cal_oxy_error"], vessel_mask)

        all_tissue_errors.append((tissue_error_mean, tissue_error_std))
        all_vessel_errors.append((vessel_error_mean, vessel_error_std))
        all_cal_tissue_errors.append((cal_tissue_error_mean, cal_tissue_error_std))
        all_cal_vessel_errors.append((cal_vessel_error_mean, cal_vessel_error_std))

        data_shape = data["oxy_error"].shape
        depth_axis = np.argmax(data_shape)

        depth_tissue_error = np.nanmean(mask_arrays(data["oxy_error"], tissue_mask), axis=depth_axis)
        depth_vessel_error = np.nanmean(mask_arrays(data["oxy_error"], vessel_mask), axis=depth_axis)
        depth_cal_tissue_error = np.nanmean(mask_arrays(data["cal_oxy_error"], tissue_mask), axis=depth_axis)
        depth_cal_vessel_error = np.nanmean(mask_arrays(data["cal_oxy_error"], vessel_mask), axis=depth_axis)

        all_depth_tissue_errors.append(depth_tissue_error)
        all_depth_vessel_errors.append(depth_vessel_error)
        all_depth_cal_tissue_errors.append(depth_cal_tissue_error)
        all_depth_cal_vessel_errors.append(depth_cal_vessel_error)

        if image_modality == "PAT":
            tissue_fluence_error_mean, tissue_fluence_error_std = calculate_mean_std(data["fluence_corr_error"],
                                                                                     tissue_mask)
            vessel_fluence_error_mean, vessel_fluence_error_std = calculate_mean_std(data["fluence_corr_error"],
                                                                                     vessel_mask)

            all_tissue_fluence_errors.append((tissue_fluence_error_mean, tissue_fluence_error_std))
            all_vessel_fluence_errors.append((vessel_fluence_error_mean, vessel_fluence_error_std))

            depth_tissue_fluence_error = np.nanmean(mask_arrays(data["fluence_corr_error"], tissue_mask),
                                                    axis=depth_axis)
            depth_vessel_fluence_error = np.nanmean(mask_arrays(data["fluence_corr_error"], vessel_mask),
                                                    axis=depth_axis)

            all_depth_tissue_fluence_errors.append(depth_tissue_fluence_error)
            all_depth_vessel_fluence_errors.append(depth_vessel_fluence_error)

    # Aggregate means and standard deviations
    tissue_error_means = [x[0] for x in all_tissue_errors]
    tissue_error_stds = [x[1] for x in all_tissue_errors]
    vessel_error_means = [x[0] for x in all_vessel_errors]
    vessel_error_stds = [x[1] for x in all_vessel_errors]

    cal_tissue_error_means = [x[0] for x in all_cal_tissue_errors]
    cal_tissue_error_stds = [x[1] for x in all_cal_tissue_errors]
    cal_vessel_error_means = [x[0] for x in all_cal_vessel_errors]
    cal_vessel_error_stds = [x[1] for x in all_cal_vessel_errors]

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

    results = {
        "Overall Tissue Error Mean": overall_tissue_error_mean,
        "Overall Tissue Error Std": overall_tissue_error_std,
        "Overall Vessel Error Mean": overall_vessel_error_mean,
        "Overall Vessel Error Std": overall_vessel_error_std,
        "Overall Calibrated Tissue Error Mean": overall_cal_tissue_error_mean,
        "Overall Calibrated Tissue Error Std": overall_cal_tissue_error_std,
        "Overall Calibrated Vessel Error Mean": overall_cal_vessel_error_mean,
        "Overall Calibrated Vessel Error Std": overall_cal_vessel_error_std,
    }

    if image_modality == "HSI":
        print(results)

    # with open(os.path.join(base_path, "results.json"), "w") as json_file:
    #     json.dump(results, json_file)

    else:
        tissue_fluence_error_means = [x[0] for x in all_tissue_fluence_errors]
        tissue_fluence_error_stds = [x[1] for x in all_tissue_fluence_errors]
        vessel_fluence_error_means = [x[0] for x in all_vessel_fluence_errors]
        vessel_fluence_error_stds = [x[1] for x in all_vessel_fluence_errors]

        overall_tissue_fluence_error_mean = np.mean(tissue_fluence_error_means)
        overall_vessel_fluence_error_mean = np.mean(vessel_fluence_error_means)

        overall_tissue_fluence_error_std = np.mean(tissue_fluence_error_stds)
        overall_vessel_fluence_error_std = np.mean(vessel_fluence_error_stds)

        results["Overall Tissue Fluence Error Mean"] = overall_tissue_fluence_error_mean,
        results["Overall Tissue Fluence Error Std"] = overall_tissue_fluence_error_std,
        results["Overall Vessel Fluence Error Mean"] = overall_vessel_fluence_error_mean,
        results["Overall Vessel Fluence Error Std"] = overall_vessel_fluence_error_std,

        print(results)

        # Calculate hierarchical std

        # Convert lists to arrays for easier manipulation
        all_depth_tissue_errors = np.array(all_depth_tissue_errors)
        all_depth_vessel_errors = np.array(all_depth_vessel_errors)
        all_depth_cal_tissue_errors = np.array(all_depth_cal_tissue_errors)
        all_depth_cal_vessel_errors = np.array(all_depth_cal_vessel_errors)
        all_depth_tissue_fluence_errors = np.array(all_depth_tissue_fluence_errors)
        all_depth_vessel_fluence_errors = np.array(all_depth_vessel_fluence_errors)


        # Calculate mean and std across all forearms for each depth
        mean_depth_tissue_error = np.nanmean(all_depth_tissue_errors, axis=0) * 100
        std_depth_tissue_error = np.nanstd(all_depth_tissue_errors, axis=0) * 100
        mean_depth_vessel_error = np.nanmean(all_depth_vessel_errors, axis=0) * 100
        std_depth_vessel_error = np.nanstd(all_depth_vessel_errors, axis=0) * 100
        mean_depth_cal_tissue_error = np.nanmean(all_depth_cal_tissue_errors, axis=0) * 100
        std_depth_cal_tissue_error = np.nanstd(all_depth_cal_tissue_errors, axis=0) * 100
        mean_depth_cal_vessel_error = np.nanmean(all_depth_cal_vessel_errors, axis=0) * 100
        std_depth_cal_vessel_error = np.nanstd(all_depth_cal_vessel_errors, axis=0) * 100

        mean_depth_tissue_fluence_error = np.nanmean(all_depth_tissue_fluence_errors, axis=0) * 100
        std_depth_tissue_fluence_error = np.nanstd(all_depth_tissue_fluence_errors, axis=0) * 100
        mean_depth_vessel_fluence_error = np.nanmean(all_depth_vessel_fluence_errors, axis=0) * 100
        std_depth_vessel_fluence_error = np.nanstd(all_depth_vessel_fluence_errors, axis=0) * 100

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
        ax1.yaxis.set_label_coords(-0.06, 0.5)
        ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # Remove x-ticks

        ax1.legend()
        ax1.set_title("Error vs. depth")

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
        plt.savefig(os.path.join(base_path, "depth_error.png"), dpi=300)
        # plt.close()