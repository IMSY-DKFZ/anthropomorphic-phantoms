import os
import matplotlib.pyplot as plt
import numpy as np
import patato as pat
import seaborn as sns
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

examples_images = {
    1: {"oxy": 0.5, "path": "Scan_25_time_series.hdf5"},
    2: {"oxy": 0.3, "path": "Scan_12_time_series.hdf5"},
    3: {"oxy": 0, "path": "Scan_19_time_series.hdf5"},
    4: {"oxy": 0.7, "path": "Scan_5_time_series.hdf5"},
    5: {"oxy": 1, "path": "Scan_9_time_series.hdf5"},
}

wavelengths = np.arange(700, 851, 10)

res_dict = dict()

for example_nr, ex_dict in examples_images.items():
    path = os.path.join(base_path, "PAT_Data", f"Phantom_0{example_nr}", ex_dict["path"])

    pa_data = pat.PAData.from_hdf5(path)
    corr_fac = pa_data.get_overall_correction_factor()

    res_dict[example_nr] = dict()
    for wl_idx, wl in enumerate(wavelengths):
        res_dict[example_nr][wl] = corr_fac[:, wl_idx]

colors = sns.color_palette("husl", 6)

# Initialize the plot
plt.figure(figsize=(10, 5))

for plot_idx in range(1, 3):
    # plt.subplot(2, 1, plot_idx)
    if plot_idx == 2:
        continue
    # Iterate over each dataset to create boxplots and scatter plots
    for i, wl in enumerate(wavelengths):
        # Extract data for the current dataset
        boxplot_data = list()
        for j in range(1, len(examples_images) + 1):
            boxplot_data.extend(res_dict[j][wl])
            x = np.random.normal(loc=i * 7 + j - 0.5, scale=0.1, size=len(res_dict[j][wl]))
            y = res_dict[j][wl]
            if i == 0:
                plt.scatter(x, y, color=colors[j - 1], alpha=0.7, label=f"Phantom Nr. {j}")
            else:
                plt.scatter(x, y, color=colors[j - 1], alpha=0.7)

            if plot_idx == 2:
                if j == 3:
                    plt.boxplot(boxplot_data, positions=[i * 7 + 1.5], widths=3)
                    boxplot_data = list()
                elif j == 6:
                    plt.boxplot(boxplot_data, positions=[i * 7 + 4.5], widths=3)

        if plot_idx == 1:
            # Create boxplot for the current dataset
            box = plt.boxplot(boxplot_data, positions=[i * 7 + 3], widths=6, notch=True, vert=True)
            plt.setp(box["medians"], color="black")

    # Set x-ticks to be in the center of each set of boxplots
    plt.xticks(np.arange(3, 7 * 16, 7), wavelengths)

    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Laser energy [mJ]')
    # plt.title(f'Laser energy distribution {"all" if plot_idx == 1 else "day 1 and day 2"}')
    plt.legend()
# plt.show()
save_path = os.path.join(base_path, "Paper_Results/Plots/laser_energies.pdf")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path,
                    dpi=400, bbox_inches="tight", pad_inches=0, transparent=False)
