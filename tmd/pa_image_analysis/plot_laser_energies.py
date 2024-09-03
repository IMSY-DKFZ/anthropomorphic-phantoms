import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import patato as pat
import seaborn as sns


# ithera2 images
examples_images = {
    1: {"oxy": 0.5, "path": os.path.join("Study_25", "Scan_25.hdf5")},
    2: {"oxy": 0.3, "path": os.path.join("Study_26", "Scan_12.hdf5")},
    3: {"oxy": 0, "path": os.path.join("Study_27", "Scan_18.hdf5")},
    4: {"oxy": 0.7, "path": os.path.join("Study_28", "Scan_6.hdf5")},
    5: {"oxy": 1, "path": os.path.join("Study_31", "Scan_9.hdf5")},
    6: {"oxy": 1, "path": os.path.join("Study_32", "Scan_21.hdf5")},
}
base_path = "/home/kris/Data/Dye_project/PAT_Data/iThera_2_data/Processed_Data"

wavelengths = np.arange(700, 851, 10)

res_dict = dict()

for example_nr, ex_dict in examples_images.items():
    path = os.path.join(base_path, ex_dict["path"])

    pa_data = pat.PAData.from_hdf5(path)
    corr_fac = pa_data.get_overall_correction_factor()

    res_dict[example_nr] = dict()
    for wl_idx, wl in enumerate(wavelengths):
        res_dict[example_nr][wl] = corr_fac[:, wl_idx]

colors = sns.color_palette("husl", 6)

# Initialize the plot
plt.figure(figsize=(18, 8))

# Iterate over each dataset to create boxplots and scatter plots
for i, wl in enumerate(wavelengths):
    # Extract data for the current dataset
    boxplot_data = list()
    for j in range(1, 7):
        boxplot_data.extend(res_dict[j][wl])
        x = np.random.normal(loc=i * 7 + j - 0.5, scale=0.1, size=len(res_dict[j][wl]))
        y = res_dict[j][wl]
        if i == 0:
            plt.scatter(x, y, color=colors[j - 1], alpha=0.7, label=f"Phantom Nr. {j}")
        else:
            plt.scatter(x, y, color=colors[j - 1], alpha=0.7)

    # Create boxplot for the current dataset
    plt.boxplot(boxplot_data, positions=[i * 7 + 3], widths=6)

# Set x-ticks to be in the center of each set of boxplots
plt.xticks(np.arange(3, 7 * 16, 7), wavelengths)

plt.xlabel('Wavelength')
plt.ylabel('Laser Energy')
plt.title('Laser Energy Distribution')
plt.legend()
# plt.show()
plt.savefig("/home/kris/Data/Dye_project/Plots/laser_energies.png", dpi=300, bbox_inches="tight")