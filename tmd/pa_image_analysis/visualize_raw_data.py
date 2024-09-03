import numpy as np
import matplotlib.pyplot as plt
import patato as pat
from mpl_toolkits.axes_grid1 import make_axes_locatable
import nrrd


# pa_data = pat.PAData.from_hdf5("/home/kris/Data/Dye_project/PAT_Data/Processed_Data/Study_25/Scan_1.hdf5")
pa_data = pat.PAData.from_hdf5(r"/home/kris/Data/Dye_project/PAT_Data/iThera_data/Processed_Data/Study_63/Scan_17.hdf5")
time_series = pa_data.get_time_series().raw_data
nr_frames = time_series.shape[0]
# frame = int(nr_frames // 2)
frame = 8
correction_factors = pa_data.get_overall_correction_factor()[frame, :, None, None]

time_series = time_series[frame, ...]
# nrrd.write("/home/kris/Data/Test/test_time_series.nrrd", time_series)
maximum = np.max(time_series)
print(f"Total maximum: {maximum} [a.u.]")

masked_time_series = time_series.copy()
masked_time_series[masked_time_series != maximum] = 0

wavelengths = np.arange(700, 951, 10)
max_wl = 0
max_wavelengths = list()
glob_max = np.max(time_series[:, :, 10:])
for idx, wl in enumerate(wavelengths):
    maximum_of_this_wl = np.max(time_series[idx, :, 10:])
    if maximum_of_this_wl == glob_max:
        max_wl = idx
        max_wavelengths.append(wl)

print(f"Signal maximum: {np.max(time_series[max_wl, :, 10:])} [a.u.] at {wavelengths[max_wl]} nm")
print(f"All saturating wavelengths: ", max_wavelengths)

plt.subplot(2, 1, 1)
img = plt.imshow(masked_time_series[max_wl, ..., 10:])
plt.title(f"Masked time series at {wavelengths[max_wl]} nm")
plt.xlabel("Time steps")
plt.ylabel("Detector element")
ax = plt.gca()
# ax.axes.xaxis.set_visible(False)
# ax.axes.xaxis.set_ticks([])
# ax.axes.yaxis.set_visible(False)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2%", pad=0.05)
# plt.clim(0, 1)
plt.colorbar(img, cax=cax, orientation="vertical")
plt.subplot(2, 1, 2)
plt.plot(time_series[4, 120, :])
plt.title(f"Raw time series at {wavelengths[max_wl]} nm at mid-detector")
plt.xlabel("Time steps")
plt.ylabel("PA signal")
plt.show()


