import numpy as np
import matplotlib.pyplot as plt
import nrrd
from tmd.utils.io_iad_results import load_iad_results
from tmd.linear_unimxing import linear_spectral_unmixing
from scipy.stats import linregress
from collections import OrderedDict
import os

image_path = "/home/kris/Work/Data/TMD/KrisPhantoms_01_IPASC/Scan_3.nrrd"
labels_path = "/home/kris/Work/Data/TMD/KrisPhantoms_01_IPASC/Scan_3-labels.nrrd"

fitting_wavelengths = np.arange(700, 901, 10)

im, im_head = nrrd.read(image_path)
labels, labels_head = nrrd.read(labels_path)

mean_spectrum = np.mean(im[:, labels[11, :, :] == 1], axis=1)
std_spectrum = np.std(im[:, labels[11, :, :] == 1], axis=1)
mean_spectrum = np.interp(fitting_wavelengths, np.arange(660, 951, 5), mean_spectrum)

abs_spectrum = load_iad_results("/home/kris/Work/Data/TMD/DyeSpectra/20230510/B13B/B13B.npz")["mua"]
abs_spectrum = np.interp(fitting_wavelengths, np.arange(650, 950), abs_spectrum)

dye_spectra_dir = "/home/kris/Work/Data/TMD/DyeSpectra/Example_Spectra"
chromophore_spectra_dict = OrderedDict()
chromophore_spectra_dict["B05"] = load_iad_results(os.path.join(dye_spectra_dir, "B05A.npz"))["mua"]
chromophore_spectra_dict["B09"] = load_iad_results(os.path.join(dye_spectra_dir, "B09A.npz"))["mua"]
chromophore_spectra_dict["B05"] = np.interp(fitting_wavelengths, np.arange(650, 950), chromophore_spectra_dict["B05"])
chromophore_spectra_dict["B09"] = np.interp(fitting_wavelengths, np.arange(650, 950), chromophore_spectra_dict["B09"])

result = linear_spectral_unmixing(abs_spectrum, chromophore_spectra_dict, non_negativity_constraint=False)
print(result)
endmember_spectra = list(chromophore_spectra_dict.values())
resulting_spectrum = 0
for c_idx, concentration in enumerate(result):
    resulting_spectrum += concentration * endmember_spectra[c_idx]

slope, intercept, r_value, p_value, std_err = linregress(abs_spectrum, mean_spectrum)
print(slope, intercept, r_value, p_value, std_err)

plt.scatter(abs_spectrum, mean_spectrum, c="black", alpha=0.5, label="Measurements")
plt.plot(abs_spectrum, intercept + slope * abs_spectrum, color="green",
         label=f"Correlation (R={r_value:.2f}, p-value={p_value:.2f})")
# plt.gca().spines.right.set_visible(False)
# plt.gca().spines.top.set_visible(False)
plt.xlabel("Measured Absorption [cm$^{-1}$]")
plt.ylabel("MSOT signal [a.u.]")
plt.legend(loc="upper left")
plt.show()

# slope, intercept, r_value, p_value, std_err = linregress(resulting_spectrum, mean_spectrum)
# print(slope, intercept, r_value, p_value, std_err)
# plt.scatter(resulting_spectrum, mean_spectrum, c="black", alpha=0.1, label="measurements")
# plt.plot(resulting_spectrum, intercept + slope * resulting_spectrum, 'b', linestyle="dotted",
#          label=f"correlation (R={r_value:.2f})", alpha=0.6)
# plt.tick_params(direction='in')
# plt.gca().spines.right.set_visible(False)
# plt.gca().spines.top.set_visible(False)
# plt.xlabel("Ground Truth Absorption [cm$^{-1}$]")
# plt.ylabel("MSOT signal [a.u.]")
# plt.legend(loc="upper left")
# plt.show()

