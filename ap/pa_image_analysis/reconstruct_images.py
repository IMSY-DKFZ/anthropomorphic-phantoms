from patato import PAData
import numpy as np
import matplotlib.pyplot as plt
import nrrd
from matplotlib_scalebar.scalebar import ScaleBar

image_path = "/home/kris/Work/Data/TMD/KrisPhantoms_01_IPASC/Scan_3.nrrd"
labels_path = "/home/kris/Work/Data/TMD/KrisPhantoms_01_IPASC/Scan_3-labels.nrrd"

im, im_head = nrrd.read(image_path)
labels, labels_head = nrrd.read(labels_path)

mean_spectrum = np.mean(im[:, labels[11, :, :] == 1], axis=1)
std_spectrum = np.std(im[:, labels[11, :, :] == 1], axis=1)

plt.subplot(1, 2, 1)
plt.imshow(labels[11, :, :], cmap="gray")
plt.imshow(im[58, :, :], cmap="viridis", alpha=0.5)
plt.axis("off")
scale_kwargs_defaults = dict(length_fraction=0.1, location="lower right",
                             font_properties=dict(size="xx-small"), box_alpha=0., color="w")
scalebar = ScaleBar(1, "m", **scale_kwargs_defaults)
ax = plt.gca()
ax.add_artist(scalebar)
plt.subplot(1, 2, 2)
plt.plot(np.arange(660, 951, 5), mean_spectrum, color="green")
plt.fill_between(np.arange(660, 951, 5), mean_spectrum, mean_spectrum + std_spectrum, color="green", alpha=0.5)
plt.fill_between(np.arange(660, 951, 5), mean_spectrum, mean_spectrum - std_spectrum, color="green", alpha=0.5)
plt.tight_layout()
plt.show()
