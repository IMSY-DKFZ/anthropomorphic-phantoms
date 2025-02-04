import os
from simpa import Tags
import simpa as sp
import numpy as np
import matplotlib.pyplot as plt
import glob
plt.switch_backend("TkAgg")

base_path = "/path/to/publication_data/publication_data/"
visualize = True

publication_data = sorted(glob.glob(os.path.join(base_path, "PAT_Data", "Phantom_*", "Scan_*_recon*")))
reproduction_data = sorted(glob.glob(os.path.join(base_path, "Paper_Results", "PAT_Reconstructions",
                                                  "Phantom_*", "Scan_*_recon*")))

assert len(publication_data) == len(reproduction_data), \
    ("Number of publication data and reproduction data do not match."
     "Please run the reconstructions using ap/pa_image_analysis/run_reconstruction.")

for phantom_idx, (pub_data_path, rep_data_path) in enumerate(zip(publication_data, reproduction_data)):
    wavelengths = sp.load_data_field(pub_data_path, Tags.SETTINGS)[Tags.WAVELENGTHS]

    pub_recon = sp.load_data_field(pub_data_path, Tags.DATA_FIELD_RECONSTRUCTED_DATA)
    rep_recon = sp.load_data_field(rep_data_path, Tags.DATA_FIELD_RECONSTRUCTED_DATA)

    for wl_idx, wl in enumerate(wavelengths):
        pub_data = pub_recon[str(wl)]
        rep_data = rep_recon[str(wl)]

        if visualize and wl_idx == 0:
            plt.figure()
            plt.subplot(1, 3, 1)
            plt.imshow(pub_data)
            plt.title("Publication data")
            plt.subplot(1, 3, 2)
            plt.imshow(rep_data)
            plt.title("Reproduction data")
            plt.subplot(1, 3, 3)
            diff = pub_data - rep_data
            vmax = np.abs(diff).max()
            plt.imshow(diff, cmap="seismic", vmax=vmax, vmin=-vmax)
            plt.colorbar()
            plt.title("Difference")
            plt.show()

        print(np.abs(pub_data - rep_data).max())
        assert np.allclose(pub_data, rep_data, atol=0.001*np.max(pub_data)), \
            f"Data for {pub_data_path} and {rep_data_path} at wavelength {wl} do not match."
