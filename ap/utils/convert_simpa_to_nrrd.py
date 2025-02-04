import simpa as sp
import numpy as np
import matplotlib.pyplot as plt
import nrrd
import os
from glob import glob


def convert_simpa_output_to_nrrd(file_path, all_wave_lengths=False):
    base_path, file_name = os.path.split(file_path)
    file_name = file_name.split(".")[0]

    if not all_wave_lengths:
        recon = np.flipud(sp.load_data_field(file_path, sp.Tags.DATA_FIELD_RECONSTRUCTED_DATA, wavelength=800))
    else:
        wavelengths = sp.load_data_field(file_path, sp.Tags.SETTINGS)[sp.Tags.WAVELENGTHS]
        recon = np.stack([np.flipud(sp.load_data_field(file_path, sp.Tags.DATA_FIELD_RECONSTRUCTED_DATA, wavelength=wl)) for wl in wavelengths])

    # plt.imshow(recon)
    # plt.show()
    # exit()

    save_path = os.path.join(base_path, file_name + ".nrrd" if not all_wave_lengths else file_name + "_wl.nrrd")
    nrrd.write(save_path, recon)


if __name__ == "__main__":
    base_path = f"/path/to/publication_data/PAT_Data/iThera_2_data/US_analysis/"
    files = glob(os.path.join(base_path, "Study*", "*.hdf5"))

    for file in files:
        try:
            convert_simpa_output_to_nrrd(file)
        except KeyError:
            continue

    # convert_simpa_output_to_nrrd("/path/to/publication_data/PAT_Data/iThera_2_data/Reconstructions_das/Study_26/Scan_4_recon.hdf5", all_wave_lengths=False)
