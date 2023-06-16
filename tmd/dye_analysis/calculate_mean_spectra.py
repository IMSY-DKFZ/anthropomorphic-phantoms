import os
import glob
import numpy as np
from tmd.utils.io_iad_results import load_iad_results

dye_base_dir = "/home/kris/Work/Data/TMD/DyeSpectra"
excluded_phantoms = ["B01A", "B01B", "B02A", "B02B", "B03A", "B03B", "B13A", "B13B", "B08A", "B08B", "B24A"]

spectrum_files = glob.glob((os.path.join(dye_base_dir, "*", "B*", "*.npz")))
spectrum_files = [file for file in spectrum_files if os.path.basename(file).split(".")[0] not in excluded_phantoms]

unique_phantom_names = sorted(list(set([os.path.basename(file).split(".")[0][:3] for file in spectrum_files])))

for phantom_name in unique_phantom_names:
    files = [file for file in spectrum_files if phantom_name in file]

    if len(files) == 1:
        save_dict = load_iad_results(files[0])

    else:
        data_dict_A = load_iad_results(files[0])
        data_dict_B = load_iad_results(files[1])

        save_dict = dict()
        for (key_A, value_A), (key_B, value_B) in zip(data_dict_A.items(), data_dict_B.items()):
            if "std" in key_A:
                save_dict[key_A] = 0.5 * np.sqrt(value_A**2 + value_B**2)
            else:
                save_dict[key_A] = (value_A + value_B)/2

    np.savez(os.path.join(dye_base_dir, "Measured_Spectra", f"{phantom_name}.npz"), **save_dict)


