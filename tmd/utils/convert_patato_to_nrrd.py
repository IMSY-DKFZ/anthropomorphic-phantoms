from patato import PAData
import numpy as np
import nrrd
import os


def convert_patato_output_to_nrrd(file_path):
    base_path, file_name = os.path.split(file_path)
    file_name = file_name.split(".")[0]

    data = PAData.from_hdf5(file_path)

    recon = data.get_scan_reconstructions()
    key_name = list(recon.keys())[0]
    recon = recon[key_name]
    recon = recon.numpy_array
    recon = np.squeeze(recon)

    save_path = os.path.join(base_path, file_name + ".nrrd")
    nrrd.write(save_path, recon)


if __name__ == "__main__":
    convert_patato_output_to_nrrd("/home/kris/Work/Data/TMD/KrisPhantoms_01_IPASC/Scan_5.hdf5")