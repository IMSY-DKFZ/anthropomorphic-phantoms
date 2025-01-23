import numpy as np


def load_icg():
    file_path = "/home/kris/Work/Repositories/tissue-mimicking-dyes/ap/data/icg.csv"
    wavelengths = np.loadtxt(file_path, skiprows=2, usecols=0)
    mua = 2.303 * np.loadtxt(file_path, skiprows=2, usecols=1)
    return wavelengths, mua
