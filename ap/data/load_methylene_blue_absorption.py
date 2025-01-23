import numpy as np
from scipy.ndimage import gaussian_filter1d


def load_mb():
    file_path = "/home/kris/Work/Repositories/tissue-mimicking-dyes/ap/data/methylene_blue.csv"
    wavelengths = np.loadtxt(file_path, skiprows=2, usecols=0)
    mua = 2.303 * np.loadtxt(file_path, skiprows=2, usecols=1)
    return wavelengths, gaussian_filter1d(mua, 2)
