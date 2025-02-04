import numpy as np
import os


def load_icg():
    try:
        run_by_bash: bool = bool(os.environ["RUN_BY_BASH"])
        print("This runner script is invoked in a bash script!")
    except KeyError:
        run_by_bash: bool = False

    if run_by_bash:
        base_path = os.environ["PYTHON_PATH"]
    else:
        # In case the script is run from an IDE, the base path has to be set manually
        base_path = "/path/to/anthropomorphic-phantoms/ap"
    file_path = os.path.join(base_path, "data/icg.csv")
    wavelengths = np.loadtxt(file_path, skiprows=2, usecols=0)
    mua = 2.303 * np.loadtxt(file_path, skiprows=2, usecols=1)
    return wavelengths, mua
