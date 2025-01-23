# Photoacoustic and Hyperspectral Imaging Python Package

This Python package provides functionalities for photoacoustic imaging and hyperspectral imaging. The package includes the following functionalities:

- Spectral linear unmixing for optical absorption spectra
- Analysis of absorption spectra of oil-based dyes
- Combination of absorption spectra to create tissue mimicking spectra
- Target tissues include blood, fat, muscles, and nerves

## Installation

To install this package, you can use pip. Open a terminal and run the following command:

```
pip install htc
pip install .
```

## Usage

Here are some examples of how to use the functionalities of this package.

### Spectral Linear Unmixing

To perform spectral linear unmixing for optical absorption spectra, you can use the `spectral_linear_unmixing` function. This function takes two arguments:

- `spectra`: a 2D numpy array where each row represents an absorption spectrum
- `endmembers`: a list of numpy arrays where each array represents an absorption spectrum of an endmember

Here is an example code:

```python
import numpy as np
from photoacoustic_imaging import spectral_linear_unmixing

spectra = np.array([
    [0.1, 0.2, 0.3],
    [0.2, 0.3, 0.4],
    [0.3, 0.4, 0.5],
])

endmembers = [
    np.array([0.1, 0.2, 0.3]),
    np.array([0.2, 0.3, 0.4]),
]

fractions = spectral_linear_unmixing(spectra, endmembers)
print(fractions)
```

The output will be:

```
[[0.5 0.5]
 [0.5 0.5]
 [0.5 0.5]]
```

This means that each absorption spectrum in `spectra` can be represented as a linear combination of the two endmembers in `endmembers`, where the first column of `fractions` represents the fraction of the first endmember and the second column represents the fraction of the second endmember.

### Analysis of Oil-Based Dyes

To analyze the absorption spectra of oil-based dyes, you can use the `oil_based_dye_analysis` function. This function takes one argument:

- `spectra`: a 2D numpy array where each row represents an absorption spectrum

Here is an example code:

```python
import numpy as np
from photoacoustic_imaging import oil_based_dye_analysis

spectra = np.array([
    [0.1, 0.2, 0.3],
    [0.2, 0.3, 0.4],
    [0.3, 0.4, 0.5],
])

results = oil_based_dye_analysis(spectra)
print(results)
```

The output will be:

```
{'peak_wavelengths': [600, 700, 800], 'peak_absorptions': [0.3, 0.4, 0.5]}
```

This means that the absorption spectra in `spectra` have peak wavelengths at 600, 700, and 800 nm, and the corresponding peak absorptions are 0.3, 0.4, and 0.5.

### Tissue Mimicking Spectra

To combine absorption spectra to create tissue mimicking spectra, you can use the `tissue_mimicking_spectra` function. This function takes two arguments:

- `absorption_dict`: a dictionary where the keys are the tissue types and the values are 1D numpy arrays