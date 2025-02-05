# Anthropomorphic phantoms with tissue-mimicking spectra

This repository contains code for the paper "Anthropomorphic tissue-mimicking phantoms for oximetry validation based on 
multispectral optical imaging", see [Paper](TODO) that was published in tje Journal of Biomedical Optics 2025.
As Python package, it provides functionalities for photoacoustic (PA) imaging and hyperspectral imaging (HSI).
The package includes the following functionalities:

- Reproducibility for all results and figures in the paper including:
  - Analysis of absorption spectra of oil-based dyes
  - Analysis of PA and HSI data
  - Simulation of PA data as quality assurance
  - Spectral linear unmixing for optical absorption spectra
- Combination of absorption spectra to create tissue-mimicking spectra

Project Organization
------------

    ├── LICENSE
    ├── README.md                           <- The top-level README for developers using this project.
    ├── pyproject.toml                      <- PEP 518 configuration file with requirements for the project.
    └── ap                                  <- Source code for use in this project.
        ├── __init__.py                     <- Makes ap a Python module
        ├── reproduce_paper.sh              <- Script to reproduce the paper results
        ├── fit_dyes_to_spectrum.py         <- Script to fit dyes to spectrum
        ├── data                            <- Functionality to load ICG and Methylene blue data
        ├── dye_analysis                    <- Functionality to plot, anylyse and optimise dye absorption spectra
        ├── examples                        <- Simple examples to show how to do linear unmixing with an optimizer
        ├── hsi_image_analysis              <- Script to analyse and estimate oxygenation in HSI images
        ├── oxy_estimation                  <- Functionality to estimate and evaluate oxygenation in spectral data
        ├── pa_image_analysis               <- Script to analyse and estimate oxygenation in PA images
        ├── simulations                     <- Scripts for all simulation studies
        ├── utils                           <- Utility functions
        └── visualization                   <- Scripts to create exploratory and results oriented visualizations

--------

* [Getting started](#installation-and-setup)
* [Run experiments](#run-experiments)
* [Custom dye optimisation](#custom-dye-optimisation)

# Installation and setup
To install this repository and its dependencies, you need to run the following commands:
1. Make sure that you have your preferred virtual environment activated (we recommend Python 3.10) with 
    one of the following commands:
    * `virtualenv ap` and then `source ap/bin/activate`
    * `conda create -n ap` and then `conda activate ap`
2. Install the project by running `pip install .` in the `anthropomorphic-phantoms` directory.

## Download data

The data needed to reproduce the paper results can be found here: TODO Link
The folder called `publication_data` has the following structure:

------------

    ├── 3D_Models                           <- 3D models of the forearm and the 3D-printable moulds
    ├── HSI_Data                            <- Hyperspectral data of 5 example forearms
    ├── Measured_Spectra                    <- Measured spectra of different dye samples (B05 - B43),
    │                                          Hb (BS0) and HbO2 (BIR3) proxies with 4 intermediate oxygenation levels
    │                                          (B90-B97), where BIR3 would be 100% oxygenated and all forearm background
    │                                          properties (BF1-BF9) including the OOD phantom (BF10A-BF10C).
    ├── PAT_Data                            <- Photoacoustic data of 5 example forearms including simulations.
    └── (Paper_Results)                     <- Location to store the results of the paper

--------


To reproduce the paper results you will have to define the environment variable indicated in line 3 of 
the `ap/reproduce_paper.sh` script.

```
export BASE_PATH="<path to where you put the downloaded data, e.g. /home/user/publication_data>"
```

# Run experiments

The easiest way of running all experiments in this repo is by using the `reproduce_paper.sh` script.
For example, open a terminal in the `ap` directory and run:

`bash reproduce_paper.sh`

or

`./reproduce_paper.sh`

This script will run all experiments in the correct order and save the results in 
the `publication_data/Paper_Results` folder.

## Run reconstructions

We provide the raw time series data of the PA images in the `publication_data/PAT_Data` folder with the filenames
`Scan_X_time_series.hdf5`.
In order to run the reconstructions, you can use the `run_reconstruction.py` script in the `ap/pa_image_analysis`
directory. Keep in mind to adjust the `base_path` variable in the script to the correct path where the data is stored.
The reconstructed images will be saved in the `Paper_Results/PAT_Reconstructions` folder which will be created.

## Run simulations

Similarly, the simulations can be run with the `air_bubble_simulation.py` and `speed_of_sound_comparison-py` scripts
in the `ap/simulations/pat` directory. The results will be saved in the `Paper_Results/PAT_Simulations` folder.
Make sure that you have simpa installed and all the requirements that it needs. Please refer to the 
[simpa GitHub page](https://github.com/IMSY-DKFZ/simpa) for all instructions.

# Custom dye optimisation

In this paper, dyes and their concentrations were selected and optimised to mimic the absorption spectra of Hemoglobin
(Hb) and Oxyhemoglobin (HbO2). The `fit_dyes_to_spectrum.py` script in the `ap` directory can be used to fit dyes to
any other spectrum, too. The script will use an optimizer to find the best combination of dyes and their concentrations.
It will use all spectra in the `publication_data/Measured_Spectra` folder and print the results to the console.

how to use it yourself:

- **Target Spectrum:**
Change the value of `target_spectrum_name` or pass your own target spectrum array.

- **Optimization Parameters:**
Adjust `n_iter` to change the number of optimization iterations or `max_concentration` to alter the allowable range of dye
concentrations.

- **Wavelength Range:**
Modify `unmixing_wavelengths` if your spectra are defined over a different set of wavelengths.
Ensure that all spectra (target and measured) are interpolated to the same wavelength grid.