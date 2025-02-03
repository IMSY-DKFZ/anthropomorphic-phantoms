#!/bin/sh

export RUN_BY_BASH="True"

#export BASE_PATH="/path/to/publication_data"
export BASE_PATH="/home/kris/Data/Dye_project/publication_data"

export PYTHON_PATH="$PWD"


python3 visualization/oxy_levels.py
python3 visualization/blood_proxies.py
python3 visualization/background_properties.py
python3 dye_analysis/plot_spectra.py
python3 pa_image_analysis/plot_laser_energies.py

python3 pa_image_analysis/example_pat_images.py
python3 hsi_image_analysis/example_hsi.py

python3 oxy_estimation/estimate_oxy_hsi.py
python3 oxy_estimation/estimate_oxy_pa.py
python3 oxy_estimation/evaluate_oxy.py

# For the supplementary material, we ran some simulations
# To rerun the simulations, set RUN_SIMULATION="True" and uncomment the following lines.

export RUN_SIMULATION="False"
export VISUALIZE="False"

python3 simulations/pat/air_bubble_simulation.py
python3 simulations/pat/speed_of_sound_comparison.py