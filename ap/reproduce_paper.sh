#!/bin/sh

export BASE_PATH="/path/to/publication_data"

export RUN_BY_BASH="True"
export PYTHON_PATH="$PWD"

# The following lines are used to run the code for the paper.
# by executing the following files, all results and figures from the paper are generated.
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
# To rerun the simulations, set RUN_SIMULATION="True".

export RUN_SIMULATION="False"
export VISUALIZE="False"

python3 simulations/pat/air_bubble_simulation.py
python3 simulations/pat/speed_of_sound_comparison.py