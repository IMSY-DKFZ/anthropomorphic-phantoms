import numpy as np
from ap.dye_analysis.potential_target_spectra import get_target_spectrum
from ap.dye_analysis.measured_spectra import get_measured_spectra
from ap.dye_analysis.optimize_dye_concentrations import optimize_dye_concentrations
import simpa as sp


dye_spectra_dir = "/Path/to/publication_data/Measured_Spectra/"
unmixing_wavelengths = np.arange(700, 851, 10)

# You can choose the target spectrum from the following list:
# Hb, HbO2, Water, Fat, Melanin, Nickel_Sulphide, Copper_Sulphide, B90, B93, B95, B97, random
# You can also provide your own target spectrum by setting target_spectrum to a np.ndarray of your target
# and unmixing_wavelengths to the corresponding wavelengths.
# Make sure len(target_spectrum) == len(unmixing_wavelengths)!
target_spectrum_name = "Hb"

target_spectrum = get_target_spectrum(target_spectrum_name=target_spectrum_name,
                                      unmixing_wavelengths=unmixing_wavelengths,
                                      dye_spectra_dir=dye_spectra_dir)

measured_spectra_dict = get_measured_spectra(spectra_dir=dye_spectra_dir,
                                             unmixing_wavelengths=unmixing_wavelengths)

result = optimize_dye_concentrations(target_spectrum=target_spectrum, unmixing_wavelengths=unmixing_wavelengths,
                                     input_spectra=measured_spectra_dict, plot_mixing_results=True,
                                     n_iter=10000, max_concentration=5)

print("Mixing Ratios:")
total_concentration = sum(result.values())
for dye_name, concentration in result.items():
    print(f"{dye_name}: {concentration/total_concentration:.3f}")
