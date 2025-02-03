import numpy as np
import matplotlib.pyplot as plt
import simpa as sp
from ap.utils.io_iad_results import load_iad_results
import os
plt.switch_backend("TkAgg")


def generate_tissue_spectra(nr_of_spectra: int,
                            wavelength_range: np.ndarray,
                            max_number_of_mixed_spectra: int = 5,
                            plot_spectra: bool = False) -> list:
    """Mixes multiple tissue spectra from the simpa_recons library in order to create new tissue-based spectra.

    :param nr_of_spectra: number opf spectra that should be returned
    :param wavelength_range: numpy array with the exact wavelengths that should be used to create the spectra
    :param max_number_of_mixed_spectra: max number of the spectra that should be used as mixing components
    :param plot_spectra: flag whether to plot the spectra
    :return: list of newly sampled spectra
    """

    # np.random.seed(42)
    abs_spectra = sp.AbsorptionSpectrumLibrary()
    new_spectra = list()

    for i in range(nr_of_spectra):
        # randomly draw the number of how many spectra should be used for the mixing
        nr_of_randomly_selected_tissue_spectra = np.random.randint(2, max_number_of_mixed_spectra)
        # randomly draw spectra from the simpa_recons library
        randomly_selected_tissue_spectra = np.random.choice(abs_spectra.spectra, nr_of_randomly_selected_tissue_spectra,
                                                            replace=False)

        new_spectrum = 0
        for tissue_spectrum in randomly_selected_tissue_spectra:
            # factor that scales the contribution of this specific spectrum
            random_factor = np.random.uniform(0.35, 0.65)
            new_spectrum += random_factor * np.interp(wavelength_range,
                                                      tissue_spectrum.wavelengths,
                                                      tissue_spectrum.values)

        # normalize the outcome by the amount of spectra that were used
        new_spectrum /= (nr_of_randomly_selected_tissue_spectra / 2)    # division by 2 to account for the
        new_spectra.append(new_spectrum)

        if plot_spectra:
            plt.plot(wavelength_range, new_spectrum)
            plt.show()
    if len(new_spectra) == 1:
        new_spectra = new_spectra[0]
    return new_spectra


def save_spectra_in_iad_format(absorption_spectra: list, save_path: str, example_file_path: str):
    """Saves absorption spectra in the form of the iad results by changing the absorption spectra in an example file.

    :param absorption_spectra: spectra to save
    :param save_path: base path of a folder, where to save the results.
    :param example_file_path: file path with actual iad results in it
    """

    example_file = load_iad_results(example_file_path)
    for s_idx, absorption_spectrum in enumerate(absorption_spectra):
        save_file = example_file.copy()
        save_file["mua"] = absorption_spectrum

        np.savez(os.path.join(save_path, f"example_spectrum_{s_idx + 1:0}"), **save_file)


if __name__ == "__main__":
    wavelengths = np.arange(650, 950)
    spectra_number = 18
    generated_spectra = generate_tissue_spectra(spectra_number, wavelengths, plot_spectra=True)
