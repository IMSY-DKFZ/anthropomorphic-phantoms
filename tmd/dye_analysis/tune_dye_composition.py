import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from tmd.utils.io_iad_results import load_iad_results
from tmd.linear_unimxing import linear_spectral_unmixing
from tmd.data.load_icg_absorption import load_icg
from collections import OrderedDict
import simpa as sp
import os

unmixing_wavelengths = np.arange(700, 850)
target_spectrum_name = "Hb"


class NamedSlider(Slider):
    def __init__(self, ax, label, valmin, valmax, name="Default_name", **kwargs):
        super().__init__(ax, label, valmin, valmax, **kwargs)

        self.name = name

    def on_changed(self, func):
        """
        Connect *func* as callback function to changes of the slider value.

        Parameters
        ----------
        func : callable
            Function to call when slider is changed.
            The function must accept a single float as its arguments.

        Returns
        -------
        int
            Connection id (which can be used to disconnect *func*).
        """
        return self._observers.connect('changed', lambda val: func(val, self.name))


hbo2_spectrum, hb_spectrum, fat_spectrum, melanin_spectrum, water_spectrum = sp.get_simpa_internal_absorption_spectra_by_names(
    [sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_OXYHEMOGLOBIN,
     sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_DEOXYHEMOGLOBIN,
     sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_FAT,
     sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_MELANIN,
     sp.Tags.SIMPA_NAMED_ABSORPTION_SPECTRUM_WATER]
)
wavelengths = hb_spectrum.wavelengths
hb_spectrum, hbo2_spectrum, fat_spectrum, melanin_spectrum, water_spectrum = hb_spectrum.values, hbo2_spectrum.values, fat_spectrum.values, melanin_spectrum.values, water_spectrum.values

target_spectra = {
    "Hb": hb_spectrum,
    "HbO2": hbo2_spectrum,
    "Fat": fat_spectrum,
    "Melanin": melanin_spectrum,
    "Water": water_spectrum,
}

target_spectrum = np.interp(unmixing_wavelengths, wavelengths, target_spectra[target_spectrum_name])

icg_wl, icg_mua = load_icg()
icg_mua = np.interp(unmixing_wavelengths, icg_wl, icg_mua)

dye_spectra_dir = "/home/kris/Data/Dye_project/Measured_Spectra"
example_spectra = os.listdir(dye_spectra_dir)

chromophore_spectra_dict = OrderedDict()
chromophore_spectra_dict["ICG"] = icg_mua
for example_spectrum in example_spectra:
    if "B15" not in example_spectrum:
        continue
    abs_spectrum = load_iad_results(os.path.join(dye_spectra_dir, example_spectrum))["mua"]
    chromophore_spectra_dict[example_spectrum.split(".")[0]] = np.interp(unmixing_wavelengths, np.arange(650, 950), abs_spectrum)

lsu_result = linear_spectral_unmixing(target_spectrum, chromophore_spectra_dict, non_negativity_constraint=True, weighted_optimization=True)
resulting_spectrum = 0
for c_idx, (chromophore_name, chromophore_value) in enumerate(chromophore_spectra_dict.items()):
    resulting_spectrum += lsu_result[c_idx] * chromophore_value

concentration_dict = {c_n: 0 for c_n in chromophore_spectra_dict.keys()}

fig, ax = plt.subplots(figsize=(15, 15))
fig.subplots_adjust(right=0.7, left=0.07)
ax.set_title(f"Target spectrum {target_spectrum_name}")
if target_spectrum_name == "Hb":
    c = "blue"
else:
    c = "red"
ax.semilogy(unmixing_wavelengths, target_spectrum, label=target_spectrum_name, color=c)
ax.semilogy(unmixing_wavelengths, resulting_spectrum, label="Mixed spectrum", color="green")
ax.set_ylabel("Absorption coefficient $\mu_a'$ [$cm^{-1}$]")
ax.set_xlabel("Wavelength [nm]")

slider_dict = {}
for c_idx, (chromophore_name, chromophore_value) in enumerate(chromophore_spectra_dict.items()):
    slider_axis = fig.add_axes([0.8, 0.05 + c_idx*0.05, 0.15, 0.03])
    if chromophore_name == "ICG":
        valmx = 1e-4
        valstep = 1e-3
    else:
        valmx = 10
        valstep = 0.01
    concentration_slider = NamedSlider(ax=slider_axis,
                                       label=f"Concentration of {chromophore_name}",
                                       valmin=0,
                                       valmax=valmx,
                                       valstep=valstep,
                                       valinit=lsu_result[c_idx],
                                       name=chromophore_name
                                       )
    concentration_dict[chromophore_name] = concentration_slider.val

    def update(value, slider_name):
        concentration_dict[slider_name] = value
        ax.clear()
        ax.semilogy(unmixing_wavelengths, target_spectrum, label=target_spectrum_name, color=c)
        resulting_spectrum = 0
        for chromophore_name, chromophore_value in chromophore_spectra_dict.items():
            resulting_spectrum += concentration_dict[chromophore_name] * chromophore_value
        ax.semilogy(unmixing_wavelengths, resulting_spectrum, label="Mixed spectrum", color="green")
        ax.set_ylabel("Absorption coefficient $\mu_a'$ [$cm^{-1}$]")
        ax.set_xlabel("Wavelength [nm]")
        ax.set_title(f"Target spectrum {target_spectrum_name}")
        plt.draw()

    concentration_slider.on_changed(update)
    slider_dict[chromophore_name] = concentration_slider

ax.legend()
plt.show()
