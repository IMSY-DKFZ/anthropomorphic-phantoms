import os
os.environ["PAT_MAXIMUM_BATCH_SIZE"] = "10"
import patato as pat
from patato.data import get_msot_time_series_example, get_msot_phantom_example
import matplotlib.pyplot as plt
import numpy as np

padata = get_msot_phantom_example("clinical")[0:1]
nx = 333 # number of pixels
lx = 2.5e-2 # m
pre_processor = pat.MSOTPreProcessor(lp_filter=7e6, hp_filter=5e3) # can specify low pass/high pass/hilbert etc.

reconstructor = pat.Backprojection(field_of_view=(lx, lx, 0),
                                   n_pixels=(nx, nx, 1)) # z axis must be specified but is empty in this case.
unmixer = pat.SpectralUnmixer(chromophores=["Hb", "HbO2"],
                              wavelengths = padata.get_wavelengths(), rescaling_factor=4)
so2_calculator = pat.SO2Calculator(nan_invalid=True)

# All processing steps can be called with .run(input, padata, ...)
# They return a dataset, a dictionary and a list of extra data (e.g. so2 and thb could be returned in the list)

filtered_time_series, settings, _ = pre_processor.run(padata.get_time_series(), padata)
# settings here is a dictionary that includes the interpolated detection geometry. It is passed into the next step

reconstruction, _, _ = reconstructor.run(filtered_time_series, padata, padata.get_speed_of_sound(), **settings)

unmixed, _, _ = unmixer.run(reconstruction, None)
so2, _, _ = so2_calculator.run(unmixed, None)

# Overlay the sO2 on top of the reconstructed image:
masks = [padata.get_rois()["tumour_right", "0"],
         padata.get_rois()["tumour_left", "0"],
         padata.get_rois()["reference_", "0"]]

im1 = reconstruction.imshow(clim=(0, None))
im = so2.imshow(clim=(0, 1), roi_mask=masks, cmap="viridis")
plt.title("Reconstructed Image")
plt.colorbar(im, label="$sO_2^{MSOT}$")
plt.colorbar(im1, label="PA Signal [a.u.]")
plt.show()