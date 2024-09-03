import matplotlib.pyplot as plt
import numpy as np
import patato as pat

file_path = "/home/kris/Data/Dye_project/PAT_Data/Processed_Data/Study_6/Scan_23.hdf5"
pa_data = pat.PAData.from_hdf5(file_path)

nx = 400 # number of pixels
lx = 4e-2 # m
geometry = pa_data.get_scan_geometry()
irf = pa_data.get_impulse_response()
pre_bp = pat.MSOTPreProcessor(lp_filter=7e6, hp_filter=5e3)

# ts_bp, settings_bp, _ = pre_bp.run(pa_data.get_time_series(), pa_data)

# das = pat.Backprojection(field_of_view=(lx, 0, lx),
#                              n_pixels=(nx, 0, nx))  # z axis must be specified but is empty in this case.

ts_model_based = pa_data.get_time_series().copy()
ts_model_based.raw_data = ts_model_based.raw_data / pa_data.get_overall_correction_factor()[:, :, None, None]

model_based = pat.ModelBasedReconstruction(field_of_view=(lx, 0, lx),
                                           n_pixels=(nx, 0, nx), model_max_iter=200, pa_example=pa_data, gpu=True,
                                           model_regulariser="TV", model_regulariser_lambda=1e100)
print("hello")
# rec_backprojection, _, _ = das.run(ts_bp, pa_data, **settings_bp)
rec_modelbased, _, _ = model_based.run(ts_model_based, pa_data)

np.save("/home/kris/Data/Dye_project/PAT_Data/Processed_Data/Study_15/test.npy", rec_modelbased.numpy_array)

fig, axes = plt.subplots(1, 2)
rec_modelbased.imshow(ax=axes[0])
axes[0].set_title("Model based")
# rec_backprojection.imshow(ax = axes[1])
# axes[1].set_title("Backprojection")
plt.show()

# arr = np.squeeze(np.load("/home/kris/Data/Dye_project/PAT_Data/Processed_Data/Study_15/test.npy"))
# recon = arr[7]
# print(arr.shape)
#
# plt.imshow(recon[10])
# plt.show()
