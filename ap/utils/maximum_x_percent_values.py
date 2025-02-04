import numpy as np
import simpa as sp
import matplotlib.pyplot as plt


def top_x_percent_indices(array, mask, x_percent):
    # Step 1: Apply the mask to the original array
    masked_array = np.where(mask, array, np.nan)

    # Step 2: Flatten the masked array
    flat_masked_array = masked_array.flatten()

    # Step 3: Determine the threshold for the top x% values
    num_values = np.sum(mask)  # Number of values in the ROI
    top_x_count = int(np.ceil(x_percent * num_values / 100))

    # Handle case where x_percent is too small to capture any value
    if top_x_count == 0:
        raise ValueError("The x_percent is too small to select any value.")

    # Get the threshold value
    threshold = np.nanpercentile(flat_masked_array, 100 - x_percent)

    # Step 4: Find the indices of the top x% values
    top_indices = np.argwhere(masked_array >= threshold)

    # Ensure that we only return the top_x_count indices in case of tie at the threshold
    top_values = masked_array[top_indices[:, 0], top_indices[:, 1]]
    sorted_indices = top_indices[np.argsort(-top_values)]

    return sorted_indices[:top_x_count]


if __name__ == "__main__":
    image_path = "/path/to/PAT_Data/Phantom_01/Scan_25_recon.hdf5"
    recon = sp.load_data_field(image_path, sp.Tags.DATA_FIELD_RECONSTRUCTED_DATA)
    recon_array = np.stack(
        [np.rot90(recon[str(wl)], 3) for wl in sp.load_data_field(image_path, sp.Tags.SETTINGS)[sp.Tags.WAVELENGTHS]])
    training_labels = np.rot90(sp.load_data_field(image_path, sp.Tags.DATA_FIELD_SEGMENTATION), 3)
    plt.subplot(2, 2, 1)
    plt.imshow(recon_array[10])
    plt.contour(training_labels, colors='r', alpha=0.5)
    # Example usage:
    array = recon_array[10]

    mask = training_labels

    x_percent = 5  # Get the top 20% maximum values within the masked region

    indices = top_x_percent_indices(array, mask, x_percent)
    print("Indices of the top 20% maximum values:", indices)

    seg_array = np.zeros_like(array)
    maximum_value_pixels = list()
    for idx in indices:
        maximum_value_pixels.append(recon_array[:, idx[0], idx[1]])
        seg_array[idx[0], idx[1]] = 1

    maximum_value_pixels = np.array(maximum_value_pixels)
    mean_spectrum = np.mean(maximum_value_pixels, axis=0)
    # mean_spectrum = np.mean(recon_array[:, training_labels == 1], axis=1)

    plt.subplot(2, 2, 2)
    plt.imshow(seg_array)

    plt.subplot(2, 1, 2)
    plt.plot(mean_spectrum)

    plt.show()