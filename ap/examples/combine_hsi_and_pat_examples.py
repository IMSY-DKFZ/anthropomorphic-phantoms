import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import glob


def combine_two_pngs(path1, path2, output_path=None):
    # Read the images from file paths
    img1 = mpimg.imread(path1)
    img2 = mpimg.imread(path2)

    # Create a figure with 1 row and 2 columns of subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Display the first image on the left subplot
    ax1.imshow(img1)
    ax1.axis('off')  # Remove axis ticks/labels

    # Display the second image on the right subplot
    ax2.imshow(img2)
    ax2.axis('off')  # Remove axis ticks/labels

    # Optionally, remove spacing between subplots
    plt.subplots_adjust(wspace=0, hspace=0)

    # If an output_path is provided, save the combined image
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=400)
    else:
        plt.show()


if __name__ == '__main__':
    base_path = ""
    pat_examples = sorted(glob.glob(os.path.join(base_path, "Paper_Results", "PAT_Measurement_Correlation", "PAT_spectrum*.png")))
    hsi_examples = sorted(glob.glob(os.path.join(base_path, "Paper_Results", "HSI_Measurement_Correlation", "HSI_spectrum*.png")))
    for pat_example, hsi_example, oxy in zip(pat_examples, hsi_examples, [0, 100, 30, 50, 70]):
        combine_two_pngs(pat_example, hsi_example, output_path=os.path.join(base_path, "Paper_Results", "Plots",
                                                                            f"oxy_{oxy}_comparison.png"))

