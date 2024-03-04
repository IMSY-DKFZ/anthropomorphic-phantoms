import numpy as np
import matplotlib.pyplot as plt
from skimage import segmentation, future, feature
from functools import partial
from sklearn.ensemble import RandomForestClassifier
from htc import Config, DataPath, DatasetImage, LabelMapping, settings, settings_atlas, tivita_wavelengths
from tmd.utils.postprocess_segmentation import filter_small_components
import os
import glob

np.random.seed(42)

base_data_path = "/home/kris/Data/Dye_project/HSI_Data/"
image_list = glob.glob(os.path.join(base_data_path, "2024_02_20_*"))

path = DataPath("/home/kris/Data/Dye_project/HSI_Data/2024_02_20_16_50_30")
rgb = path.read_rgb_reconstructed()

training_labels = np.zeros(rgb.shape[:2], dtype=np.uint8)
training_labels[:120] = 1
training_labels[:170, 400:] = 1
training_labels[380:] = 1
training_labels[:, 570:] = 1
training_labels[:, :60] = 1
training_labels[150:230, 100:200] = 2
training_labels[255:365, 325:500] = 2
training_labels[270:360, 80:250] = 2
training_labels[229:233, 435:468] = 3
training_labels[232:238, 249:335] = 3
training_labels[303:320, 528:558] = 3
training_labels[219:224, 530:543] = 3
training_labels[215:219, 548:555] = 3

features_func = partial(feature.multiscale_basic_features,
                        intensity=True, edges=True, texture=True,
                        channel_axis=-1)
features = features_func(rgb)
clf = RandomForestClassifier(n_estimators=50, n_jobs=-1,
                             max_depth=10, max_samples=0.05)
clf = future.fit_segmenter(training_labels, features, clf)

for image in image_list:
    print(os.path.basename(image))
    path = DataPath(image)
    rgb = path.read_rgb_reconstructed()
    features = features_func(rgb)
    seg = future.predict_segmenter(features, clf)
    seg = filter_small_components(seg)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rgb)
    axes[0].set_title('Image, mask and segmentation boundaries')
    axes[1].imshow(seg)
    axes[1].set_title('Segmentation before postprocessing')

    fig.tight_layout()

    plt.savefig(os.path.join(image, "segmentation.png"), dpi=300)
    plt.close()



