import numpy as np
from skimage.measure import regionprops, label
from typing import Tuple


def label_with_component_sizes(binary_image: np.ndarray, connectivity: int = None) -> Tuple[np.ndarray, dict]:
    labeled_image, num_components = label(binary_image, return_num=True, connectivity=connectivity)
    component_sizes = {i + 1: j for i, j in enumerate(np.bincount(labeled_image.ravel())[1:])}
    return labeled_image, component_sizes


def filter_small_components(segmentation_mask, min_size: int = 200, connectivity: int = 1, nr_filters: int = 2):

    for n in range(nr_filters):
        seg_one_hot = np.zeros([*segmentation_mask.shape, 3])
        for idx in range(1, 4):
            seg_one_hot[segmentation_mask == idx, idx - 1] = 1
        binary_image = np.copy(segmentation_mask)
        labeled_image, component_sizes = label_with_component_sizes(binary_image, connectivity)
        properties = regionprops(labeled_image)
        keep = np.array([i for i, j in component_sizes.items() if j >= min_size])
        for components_value, size in component_sizes.items():
            if components_value not in keep:
                bbox = properties[components_value - 1].bbox
                zrr = np.zeros_like(segmentation_mask)
                zrr[bbox[0]-1:bbox[2]+1, bbox[1]-1:bbox[3]+1] = 1
                zrr[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 0
                one_hot_values = seg_one_hot[zrr == 1, :]
                label = np.argmax(np.count_nonzero(one_hot_values, axis=0))
                segmentation_mask[labeled_image == components_value] = label + 1

    return segmentation_mask
