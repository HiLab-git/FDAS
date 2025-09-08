"""
Process Inference Masks
-----------------------
This script processes the masks generated from SAM inference
"""

import numpy as np
import os
import h5py
import cv2
from tqdm import tqdm
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt


def normalize(x):
    return x / 255.0 * 2 - 1


def caculate_anchor_submasks_num(input_folder):
    count_values = []
    for file in tqdm(os.listdir(input_folder)):
        if file.endswith('.h5'):
            file_path = os.path.join(input_folder, file)
            with h5py.File(file_path, 'r') as f:
                masks = f['masks'][:]
            labeled_masks = label(masks)
            props = regionprops(labeled_masks)
            sorted_regions = sorted(props, key=lambda x: x.area, reverse=True)
            total_foreground_area = np.sum(masks > 0)
            threshold_area = 0.95 * total_foreground_area if total_foreground_area > 0 else 0
            accumulated_area = 0
            count = 0
            for region in sorted_regions:
                accumulated_area += region.area
                count += 1
                if accumulated_area >= threshold_area:
                    break
            count_values.append(count)
    area_num = int(np.ceil(np.mean(count_values))) if count_values else 0

    return area_num


def reorder_masks_by_area(new_masks):
    unique_labels, counts = np.unique(new_masks, return_counts=True)
    sorted_indices = np.argsort(-counts)
    sorted_labels = unique_labels[sorted_indices]
    label_mapping = {label: new_idx for new_idx, label in enumerate(sorted_labels)}
    remapped_masks = np.vectorize(label_mapping.get)(new_masks)
    remapped_masks += 1

    return remapped_masks


def process_masks(input_folder, output_folder, area_num, target_size):
    for file in tqdm(os.listdir(input_folder)):
        if file.endswith('.h5'):
            file_path = os.path.join(input_folder, file)
            with h5py.File(file_path, 'r') as f:
                masks = f['masks'][:]
                images = f['image'][:]
            labeled_masks = masks
            props = regionprops(labeled_masks)
            sorted_regions = sorted(props, key=lambda x: x.area, reverse=True)
            largest_regions = sorted_regions[:area_num]
            largest_labels = [region.label for region in largest_regions]

            new_masks = np.full_like(labeled_masks, fill_value=-1, dtype=np.int32)
            for idx, label_value in enumerate(largest_labels, start=1):
                new_masks[labeled_masks == label_value] = idx

            new_masks[labeled_masks == 0] = 0
            remaining_mask = (new_masks == -1)
            large_region_mask = (new_masks > -1)

            distance_map, nearest_index = distance_transform_edt(~large_region_mask, return_indices=True)
            nearest_large_region = new_masks[nearest_index[0], nearest_index[1]]
            new_masks[remaining_mask] = nearest_large_region[remaining_mask]
            new_masks = reorder_masks_by_area(new_masks)
            unique_labels = np.unique(new_masks)

            resized_image = cv2.resize(images, target_size, interpolation=cv2.INTER_LINEAR)
            normalized_image = normalize(resized_image)
            normalized_masks = cv2.resize(new_masks, target_size, interpolation=cv2.INTER_NEAREST)

            output_h5_path = os.path.join(output_folder, file)
            with h5py.File(output_h5_path, 'w') as out_f:
                out_f.create_dataset('image', data=normalized_image, compression='gzip')
                out_f.create_dataset('masks', data=normalized_masks, compression='gzip')

    print("Number of classes in new_masks:", len(unique_labels))


def write_list(data_dir, list_file):
    file_names = os.listdir(data_dir)
    file_names.sort()
    with open(list_file, 'w') as f:
        for name in file_names:
            f.write(name + '\n')


if __name__ == '__main__':
    dataset = "mms"
    input_folder = f"./outputs/{dataset}_masks"
    output_folder = f"./outputs/{dataset}_pre_processed_masks/data"
    os.makedirs(output_folder, exist_ok=True)
    list_file = f"./outputs/{dataset}_pre_processed_masks/train_slices.list"
    target_size = (256, 256)
    area_num = caculate_anchor_submasks_num(input_folder)
    process_masks(input_folder, output_folder, area_num, target_size)
    write_list(output_folder, list_file)
