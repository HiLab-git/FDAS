"""
Inference script using SAM to generate segmentation masks
"""

import torch
import os
import numpy as np
import cv2
import SimpleITK as sitk
import random
import matplotlib.pyplot as plt
import h5py
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


def convert_gray(rgb_img_array):
    gray_img = np.zeros((rgb_img_array.shape[0], rgb_img_array.shape[1]), dtype=np.uint8)
    unique_colors = np.unique(rgb_img_array.reshape(-1, rgb_img_array.shape[2]), axis=0)
    color_to_gray_map = {tuple(color): i for i, color in enumerate(unique_colors)}
    for i in range(rgb_img_array.shape[0]):
        for j in range(rgb_img_array.shape[1]):
            rgb_value = tuple(rgb_img_array[i, j, :])
            gray_img[i, j] = color_to_gray_map[rgb_value]
    return gray_img


def save_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    return img


def main(input_fold, save_data_root, sam_path):
    sam = sam_model_registry["vit_h"](checkpoint=sam_path).to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    all_niis = [f for f in os.listdir(input_fold) if f.endswith("nii.gz")]
    random.shuffle(all_niis)

    for nii_file in all_niis:
        nii_path = os.path.join(input_fold, nii_file)
        img_sitk = sitk.ReadImage(nii_path)
        img_arrays = sitk.GetArrayFromImage(img_sitk)
        img_arrays = (img_arrays - img_arrays.min()) / (img_arrays.max() - img_arrays.min())
        case_name = nii_file.split('.')[0]
        print(case_name)
        for slice_idx in range(img_arrays.shape[0]):
            if not os.path.exists(save_data_root + f'/{case_name}_slice{slice_idx}.h5'):
                image = img_arrays[slice_idx, :, :]
                img_max, img_min = image.max(), image.min()
                image = (image - img_min) / (img_max - img_min)
                image = (image * 255).astype(np.uint8)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                masks = mask_generator.generate(image_rgb)
                mask_ = save_anns(masks)

                if mask_ is not None:
                    mask_ = (mask_ * 255).astype(np.uint8)
                    mask_ = mask_[:, :, :3]
                    gray_mask = convert_gray(mask_)
                    os.makedirs(save_data_root, exist_ok=True)
                    with h5py.File(save_data_root + f'/{case_name}_slice{slice_idx}.h5', 'w') as h5f:
                        h5f.create_dataset('masks', data=gray_mask)
                        h5f.create_dataset('image', data=image)


if __name__ == "__main__":
    dataset = "mms"
    input_fold = "./data/mms/train/img/full"
    save_data_root = f"./outputs/{dataset}_masks"

    os.makedirs(save_data_root, exist_ok=True)
    sam_path = "./pretrained_model/sam_vit_h.pth"
    main(input_fold, save_data_root, sam_path)
