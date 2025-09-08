import os
import SimpleITK as sitk
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset
from scipy import ndimage



def winadj_mri(array):
    v0 = np.percentile(array, 1)
    v1 = np.percentile(array, 99)
    array[array < v0] = v0
    array[array > v1] = v1
    v0 = array.min()
    v1 = array.max()
    array = (array - v0) / (v1 - v0) * 2.0 - 1.0
    return array


def resize(img, lab):
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=0)
    if len(lab.shape) == 2:
        lab = np.expand_dims(lab, axis=0)
    D, H, W = img.shape
    zoom = [1, 256/H, 256/W]
    img = ndimage.zoom(img, zoom, order=2)
    lab = ndimage.zoom(lab, zoom, order=0)
    return img, lab


def crop_depth(img, lab, phase = 'train'):
    D,H,W = img.shape
    if D > 10:
        if phase == 'train':
            target_ssh = np.random.randint(0, int(D-10), 1)[0]
            zero_img = img[target_ssh:target_ssh+10, :, :]
            zero_lab = lab[target_ssh:target_ssh+10, :, :]
        elif phase == 'valid':
            zero_img, zero_lab = img, lab
    else:
        zero_img = np.zeros((10, H, W))
        zero_lab = np.zeros((10, H, W))
        zero_img[0:D, :, :] = img
        zero_lab[0:D, :, :] = lab

    return zero_img, zero_lab


class niiDataset(Dataset):
    def __init__(self, source_img, source_lab, dataset, phase='test'):
        self.dataset = dataset
        self.source_img = source_img
        self.source_lab = source_lab
        self.phase = phase
        nii_names = os.listdir(source_img)
        self.all_files = []
        for nii_name in nii_names:
            self.img_path = os.path.join(self.source_img, nii_name)
            if self.dataset == 'fb':
                self.lab_path = os.path.join(self.source_lab, nii_name[:-7] + '_segmentation.nii.gz')
            elif self.dataset == 'mms':
                self.lab_path = os.path.join(self.source_lab, nii_name[:-7] + '_gt.nii.gz')
            else:
                print(self.dataset)
                raise Exception('Unrecognized dataset.')

            self.nii_name = str(nii_name)
            self.all_files.append({
                "img": self.img_path,
                "lab": self.lab_path,
                "img_name": self.nii_name
            })

    def __getitem__(self, index):
        fname = self.all_files[index]
        img_obj = sitk.ReadImage(fname["img"])
        A = sitk.GetArrayFromImage(img_obj)
        lab_obj = sitk.ReadImage(fname["lab"])
        A_gt = sitk.GetArrayFromImage(lab_obj)
        if self.phase == 'train' and self.dataset == 'mms':
            A, A_gt = crop_depth(A, A_gt)
        A = winadj_mri(A)
        A, A_gt = resize(A, A_gt)
        A = torch.from_numpy(A.copy()).unsqueeze(0).float()
        A_gt = torch.from_numpy(A_gt.copy()).unsqueeze(0).float()

        return A, A_gt, fname["img_name"], fname["lab"]

    def __len__(self):
        return len(self.all_files)



class Pretrain_Datasets(Dataset):
    def __init__(
            self,
            base_dir=None,
            datasets=None,
            transform=None
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.transform = transform
        self.datasets = datasets

        for dataset in self.datasets:
            with open(self._base_dir + '/' + dataset + "/train_slices.list", "r") as f1:
                dataset_sample_list = f1.readlines()
            dataset_sample_list = [dataset + '/data/' + item.replace("\n", "") for item in dataset_sample_list]
            self.sample_list.extend(dataset_sample_list)
        print("total {} samples from {}".format(len(self.sample_list), self.datasets))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if not case.endswith('.h5'):
            case = case + '.h5'
        h5f = h5py.File(self._base_dir + '/' + "{}".format(case), "r")
        case_name = os.path.splitext(os.path.basename(case))[0]
        label = h5f["image"][:]
        mask = h5f["masks"][:]
        if len(label.shape) != 3:
            label = np.expand_dims(label, axis=0)
        if self.transform == None:
            label = torch.from_numpy(label.astype(np.float32))
            mask_gt = torch.from_numpy(mask.astype(np.float32))
            sample = {"label": label, "mask_gt": mask_gt, "name": case_name}
        else:
            label = torch.from_numpy(label.astype(np.float32))
            mask_gt = torch.from_numpy(mask.astype(np.float32))
            sample = {"label": label, "mask_gt": mask_gt, "name": case_name}
            sample = self.transform(sample)

        sample["idx"] = idx

        return sample
