import numpy as np
import nibabel as nib

import torch
from torch.utils.data import Dataset

class InferAllDataset(Dataset):
    def __init__(self, nii_path, patch_size=48, stride=40, pad_size=24):
        self.patch_size = patch_size
        self.stride = stride

        raw_img = np.squeeze(nib.load(nii_path).get_fdata())

        # ---- padding: ensure all regions could be extracted ----
        self.pad_size = pad_size
        pad = ((pad_size, pad_size), (pad_size, pad_size), (pad_size, pad_size))
        self.img = np.pad(raw_img, pad, mode='constant', constant_values=0)

        self.patches, self.coords = self.extract_patches(self.img)

    def extract_patches(self, volume):
        patches = []
        coords = []

        z_max, y_max, x_max = volume.shape
        ps = self.patch_size
        st = self.stride

        for z in range(0, z_max - ps + 1, st):
            for y in range(0, y_max - ps + 1, st):
                for x in range(0, x_max - ps + 1, st):

                    patch = volume[z:z+ps, y:y+ps, x:x+ps]

                    if np.any(patch):  
                        patches.append(patch)
                        coords.append((z, y, x))

        patches = np.array(patches)[:, None]

        return patches, coords

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = torch.from_numpy(self.patches[idx]).float()
        coord = torch.tensor(self.coords[idx], dtype=torch.int16)

        return patch, coord
