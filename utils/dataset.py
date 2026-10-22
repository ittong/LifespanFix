import os
import glob
import numpy as np
import nibabel as nib
from scipy import ndimage

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from utils.simulation import simulate_defects


def patch_starts(length, patch_size, stride):
    """Start positions along one axis; the last patch is aligned to the end so every voxel is covered."""
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    if starts[-1] != length - patch_size:
        starts.append(length - patch_size)
    return starts


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

        for z in patch_starts(z_max, ps, st):
            for y in patch_starts(y_max, ps, st):
                for x in patch_starts(x_max, ps, st):

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


# -----------------------------
# Training
# -----------------------------
def list_subjects(data_dir, num_templates=30):
    """Each subject folder contains:
        wm.nii.gz                original (automatic) WM segmentation, I_GT
        templates/*.nii.gz       WM templates deformably registered to wm.nii.gz
        defect.nii.gz            (optional) detected defect regions; patches overlapping them are skipped
    """
    subjects = []
    for subj_dir in sorted(glob.glob(os.path.join(data_dir, '*'))):
        wm = os.path.join(subj_dir, 'wm.nii.gz')
        templates = sorted(glob.glob(os.path.join(subj_dir, 'templates', '*.nii.gz')))
        if not os.path.exists(wm):
            continue
        assert len(templates) == num_templates, \
            f'{subj_dir}: found {len(templates)} templates, expected {num_templates}'
        defect = os.path.join(subj_dir, 'defect.nii.gz')
        subjects.append({'name': os.path.basename(subj_dir), 'wm': wm, 'templates': templates,
                         'defect': defect if os.path.exists(defect) else None})
    return subjects


def load_binary(path):
    return (np.asarray(nib.load(path).dataobj) > 0.5).astype(np.uint8)


class TrainPatchDataset(IterableDataset):
    """Yields (defective patch I_D, registered template patches {T_i}, original WM patch I_GT).

    - Defects are simulated on the whole WM volume (Section 2.2), with `n_realizations` independent
      realizations per subject. The random seeds are fixed, so every epoch sees the same defects.
    - `patches_per_subject` patches are centred at simulated defect voxels, split evenly across the
      realizations, and only where the whole patch lies outside the detected defect regions.
    - A fixed proportion of subjects (`defect_free_ratio`) receives no simulated defects; for them the
      patches are centred on the WM boundary and the template inputs are replaced by copies of I_GT
      (identity mapping).
    - Each worker keeps `subjects_in_memory` subjects loaded and draws patches from them in random
      order, so that consecutive patches come from different subjects.
    """
    def __init__(self, subjects, patch_size=48, patches_per_subject=1000, defect_free_ratio=0.1,
                 subjects_in_memory=4, seed=0, n_realizations=2):
        self.subjects = subjects
        self.patch_size = patch_size
        self.patches_per_subject = patches_per_subject
        self.subjects_in_memory = subjects_in_memory
        self.seed = seed
        self.n_realizations = n_realizations
        self.epoch = 0

        rng = np.random.default_rng(seed)
        n_free = int(round(defect_free_ratio * len(subjects)))
        self.defect_free = set(rng.choice(len(subjects), n_free, replace=False).tolist())

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.subjects) * self.patches_per_subject

    def sample_centres(self, candidates, defect, n, rng):
        half = self.patch_size // 2
        valid = np.zeros_like(candidates)
        valid[half:-half, half:-half, half:-half] = True
        if defect is not None:
            # a patch centred at c must not contain any detected defect voxel
            valid &= ~ndimage.maximum_filter(defect.astype(bool), size=self.patch_size + 1)
        coords = np.argwhere(candidates & valid)
        if len(coords) == 0:
            return coords
        return coords[rng.choice(len(coords), min(n, len(coords)), replace=False)]

    def load_subject(self, i):
        subj = self.subjects[i]
        half = self.patch_size // 2
        pad = [(half, half)] * 3

        wm = load_binary(subj['wm'])
        templates = np.stack([np.pad(load_binary(t), pad) for t in subj['templates']])
        defect = np.pad(load_binary(subj['defect']), pad) if subj['defect'] else None

        # (realization, centre) pairs; realization -1 means no simulated defects
        defective, centres = [], []
        if i in self.defect_free:
            rng = np.random.default_rng([self.seed, i])
            wm_pad = np.pad(wm, pad).astype(bool)
            boundary = wm_pad & ~ndimage.binary_erosion(wm_pad)
            centres = [(-1, c) for c in self.sample_centres(boundary, defect, self.patches_per_subject, rng)]
        else:
            n = self.patches_per_subject // self.n_realizations
            for v in range(self.n_realizations):
                rng = np.random.default_rng([self.seed, i, v])
                defective_v, defect_label = simulate_defects(wm, rng)
                defective.append(np.pad(defective_v, pad))
                centres += [(v, c) for c in self.sample_centres(np.pad(defect_label > 0, pad), defect, n, rng)]

        return {'index': i, 'wm': np.pad(wm, pad), 'templates': templates,
                'defective': defective, 'centres': centres}

    def make_sample(self, subj, v, c):
        half = self.patch_size // 2
        z, y, x = c - half
        ps = self.patch_size
        gt = subj['wm'][z:z+ps, y:y+ps, x:x+ps]

        if v < 0:
            defective = gt
            templates = np.repeat(gt[None], len(subj['templates']), axis=0)
        else:
            defective = subj['defective'][v][z:z+ps, y:y+ps, x:x+ps]
            templates = subj['templates'][:, z:z+ps, y:y+ps, x:x+ps]

        return (torch.from_numpy(defective[None].astype(np.float32)),
                torch.from_numpy(templates.astype(np.float32)),
                torch.from_numpy(gt[None].astype(np.float32)))

    def __iter__(self):
        worker = get_worker_info()
        worker_id, n_workers = (worker.id, worker.num_workers) if worker else (0, 1)

        rng = np.random.default_rng([self.seed, self.epoch, worker_id])
        order = rng.permutation(len(self.subjects))[worker_id::n_workers].tolist()

        pool = []
        while order or pool:
            while order and len(pool) < self.subjects_in_memory:
                pool.append(self.load_subject(order.pop()))
            k = int(rng.integers(len(pool)))
            if not pool[k]['centres']:
                pool.pop(k)
                continue
            v, c = pool[k]['centres'].pop(int(rng.integers(len(pool[k]['centres']))))
            yield self.make_sample(pool[k], v, c)
