import os
import argparse
from tqdm import tqdm
import numpy as np
import nibabel as nib

import torch
from torch.utils.data import DataLoader

from model.DUNet import DUNet
from utils.dataset import InferAllDataset
from utils.utils import reconstruct_volume


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str, default="checkpoint/correction_net_10error_free_wm.pth")
    ap.add_argument('--input_dir', type=str, default='./test_example',
                    help='All files must be binarized masks and end with .nii.gz')
    ap.add_argument('--output_dir', type=str, default='./results')

    ap.add_argument('--iters', type=int, default=3)
    ap.add_argument('--patch_size', type=int, default=48)
    ap.add_argument('--stride', type=int, default=40)
    ap.add_argument('--threshold', type=float, default=0.5,
                    help='Confidence threshold for correction; higher values allow less modifications.')

    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--device', type=str, default='cuda:0')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    correction_net = DUNet().to(device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    correction_net.load_state_dict(state_dict)
    correction_net.eval()

    # -----------------------------
    # Loop over subjects
    # -----------------------------
    for subj_id in tqdm(sorted([f for f in os.listdir(args.input_dir) if f.endswith('.nii.gz')])):
        stem = subj_id.replace('.nii.gz', '')
        print(f"\n>>> Processing {subj_id} <<<")

        for it in range(1, args.iters + 1):
            print(f'=== Iteration {it}/{args.iters} for {subj_id} ===')

            if it == 1:
                dataset = InferAllDataset(
                    os.path.join(args.input_dir, subj_id),
                    args.patch_size, args.stride)
            else:
                dataset = InferAllDataset(
                    os.path.join(args.output_dir, f"{stem}_iter_tmp_probability.nii.gz"),
                    args.patch_size, args.stride)

            loader = DataLoader(dataset, batch_size=args.batch_size)

            pred_patches = []
            coords_list = []

            with torch.no_grad():
                for batch, coords in loader:
                    batch = batch.to(device)
                    pred = correction_net(batch)
                    pred = pred.squeeze(1).cpu().numpy()

                    pred_patches.append(pred)
                    coords_list.extend(coords)

            pred_patches = np.concatenate(pred_patches, axis=0)

            if it == 1:
                orig_img = nib.load(os.path.join(args.input_dir, subj_id))
            else:
                orig_img = nib.load(os.path.join(args.output_dir, f"{stem}_iter_tmp_label.nii.gz"))

            orig_label = orig_img.get_fdata().astype(np.int16)
            recon_volume, recon_label = reconstruct_volume(
                pred_patches, coords_list, orig_label, args.patch_size, threshold=args.threshold)

            # tmp saves
            nib.save(nib.Nifti1Image(recon_label.astype(np.int16), orig_img.affine, orig_img.header),
                     os.path.join(args.output_dir, f"{stem}_iter_tmp_label.nii.gz"))

            nib.save(nib.Nifti1Image(recon_volume.astype(np.float32), orig_img.affine, orig_img.header),
                     os.path.join(args.output_dir, f"{stem}_iter_tmp_probability.nii.gz"))

            # final save
            nib.save(nib.Nifti1Image(recon_label.astype(np.int16), orig_img.affine, orig_img.header),
                     os.path.join(args.output_dir, f"{stem}_iter{it}_label_0error_free_wm.nii.gz"))

            # delete tmp at last iter
            if it == args.iters:
                tmp_label = os.path.join(args.output_dir, f"{stem}_iter_tmp_label.nii.gz")
                tmp_prob  = os.path.join(args.output_dir, f"{stem}_iter_tmp_probability.nii.gz")

                if os.path.exists(tmp_label):
                    os.remove(tmp_label)
                    print(f"Deleted: {tmp_label}")
                if os.path.exists(tmp_prob):
                    os.remove(tmp_prob)
                    print(f"Deleted: {tmp_prob}")

    print('All subjects done.')


if __name__ == '__main__':
    print("\033[31mNote: Input files must be binarized wm masks (0/1 values) and end with '.nii.gz'.\033[0m")
    main()
