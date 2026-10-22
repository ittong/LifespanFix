import os
import csv
import argparse
from tqdm import tqdm
import numpy as np
import nibabel as nib
from scipy import ndimage

from utils.topology import enforce_genus0, largest_component


def dice(a, b):
    s = a.sum() + b.sum()
    return 2.0 * (a & b).sum() / s if s > 0 else 1.0


def boundary(mask):
    return mask & ~ndimage.binary_erosion(mask)


def average_surface_distance(a, b, region, spacing):
    """Symmetric average distance (mm) between the boundaries of a and b inside region."""
    ba, bb = boundary(a) & region, boundary(b) & region
    if not ba.any() or not bb.any():
        return np.nan
    dist_to_b = ndimage.distance_transform_edt(~boundary(b), sampling=spacing)
    dist_to_a = ndimage.distance_transform_edt(~boundary(a), sampling=spacing)
    return (dist_to_b[ba].sum() + dist_to_a[bb].sum()) / (ba.sum() + bb.sum())


def main():
    ap = argparse.ArgumentParser(
        description='SR, DR and ASD within manually annotated defect regions (Section 3.2). '
                    'Files are matched by identical names across the three folders.')
    ap.add_argument('--pred_dir', type=str, required=True, help='Corrected WM segmentations')
    ap.add_argument('--gt_dir', type=str, required=True, help='Manually corrected WM segmentations')
    ap.add_argument('--region_dir', type=str, required=True,
                    help='Localized defect regions; each annotated defect has its own integer label (1..K)')
    ap.add_argument('--min_agreement', type=float, default=0.7,
                    help='Minimum Dice with the ground truth within a defect region for a successful correction')
    ap.add_argument('--output_csv', type=str, default='./evaluation.csv')
    args = ap.parse_args()

    rows = []
    for name in tqdm(sorted([f for f in os.listdir(args.pred_dir) if f.endswith('.nii.gz')])):
        pred_img = nib.load(os.path.join(args.pred_dir, name))
        pred = np.squeeze(pred_img.get_fdata()) > 0.5
        gt = np.squeeze(nib.load(os.path.join(args.gt_dir, name)).get_fdata()) > 0.5
        regions = np.squeeze(nib.load(os.path.join(args.region_dir, name)).get_fdata()).astype(np.int32)
        spacing = pred_img.header.get_zooms()[:3]

        # residual topological defects = voxels that genus-0 filling would still change
        main_body = largest_component(pred)
        residual = enforce_genus0(main_body, keep_largest=False) ^ main_body

        # SR: (1) no residual defect in the region and (2) sufficient agreement with the ground truth
        labels = [k for k in np.unique(regions) if k > 0]
        n_correct = 0
        for k in labels:
            r = regions == k
            if not residual[r].any() and dice(pred[r], gt[r]) >= args.min_agreement:
                n_correct += 1

        union = regions > 0
        rows.append({
            'subject': name,
            'n_defects': len(labels),
            'SR': 100.0 * n_correct / len(labels) if labels else np.nan,
            'DR': 100.0 * dice(pred[union], gt[union]),
            'ASD': average_surface_distance(pred, gt, union, spacing),
        })
        print(rows[-1])

    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for key in ('SR', 'DR', 'ASD'):
        values = np.array([r[key] for r in rows], dtype=float)
        print(f'{key}: {np.nanmean(values):.3f} ± {np.nanstd(values):.3f}')
    print(f'Saved to {args.output_csv}')


if __name__ == '__main__':
    main()
