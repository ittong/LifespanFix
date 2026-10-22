import os
import argparse
from tqdm import tqdm
import numpy as np
import nibabel as nib
from scipy import ndimage

from utils.topology import enforce_genus0, largest_component, topology_summary, structure


def main():
    ap = argparse.ArgumentParser(
        description='Optional genus-0 enforcement for downstream applications (e.g., spherical mapping). '
                    'Not part of LifespanFix itself. Fast-marching topology correction of Bazin & Pham, '
                    'written with reference to https://github.com/m-qiang/CortexODE/blob/main/util/tca.py')
    ap.add_argument('--input_dir', type=str, default='./results',
                    help='Binary WM segmentations (e.g., LifespanFix outputs) ending with .nii.gz')
    ap.add_argument('--output_dir', type=str, default='./results_genus0')
    ap.add_argument('--keep_islands', action='store_true',
                    help='Do not remove WM components disconnected from the main body (they will be bridged)')
    ap.add_argument('--save_defects', action='store_true',
                    help='Also save the voxels changed by genus-0 enforcement (localized topological defects)')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for subj_id in tqdm(sorted([f for f in os.listdir(args.input_dir) if f.endswith('.nii.gz')])):
        stem = subj_id.replace('.nii.gz', '')
        img = nib.load(os.path.join(args.input_dir, subj_id))
        wm = np.squeeze(img.get_fdata()) > 0.5

        main_body = wm if args.keep_islands else largest_component(wm)
        fixed = enforce_genus0(main_body, keep_largest=False)

        before, after = topology_summary(wm), topology_summary(fixed)
        n_defects = ndimage.label(fixed ^ wm, structure=structure(26))[1]
        print(f'\n{subj_id}: {n_defects} defect components, '
              f'{int((fixed & ~wm).sum())} voxels filled, {int((wm & ~fixed).sum())} voxels removed (islands)')
        print(f'  before: {before}')
        print(f'  after : {after}')

        nib.save(nib.Nifti1Image(fixed.astype(np.int16), img.affine, img.header),
                 os.path.join(args.output_dir, f'{stem}_genus0.nii.gz'))
        if args.save_defects:
            nib.save(nib.Nifti1Image((fixed ^ wm).astype(np.int16), img.affine, img.header),
                     os.path.join(args.output_dir, f'{stem}_defect.nii.gz'))

    print('All subjects done.')


if __name__ == '__main__':
    main()
