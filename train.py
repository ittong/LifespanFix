import os
import argparse
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

from model.DUNet import DUNet
from utils.dataset import list_subjects, TrainPatchDataset
from utils.utils import ce_loss, dice_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=str, required=True,
                    help='One folder per subject with wm.nii.gz, templates/*.nii.gz and optional defect.nii.gz')
    ap.add_argument('--save_dir', type=str, default='./checkpoint')

    # data
    ap.add_argument('--num_templates', type=int, default=30)
    ap.add_argument('--patch_size', type=int, default=48)
    ap.add_argument('--patches_per_subject', type=int, default=1000)
    ap.add_argument('--defect_free_ratio', type=float, default=0.1,
                    help='Proportion of training subjects without simulated defects (identity mapping)')

    # optimization
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--lr_decay', type=float, default=0.5, help='Learning rate multiplier after each epoch')
    ap.add_argument('--lambda_dice', type=float, default=0.5, help='lambda in L_cor = L_ce + lambda * L_dice')
    ap.add_argument('--alpha', type=float, default=1e-4, help='alpha in L_total = L_gen + alpha * L_cor')

    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--subjects_in_memory', type=int, default=4, help='Subjects kept in memory per worker')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', type=str, default='cuda:0')
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # -----------------------------
    # Data
    # -----------------------------
    subjects = list_subjects(args.data_dir, args.num_templates)
    print(f'{len(subjects)} training subjects found in {args.data_dir}')

    dataset = TrainPatchDataset(
        subjects, args.patch_size, args.patches_per_subject, args.defect_free_ratio,
        args.subjects_in_memory, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    # -----------------------------
    # Networks: generation (templates -> pseudo label) and correction (LifespanFix)
    # -----------------------------
    device = torch.device(args.device)
    generation_net = DUNet(in_channel=args.num_templates).to(device)
    correction_net = DUNet(in_channel=1).to(device)

    params = list(generation_net.parameters()) + list(correction_net.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)

    tag = f'{round(args.defect_free_ratio * 100)}error_free_wm'

    # -----------------------------
    # Joint training
    # -----------------------------
    for epoch in range(1, args.epochs + 1):
        dataset.set_epoch(epoch)
        generation_net.train()
        correction_net.train()

        running = {'gen': 0., 'cor': 0., 'total': 0.}
        pbar = tqdm(loader, total=len(dataset) // args.batch_size, desc=f'Epoch {epoch}/{args.epochs}')
        for step, (defective, templates, gt) in enumerate(pbar, 1):
            defective, templates, gt = defective.to(device), templates.to(device), gt.to(device)

            pseudo_label = generation_net(templates)    # I'_GT
            corrected = correction_net(defective)       # I_C

            loss_gen = ce_loss(pseudo_label, gt)                                       # Eq. 1
            loss_cor = ce_loss(corrected, pseudo_label) \
                + args.lambda_dice * dice_loss(corrected, pseudo_label)                # Eqs. 2-4
            loss = loss_gen + args.alpha * loss_cor                                    # Eq. 5

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running['gen'] += loss_gen.item()
            running['cor'] += loss_cor.item()
            running['total'] += loss.item()
            pbar.set_postfix({k: f'{v / step:.4f}' for k, v in running.items()})

        scheduler.step()

        # only the correction network is needed at inference (infer.py loads this state dict directly)
        torch.save(correction_net.state_dict(), os.path.join(args.save_dir, f'correction_net_{tag}_epoch{epoch}.pth'))
        torch.save(generation_net.state_dict(), os.path.join(args.save_dir, f'generation_net_{tag}_epoch{epoch}.pth'))

    torch.save(correction_net.state_dict(), os.path.join(args.save_dir, f'correction_net_{tag}.pth'))
    print('Training done.')


if __name__ == '__main__':
    main()
