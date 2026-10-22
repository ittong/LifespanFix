import numpy as np

def reconstruct_volume(patches, coords, orig_label, patch_size, pad_size=24, threshold=0.5):
    orig_shape = orig_label.shape[0] + 2 * pad_size, orig_label.shape[1] + 2 * pad_size, orig_label.shape[2] + 2 * pad_size
    recon = np.zeros(orig_shape, dtype=np.float32)
    count = np.zeros(orig_shape, dtype=np.float32)

    for p, (z, y, x) in zip(patches, coords):
        recon[z:z+patch_size, y:y+patch_size, x:x+patch_size] += p
        count[z:z+patch_size, y:y+patch_size, x:x+patch_size] += 1

    count[count == 0] = 1
    recon = recon / count
    
    orig_label = np.pad(
        orig_label,
        ((pad_size, pad_size),
         (pad_size, pad_size),
         (pad_size, pad_size)),
        mode='edge'
    )
    
    label = orig_label.copy()
    label[(orig_label == 0) & (recon > threshold)] = 1
    label[(orig_label == 1) & (recon < (1 - threshold))] = 0

    # remove padding
    oz, oy, ox = orig_shape[0] - 2 * pad_size, orig_shape[1] - 2 * pad_size, orig_shape[2] - 2 * pad_size
    recon = recon[pad_size:pad_size+oz, pad_size:pad_size+oy, pad_size:pad_size+ox]
    label = label[pad_size:pad_size+oz, pad_size:pad_size+oy, pad_size:pad_size+ox]

    return recon, label

# -----------------------------
# Losses (Eqs. 1-4)
# -----------------------------
def ce_loss(pred, target):
    """Voxel-wise cross-entropy with a (possibly soft) target; pred is a WM probability in (0, 1).
    Written explicitly so that gradients also flow through a probabilistic target (the pseudo label)."""
    return -(target * pred.log() + (1 - target) * (1 - pred).log()).mean()


def dice_loss(pred, target, eps=1e-6):
    """1 - 2<target, pred> / (|target|_1 + |pred|_1), averaged over the batch."""
    dims = tuple(range(1, pred.dim()))
    inter = (pred * target).sum(dims)
    return (1 - 2 * inter / (pred.sum(dims) + target.sum(dims) + eps)).mean()
