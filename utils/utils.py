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