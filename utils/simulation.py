"""
Simulation of topological and anatomical defects (Section 2.2 of the paper).

  (i)  random tissue removal  -> holes and exclusion defects
  (ii) spurious bridges       -> handles and inclusion defects

Both corruptions are built from 3D primitive shapes (sphere, cube, pyramid, ellipsoid) of varying
sizes that are removed from or added to a defect-free WM segmentation. Defects are simulated on the
whole volume and seeded on the WM boundary. The default parameters are the ones used in the paper.
"""
import numpy as np
from scipy import ndimage
from skimage.draw import line_nd


SHAPES = ('sphere', 'cube', 'pyramid', 'ellipsoid')


def random_shape(rng, max_radius):
    """Voxel offsets (k, 3) of a random primitive centred at the origin, with a size of 1..max_radius voxels."""
    size = int(rng.integers(1, max_radius + 1))
    shape = SHAPES[rng.integers(len(SHAPES))]
    x, y, z = np.mgrid[-size:size + 1, -size:size + 1, -size:size + 1]

    if shape == 'sphere':
        mask = x ** 2 + y ** 2 + z ** 2 <= size ** 2
    elif shape == 'cube':
        mask = np.ones_like(x, dtype=bool)
    elif shape == 'pyramid':
        mask = np.abs(x) + np.abs(y) + np.abs(z) <= size
    else:
        rx, ry, rz = rng.integers(1, size + 1, size=3)
        mask = (x / rx) ** 2 + (y / ry) ** 2 + (z / rz) ** 2 <= 1

    return np.argwhere(mask) - size


def place(offsets, centre, shape):
    """Coordinates of a shape placed at centre, clipped to the volume."""
    pts = offsets + np.asarray(centre)
    inside = np.all((pts >= 0) & (pts < np.array(shape)), axis=1)
    return pts[inside]


def boundary(wm):
    """WM boundary voxels (removed by a one-voxel erosion); defects are seeded here."""
    return wm & ~ndimage.binary_erosion(wm)


def simulate_holes(wm, candidates, rng, n_holes, max_radius, max_steps):
    """Holes / exclusion defects: from each seed, place 2..max_steps random shapes, each centred at a
    random voxel of the evolving defect region, and remove the resulting cluster from the WM."""
    wm = wm.copy()
    hole = np.zeros(wm.shape, dtype=bool)
    seeds = candidates[rng.choice(len(candidates), min(n_holes, len(candidates)), replace=False)]

    for centre in seeds:
        region = np.zeros((0, 3), dtype=np.int64)
        for _ in range(int(rng.integers(2, max_steps + 1))):
            pts = place(random_shape(rng, max_radius), centre, wm.shape)
            pts = pts[wm[tuple(pts.T)]]  # only WM voxels can be removed
            region = np.unique(np.concatenate([region, pts]), axis=0)
            if len(region) > 0:
                centre = region[rng.integers(len(region))]

        wm[tuple(region.T)] = False
        hole[tuple(region.T)] = True
    return hole


def simulate_handles(wm, candidates, rng, n_handles, max_radius=2, min_half=3, max_half=7,
                     p_shape=0.8, max_trials=100000):
    """Handles / inclusion defects: in the smallest cubic window around a seed (half-width min_half to
    max_half - 1 voxels) that contains at least two disconnected WM components, draw a line between the
    centroids of two of them and place random shapes along it. Only non-WM voxels are added."""
    handle = np.zeros(wm.shape, dtype=bool)
    structure = ndimage.generate_binary_structure(3, 1)

    generated, trials = 0, 0
    while generated < n_handles and trials < max_trials:
        trials += 1
        c = candidates[rng.integers(len(candidates))]
        for r in range(min_half, max_half):
            lo = np.maximum(c - r, 0)
            hi = np.minimum(c + r + 1, wm.shape)
            labels, n = ndimage.label(wm[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], structure=structure)
            if n < 2:
                continue

            p1 = np.argwhere(labels == 1).mean(0) + lo
            p2 = np.argwhere(labels == 2).mean(0) + lo
            for v in np.stack(line_nd(p1.astype(int), p2.astype(int), endpoint=True), axis=1):
                handle[tuple(v)] = True
                if rng.random() < p_shape:
                    handle[tuple(place(random_shape(rng, max_radius), v, wm.shape).T)] = True
            generated += 1
            break

    return handle & ~wm


def simulate_defects(wm, rng, n_holes=150, n_lower_holes=50, n_handles=200):
    """Simulate holes and handles on a defect-free WM volume.

    - n_holes holes over the whole WM boundary (shape size up to 3 voxels, 2-6 shapes each),
    - n_lower_holes larger holes in the lower half of the volume along the third axis
      (shape size up to 4 voxels, 2-7 shapes each),
    - n_handles bridges between disconnected WM components (shape size up to 2 voxels).

    Returns the defective WM and a label map of the simulated defects (1 = hole, 2 = handle).
    """
    wm = wm.astype(bool)
    candidates = boundary(wm)
    lower = candidates.copy()
    lower[:, :, wm.shape[2] // 2:] = False

    hole = simulate_holes(wm, np.argwhere(candidates), rng, n_holes, max_radius=3, max_steps=6) \
        | simulate_holes(wm, np.argwhere(lower), rng, n_lower_holes, max_radius=4, max_steps=7)
    handle = simulate_handles(wm, np.argwhere(candidates), rng, n_handles)

    defective = (wm & ~hole) | handle
    defect_label = np.zeros(wm.shape, dtype=np.uint8)
    defect_label[hole] = 1
    defect_label[handle] = 2
    return defective.astype(np.uint8), defect_label
