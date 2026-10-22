"""
Genus-0 topology enforcement for binary WM segmentations.

This code is written with reference to the topology correction released with CortexODE [3]
(util/tca.py in https://github.com/m-qiang/CortexODE), which re-implements the fast-marching
topology correction of Bazin & Pham [1, 2] in Python/Numba. Please cite [1, 2] (and [3]) if you use it.
We follow its "fill" propagation and differ in two points: simple points are tested directly with the
topological numbers of Bertrand & Malandain [4] (6-connected WM, 26-connected background) instead of
the look-up table from Nighres, and it runs on a binary mask using the signed distance to the WM boundary.

The background is grown from the border of the volume towards the WM in the order of decreasing signed
distance. A voxel joins the background only if it is a simple point, so the background keeps the
topology of a spherical shell and the remaining object is a single component without handles or
cavities (genus 0). Every residual handle or hole is therefore resolved by FILLING; WM is never removed.
Surfaces extracted from the output with classic marching cubes (Lorensen) are genus 0.

[1] Bazin & Pham, MICCAI 2005.  [2] Bazin & Pham, Comput. Methods Programs Biomed. 2007.
[3] Ma et al., IEEE TMI 2023.   [4] Bertrand & Malandain, Pattern Recognit. Lett. 1994.
"""
import heapq
import numpy as np
from numba import njit
from scipy import ndimage


# -----------------------------
# 3x3x3 neighbourhood tables (flat index 9*i + 3*j + k, centre = 13)
# -----------------------------
def _neighbourhood_tables():
    offsets = np.array([(i, j, k) for i in range(3) for j in range(3) for k in range(3)]) - 1
    dist = np.abs(offsets).sum(1)

    n6 = dist == 1                      # 6-neighbours of the centre
    n18_star = (dist >= 1) & (dist <= 2)  # N18 without the centre
    n26_star = dist >= 1                # N26 without the centre

    nbrs6 = -np.ones((27, 26), dtype=np.int64)
    nbrs26 = -np.ones((27, 26), dtype=np.int64)
    n_nbrs6 = np.zeros(27, dtype=np.int64)
    n_nbrs26 = np.zeros(27, dtype=np.int64)
    for a in range(27):
        for b in range(27):
            d = np.abs(offsets[a] - offsets[b])
            if a == b or d.max() > 1:
                continue
            nbrs26[a, n_nbrs26[a]] = b
            n_nbrs26[a] += 1
            if d.sum() == 1:
                nbrs6[a, n_nbrs6[a]] = b
                n_nbrs6[a] += 1
    return n6, n18_star, n26_star, nbrs6, n_nbrs6, nbrs26, n_nbrs26


N6, N18_STAR, N26_STAR, NBRS6, N_NBRS6, NBRS26, N_NBRS26 = _neighbourhood_tables()
C6 = np.array([(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)], dtype=np.int64)


@njit(cache=True)
def _count_components(member, allowed, nbrs, n_nbrs, need_n6, n6):
    """Number of connected components of (member & allowed) in a 3x3x3 cube.
    If need_n6, only components containing a 6-neighbour of the centre are counted."""
    visited = np.zeros(27, dtype=np.bool_)
    stack = np.empty(27, dtype=np.int64)
    count = 0
    for s in range(27):
        if visited[s] or not (member[s] and allowed[s]):
            continue
        visited[s] = True
        top = 0
        stack[0] = s
        touches_n6 = n6[s]
        while top >= 0:
            p = stack[top]
            top -= 1
            for t in range(n_nbrs[p]):
                q = nbrs[p, t]
                if not visited[q] and member[q] and allowed[q]:
                    visited[q] = True
                    top += 1
                    stack[top] = q
                    if n6[q]:
                        touches_n6 = True
        if touches_n6 or not need_n6:
            count += 1
    return count


@njit(cache=True)
def is_simple(cube, wm_conn, n6, n18_star, n26_star, nbrs6, n_nbrs6, nbrs26, n_nbrs26):
    """Whether the centre voxel can be added to the background set `cube` (flat bool[27])
    without changing the topology of the background or of the WM [4].
    wm_conn = 6: WM is 6-connected and background 26-connected; wm_conn = 26: the reverse."""
    rest = ~cube
    if wm_conn == 6:
        t_bg = _count_components(cube, n26_star, nbrs26, n_nbrs26, False, n6)
        t_wm = _count_components(rest, n18_star, nbrs6, n_nbrs6, True, n6)
    else:
        t_bg = _count_components(cube, n18_star, nbrs6, n_nbrs6, True, n6)
        t_wm = _count_components(rest, n26_star, nbrs26, n_nbrs26, False, n6)
    return t_bg == 1 and t_wm == 1


@njit(cache=True)
def _fast_marching_fill(levelset, processed, wm_conn, c6, n6, n18_star, n26_star,
                        nbrs6, n_nbrs6, nbrs26, n_nbrs26):
    """Grow `processed` (background) from high to low level-set values through simple points only.
    Returns the corrected level set; voxels never reached are set to the last (lowest) value."""
    min_step = 1e-5
    nx, ny, nz = levelset.shape
    corrected = levelset.copy()
    in_heap = np.zeros(levelset.shape, dtype=np.bool_)

    # max-heap on the level set (stored negated)
    heap = [(0.0, 0, 0, 0)]
    heap.pop()
    for x in range(1, nx - 1):
        for y in range(1, ny - 1):
            for z in range(1, nz - 1):
                if processed[x, y, z]:
                    continue
                for d in range(6):
                    if processed[x + c6[d, 0], y + c6[d, 1], z + c6[d, 2]]:
                        heap.append((-levelset[x, y, z], x, y, z))
                        in_heap[x, y, z] = True
                        break
    heapq.heapify(heap)

    last = np.inf
    while len(heap) > 0:
        neg_val, x, y, z = heapq.heappop(heap)
        in_heap[x, y, z] = False
        if processed[x, y, z]:
            continue
        cube = processed[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2].flatten()
        if not is_simple(cube, wm_conn, n6, n18_star, n26_star, nbrs6, n_nbrs6, nbrs26, n_nbrs26):
            continue  # may be pushed again once its neighbourhood changes

        val = -neg_val
        corrected[x, y, z] = val
        processed[x, y, z] = True
        last = val

        for d in range(6):
            xn, yn, zn = x + c6[d, 0], y + c6[d, 1], z + c6[d, 2]
            if xn < 1 or yn < 1 or zn < 1 or xn >= nx - 1 or yn >= ny - 1 or zn >= nz - 1:
                continue
            if not processed[xn, yn, zn] and not in_heap[xn, yn, zn]:
                # values are non-increasing along the propagation, which keeps every level set topologically consistent
                heapq.heappush(heap, (-min(levelset[xn, yn, zn], val - min_step), xn, yn, zn))
                in_heap[xn, yn, zn] = True

    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                if not processed[x, y, z]:
                    corrected[x, y, z] = min(corrected[x, y, z], last)
    return corrected


# -----------------------------
# Public helpers
# -----------------------------
def structure(conn):
    """scipy structuring element for 6- or 26-connectivity."""
    return ndimage.generate_binary_structure(3, 1 if conn == 6 else 3)


def largest_component(wm, conn=6):
    labels, n = ndimage.label(wm, structure=structure(conn))
    if n <= 1:
        return wm.astype(bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == sizes.argmax()


def topology_summary(wm, conn=6):
    """Betti numbers of a binary object: components (b0), handles (b1), cavities (b2), and genus.
    The genus is reported for a single component without cavities (genus 0 <=> b0 = 1, b1 = b2 = 0)."""
    from skimage.measure import euler_number

    wm = wm.astype(bool)
    bg_conn = 26 if conn == 6 else 6
    b0 = ndimage.label(wm, structure=structure(conn))[1]
    # background components minus the outer one = cavities
    b2 = ndimage.label(np.pad(~wm, 1, constant_values=True), structure=structure(bg_conn))[1] - 1
    chi = euler_number(wm, connectivity=1 if conn == 6 else 3)
    b1 = b0 + b2 - chi
    return {'components': int(b0), 'handles': int(b1), 'cavities': int(b2), 'euler': int(chi),
            'genus': int(b1) if (b0 == 1 and b2 == 0) else None}


def enforce_genus0(wm, conn=6, keep_largest=True, margin=3):
    """Enforce genus-0 topology on a binary WM segmentation by filling residual handles and holes.

    Args:
        wm: binary 3D array (1 = WM).
        conn: WM connectivity, 6 (background 26) or 26 (background 6).
        keep_largest: keep only the largest WM component first. Otherwise every isolated
            component would be bridged to the main body to make the object connected.
        margin: background margin (voxels) kept around the WM bounding box.
    Returns:
        corrected binary WM (same shape as `wm`).
    """
    assert conn in (6, 26), 'conn must be 6 or 26'
    wm = wm.astype(bool)
    if keep_largest:
        wm = largest_component(wm, conn)
    if not wm.any():
        return wm

    # crop to the WM bounding box plus a margin; the outer layer is the initial background shell
    idx = np.argwhere(wm)
    lo = idx.min(0)
    hi = idx.max(0) + 1
    pad_lo = np.maximum(margin - lo, 0)
    pad_hi = np.maximum(hi + margin - np.array(wm.shape), 0)
    padded = np.pad(wm, list(zip(pad_lo, pad_hi)))
    lo, hi = lo + pad_lo - margin, hi + pad_lo + margin
    crop = padded[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

    # signed distance: negative inside WM, positive outside
    levelset = (ndimage.distance_transform_edt(~crop) - ndimage.distance_transform_edt(crop)).astype(np.float64)
    processed = np.zeros(crop.shape, dtype=np.bool_)
    processed[[0, -1], :, :] = True
    processed[:, [0, -1], :] = True
    processed[:, :, [0, -1]] = True

    corrected = _fast_marching_fill(levelset, processed, conn, C6, N6, N18_STAR, N26_STAR,
                                    NBRS6, N_NBRS6, NBRS26, N_NBRS26)
    fixed_crop = corrected < 0

    out = np.zeros_like(padded)
    out[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = fixed_crop
    sx, sy, sz = [slice(p, p + s) for p, s in zip(pad_lo, wm.shape)]
    return out[sx, sy, sz]
