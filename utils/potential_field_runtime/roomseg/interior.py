"""Interior free-space classification (paper section 3.1)."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow


def remove_clutter(busy):
    """Free busy voxels matching the vertical busy-free-BUSY-free-busy pattern."""
    below = np.maximum.accumulate(busy, axis=2)  # busy anywhere at z' <= z
    above = np.maximum.accumulate(busy[:, :, ::-1], axis=2)[:, :, ::-1]
    freed = np.zeros_like(busy)
    freed[:, :, 2:-2] = (busy[:, :, 2:-2] & ~busy[:, :, 1:-3] & ~busy[:, :, 3:-1]
                         & below[:, :, :-4] & above[:, :, 4:])
    return busy & ~freed


def dominant_directions(normals):
    """Two dominant horizontal axes from wall normals (axial mean folded mod 90deg)."""
    n = normals[np.abs(normals[:, 2]) < 0.3]
    theta = np.arctan2(n[:, 1], n[:, 0])
    a = np.angle(np.exp(4j * theta).mean()) / 4
    return np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])


def _blocked(busy, idx, d, chunk=4096):
    """True per voxel in idx if a busy voxel lies along horizontal direction d."""
    nx, ny, _ = busy.shape
    steps = 0.5 * np.arange(1, 2 * max(nx, ny) + 1)
    out = np.zeros(len(idx), bool)
    for s in range(0, len(idx), chunk):
        p = idx[s:s + chunk, None, :2] + 0.5 + steps[None, :, None] * d  # (m, K, 2)
        q = np.floor(p).astype(np.int64)
        inb = (q >= 0).all(-1) & (q[..., 0] < nx) & (q[..., 1] < ny)
        q[~inb] = 0
        out[s:s + chunk] = (busy[q[..., 0], q[..., 1], idx[s:s + chunk, None, 2]] & inb).any(1)
    return out


def _blocked_volume(busy, d):
    """The same sampled horizontal rays, evaluated with Boolean volume shifts.

    Translating the ray offsets avoids materializing a (free voxels × samples
    × XY) coordinate array. Rare floating-point half-cell ties are handled by
    splitting the starting-axis ranges, preserving ``idx + .5 + step * d``
    exactly rather than assuming all integer translations round identically.
    """
    busy = np.asarray(busy, dtype=bool)
    d = np.asarray(d, dtype=float)
    if busy.ndim != 3 or d.shape != (2,) or not np.isfinite(d).all():
        raise ValueError("Expected a 3D occupancy grid and finite XY ray direction")
    out = np.zeros_like(busy)
    if not busy.size:
        return out
    nx, ny, _ = busy.shape
    # Exact coordinate axes have simple prefix/suffix rays. Positive rays
    # begin at the next cell; negative rays include the starting cell, exactly
    # as the original half-voxel sampling (irrelevant at free query voxels).
    for axis in range(2):
        other = 1 - axis
        if d[other] == 0 and abs(d[axis]) == 1:
            if d[axis] < 0:
                return np.maximum.accumulate(busy, axis=axis)
            suffix = np.flip(np.maximum.accumulate(np.flip(busy, axis=axis), axis=axis), axis=axis)
            lo = tuple(slice(None, -1) if index == axis else slice(None) for index in range(3))
            hi = tuple(slice(1, None) if index == axis else slice(None) for index in range(3))
            out[lo] = suffix[hi]
            return out

    ix, iy = np.arange(nx), np.arange(ny)
    start_x, start_y = ix + .5, iy + .5
    offsets = set()
    exceptional = set()

    def runs(values):
        boundaries = np.r_[0, np.flatnonzero(np.diff(values)) + 1, len(values)]
        return [(int(a), int(b), int(values[a])) for a, b in zip(boundaries[:-1], boundaries[1:])]

    for step in .5 * np.arange(1, 2 * max(nx, ny) + 1):
        dx = np.floor(start_x + step * d[0]).astype(np.int64) - ix
        dy = np.floor(start_y + step * d[1]).astype(np.int64) - iy
        if np.all(dx == dx[0]) and np.all(dy == dy[0]):
            offsets.add((int(dx[0]), int(dy[0])))
        else:
            for x0, x1, ox in runs(dx):
                for y0, y1, oy in runs(dy):
                    exceptional.add((x0, x1, y0, y1, ox, oy))

    def shifted_or(x0, x1, y0, y1, dx, dy):
        x0, x1 = max(x0, -dx), min(x1, nx - dx)
        y0, y1 = max(y0, -dy), min(y1, ny - dy)
        if x0 < x1 and y0 < y1:
            target = out[x0:x1, y0:y1]
            np.logical_or(target, busy[x0 + dx:x1 + dx, y0 + dy:y1 + dy], out=target)

    for dx, dy in sorted(offsets):
        shifted_or(0, nx, 0, ny, dx, dy)
    for x0, x1, y0, y1, dx, dy in sorted(exceptional):
        if (dx, dy) not in offsets:
            shifted_or(x0, x1, y0, y1, dx, dy)
    return out


def evidence(busy, doms):
    """Interior evidence E(v) per Eq. (1); zero outside free space."""
    cum = np.maximum.accumulate(busy, axis=2)
    rcm = np.maximum.accumulate(busy[:, :, ::-1], axis=2)[:, :, ::-1]
    below = np.zeros_like(busy)
    above = np.zeros_like(busy)
    below[:, :, 1:] = cum[:, :, :-1]  # busy strictly below
    above[:, :, :-1] = rcm[:, :, 1:]  # busy strictly above
    # Weight boolean terms separately so joint enclosure contributes an extra term.
    E = 0.43 * below + 0.1425 * above + 0.1425 * (above & below)
    for d in doms:
        E += 0.1425 * (_blocked_volume(busy, d) & _blocked_volume(busy, -d))
    E[busy] = 0.0
    return E


def segment_interior(E, busy, smoothness=0.6, scale=1000):
    """MRF over free voxels solved by min-cut; returns bool interior mask."""
    free = ~busy
    n = int(free.sum())
    node = np.full(busy.shape, -1, np.int64)
    node[free] = np.arange(2, n + 2)  # 0 = source (interior), 1 = sink (exterior)
    e = np.rint(E[free] * scale).astype(np.int32)

    R = [np.zeros(n, np.int64), node[free]]
    C = [node[free], np.ones(n, np.int64)]
    V = [e, scale - e]
    w = np.int32(round(smoothness * scale))
    for ax in range(3):
        lo = tuple(slice(None, -1) if i == ax else slice(None) for i in range(3))
        hi = tuple(slice(1, None) if i == ax else slice(None) for i in range(3))
        a, b = node[lo], node[hi]
        m = (a >= 2) & (b >= 2)
        a, b = a[m], b[m]
        R += [a, b]
        C += [b, a]
        V += [np.full(a.size, w), np.full(b.size, w)]

    g = coo_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))),
                   shape=(n + 2, n + 2), dtype=np.int32).tocsr()
    residual = g - maximum_flow(g, 0, 1).flow
    residual.data[residual.data < 0] = 0
    residual.eliminate_zeros()
    reachable = breadth_first_order(residual, 0, return_predecessors=False)

    on_source_side = np.zeros(n + 2, bool)
    on_source_side[reachable] = True
    interior = np.zeros_like(busy)
    interior[free] = on_source_side[node[free]]
    return interior
