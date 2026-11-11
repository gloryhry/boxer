"""Anisotropic potential field and voxel visibility (paper section 3.2)."""
import numpy as np
from scipy.spatial import cKDTree


def potential_field(grid, interior):
    """L2 distance from each interior voxel to the nearest busy voxel in the
    upper half-space (z' >= z), in meters."""
    busy_idx = np.argwhere(grid.busy)
    order = np.argsort(busy_idx[:, 2])[::-1]  # descending z
    busy_idx = busy_idx[order]
    pf = np.zeros(grid.shape)
    for z in range(grid.shape[2]):
        cells = np.argwhere(interior[:, :, z])
        upper = busy_idx[busy_idx[:, 2] >= z]
        if cells.size and upper.size:
            q = np.column_stack([cells, np.full(len(cells), z)])
            d, _ = cKDTree(upper).query(q)
            pf[cells[:, 0], cells[:, 1], z] = d * grid.voxel
    return pf


def pf_map(pf, interior):
    """Per-stack max PF (2D) and index of the highest interior voxel (-1 = none)."""
    nz = interior.shape[2]
    top_z = np.where(interior.any(2), nz - 1 - np.argmax(interior[:, :, ::-1], 2), -1)
    return pf.max(axis=2), top_z


def height_consistent_map(grid, interior, pf):
    """Aggregate and sample one supported storey instead of mixing roof layers.

    The occupancy and original interior classification remain untouched. A
    supported column uses only PF values in the main structural height band,
    and its visibility point is the interior voxel nearest a shared mid-band
    height. Columns on another landing, or with no nearby support, keep the
    original PF/stack-top pair. This does not classify or delete light fixtures.
    """
    from .refine import _structure_footprint

    if interior.shape != grid.busy.shape or pf.shape != grid.busy.shape:
        raise ValueError("Potential field, interior and occupancy shapes must match")
    values, sample_z = pf_map(pf, interior)
    active = sample_z >= 0
    active_count = int(active.sum())
    diagnostics = dict(mode="legacy_fallback", active_columns=active_count,
                       supported_columns=0, fallback_columns=active_count,
                       band_interior_columns=0, far_reference_fallback_columns=0,
                       changed_z_columns=0, excluded_plenum_columns=0,
                       excluded_plenum_voxels=0, excluded_plenum_maxima=0,
                       local_max_delta_m=0., mean_pf_delta_m=0.,
                       reference_z=None, reference_world_z=None, band_voxels=None,
                       max_reference_offset_m=float(max(.4, 1.5 * grid.voxel)))
    if not active_count:
        return values, sample_z, diagnostics
    _, _, band = _structure_footprint(grid, interior)
    if band is None:
        return values, sample_z, diagnostics

    start, stop = map(int, band)
    reference = int(round((start + stop - 1) / 2))
    levels = np.arange(start, stop)
    in_band = interior[:, :, start:stop]
    offsets = np.where(in_band, np.abs(levels - reference), interior.shape[2] + 1)
    nearest = offsets.argmin(axis=2)
    closest_offset = np.take_along_axis(offsets, nearest[:, :, None], axis=2)[:, :, 0]
    band_supported = in_band.any(axis=2)
    supported = band_supported & (closest_offset * grid.voxel <= diagnostics["max_reference_offset_m"] + 1e-12)
    diagnostics.update(band_voxels=[start, stop], reference_z=reference,
                       reference_world_z=float(grid.origin[2] + (reference + .5) * grid.voxel),
                       band_interior_columns=int(band_supported.sum()),
                       far_reference_fallback_columns=int(np.count_nonzero(band_supported & ~supported)))
    if not supported.any():
        return values, sample_z, diagnostics

    legacy_values, legacy_z = values, sample_z
    values, sample_z = legacy_values.copy(), legacy_z.copy()
    band_values = np.where(in_band, pf[:, :, start:stop], 0.).max(axis=2)
    values[supported] = band_values[supported]
    sample_z[supported] = nearest[supported] + start
    delta = legacy_values[supported] - values[supported]
    diagnostics.update(
        mode="height_consistent", supported_columns=int(supported.sum()),
        fallback_columns=int(np.count_nonzero(active & ~supported)),
        changed_z_columns=int(np.count_nonzero(sample_z != legacy_z)),
        excluded_plenum_columns=int(np.count_nonzero(supported & (legacy_z >= stop))),
        excluded_plenum_voxels=int(interior[supported][:, stop:].sum()),
        excluded_plenum_maxima=int(np.count_nonzero(supported & (pf.argmax(axis=2) >= stop))),
        local_max_delta_m=float(delta.max()), mean_pf_delta_m=float(delta.mean()))
    return values, sample_z, diagnostics


def relative_height_map(grid, interior, pf, heights, *, fallback=None):
    """Aggregate each trusted column inside its own observed storey.

    This changes neither coordinates nor occupancy. Unsupported columns keep
    the original PF/top pair; local floor estimates are never extrapolated.
    """
    values, sample_z = pf_map(pf, interior)
    if fallback is not None:
        values, sample_z = map(np.asarray, fallback)
        if values.shape != grid.shape[:2] or sample_z.shape != grid.shape[:2]:
            raise ValueError("Fallback PF samples must match XY occupancy")
    floor = np.asarray(heights["floor_z"], float)
    ceiling = np.asarray(heights["ceiling_z"], float)
    valid = np.asarray(heights["valid"], bool)
    confidence = np.asarray(heights.get("confidence", np.zeros(floor.shape)), float)
    if (floor.shape != grid.shape[:2] or ceiling.shape != floor.shape or valid.shape != floor.shape
            or confidence.shape != floor.shape
            or interior.shape != grid.shape or pf.shape != grid.shape):
        raise ValueError("Local heights and PF must match the occupancy grid")
    trusted = (valid & np.isfinite(floor) & np.isfinite(ceiling) & np.isfinite(confidence)
               & (confidence >= .5) & (confidence <= 1.))
    levels = np.arange(grid.shape[2])[None, None, :]
    band = ((levels > floor[:, :, None] + .25 / grid.voxel)
            & (levels < ceiling[:, :, None] - .25 / grid.voxel))
    supported = band & interior & trusted[:, :, None]
    # Use the middle of the local usable storey, bounded away from both
    # surfaces; exact supported sample layers are audited by multiview.
    reference = (floor + ceiling) / 2
    offsets = np.where(supported, np.abs(levels - reference[:, :, None]), np.inf)
    nearest = offsets.argmin(2)
    nearest_offset = np.take_along_axis(offsets, nearest[:, :, None], axis=2)[:, :, 0]
    use = trusted & (nearest_offset * grid.voxel <= max(.4, 1.5 * grid.voxel) + 1e-12)
    legacy_z = sample_z.copy()
    values = values.copy()
    sample_z = sample_z.copy()
    values[use] = np.where(supported, pf, 0.).max(2)[use]
    sample_z[use] = nearest[use]
    info = dict(enabled=True, mode="local_floor_relative", active_columns=int((legacy_z >= 0).sum()),
                supported_columns=int(use.sum()), fallback_columns=int(((legacy_z >= 0) & ~use).sum()),
                changed_z_columns=int((sample_z != legacy_z).sum()),
                relative_surface_margin_m=.25, max_reference_offset_m=float(max(.4, 1.5 * grid.voxel)),
                coordinate_rule="original physical voxel coordinates; no floor flattening",
                fallback_rule="previous supported height samples" if fallback is not None else "original highest interior")
    return values, sample_z, info


def visibility(grid, top_z, cells, targets=None, step=0.5, chunk=8, *,
               heights=None, interior=None, protected_walls=None, audit=None, fallback_z=None):
    """Mutual visibility between the highest interior voxels of the given stacks.

    Returns bool (S, T): True where the segment between voxel centers crosses
    no busy voxel, sampled every `step` voxels. targets selects a column
    subset of cells (default: all)."""
    if heights is not None:
        from .multiview import visibility_multilevel
        result, diagnostics, layers = visibility_multilevel(
            grid, interior, cells, heights, protected_walls, targets, step=step, chunk=chunk, fallback_z=fallback_z)
        if audit is not None:
            audit.update(diagnostics=diagnostics, sample_z=layers)
        return result
    p = (np.column_stack([cells, top_z[tuple(cells.T)]]) + 0.5).astype(np.float32)
    q = p if targets is None else p[targets]
    k = int(np.ceil(np.linalg.norm(grid.shape) / step))
    t = ((np.arange(k, dtype=np.float32) + 0.5) / k)[None, None, :, None]
    vis = np.empty((len(p), len(q)), bool)
    for s in range(0, len(p), chunk):
        a = p[s:s + chunk, None, None, :]
        smp = (a + t * (q[None, :, None, :] - a)).astype(np.int64)  # (m, T, k, 3)
        vis[s:s + chunk] = ~grid.busy[smp[..., 0], smp[..., 1], smp[..., 2]].any(2)
    return vis
