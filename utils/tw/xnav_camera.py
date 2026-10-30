"""Exact Torch geometry for x-nav cameras (no calibration fitting).

CameraTW's legacy 22-value prefix is followed by a self-contained 22-value
extension: model, cross intrinsics, xi/lambda/alpha/beta, six coefficients,
invertible angular limit, content rectangle and source-to-image pixel affine.
The prefix intrinsics and rectangle always describe the current image.
"""

import contextlib
import math

import torch

MODEL_NAMES = (
    "Pinhole",
    "PolynomialCamera",
    "EquidistantCamera",
    "OmniRadtanCamera",
    "EucmCamera",
    "TripleSphereCamera",
)
NATIVE_CAMERA_SIZE = 44
PREPROCESS_VERSION = 1


def pack_camera(entry):
    # Reuse the map reader's calibration/domain validation, not ROS state.
    from loaders.xnav_camera import XNavCamera

    source = XNavCamera(entry)
    p, d = source.p, source.d
    keys = {
        "Pinhole": ["k1", "k2", "p1", "p2"],
        "PolynomialCamera": [f"k{i}" for i in range(2, 8)],
        "EquidistantCamera": ["k1", "k2", "k3", "k4"],
        "OmniRadtanCamera": ["k1", "k2", "p1", "p2"],
    }.get(source.model, [])
    coefficients = [d[k] for k in keys] + [0.0] * (6 - len(keys))
    extra = [
        MODEL_NAMES.index(source.model) + 1,
        p.get("skew", p.get("b", 0.0)),
        p.get("c", 0.0),
        p.get("xi", 0.0),
        p.get("lambda", 0.0),
        p.get("alpha", 0.0),
        p.get("beta", 1.0),
        *coefficients,
        source.max_theta,
        0.0,
        0.0,
        source.width - 1.0,
        source.height - 1.0,
        1.0,
        1.0,
        0.0,
        0.0,
    ]
    return source, torch.tensor(extra, dtype=torch.float32)


def _unit(x):
    return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-30)


def _radial(theta, k, model):
    powers = range(2, 8) if model == 2 else range(3, 10, 2)
    return theta + sum(k[..., i] * theta**power for i, power in enumerate(powers))


def _distort(x, y, k, model, jacobian=False):
    k1, k2, p1, p2 = k[..., :4].unbind(-1)
    enabled = (k1.abs() > 1e-7) if model == 1 else torch.ones_like(x, dtype=torch.bool)
    raw = x * x + y * y
    r2 = raw.clamp_max(1.0) if model == 1 else raw
    radial = 1 + k1 * r2 + k2 * r2 * r2
    dx = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    dy = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    if not jacobian:
        return torch.where(enabled, dx, x), torch.where(enabled, dy, y)
    drx, dry = 2 * x, 2 * y
    if model == 1:
        drx, dry = drx * (raw < 1), dry * (raw < 1)
    rx, ry = (k1 + 2 * k2 * r2) * drx, (k1 + 2 * k2 * r2) * dry
    jxx = radial + x * rx + 2 * p1 * y + p2 * (drx + 4 * x)
    jxy = x * ry + 2 * p1 * x + p2 * dry
    jyx = y * rx + p1 * drx + 2 * p2 * y
    jyy = radial + y * ry + p1 * (dry + 4 * y) + 2 * p2 * x
    return (
        torch.where(enabled, dx, x),
        torch.where(enabled, dy, y),
        torch.where(enabled, jxx, 1.0),
        torch.where(enabled, jxy, 0.0),
        torch.where(enabled, jyx, 0.0),
        torch.where(enabled, jyy, 1.0),
    )


def _project(points, data, model):
    x, y, z = points.unbind(-1)
    p = data.unsqueeze(-2)
    k = p[..., 29:35]
    norm = torch.linalg.vector_norm(points, dim=-1)
    valid = torch.isfinite(points).all(-1) & (norm > 1e-12) & (z > 0)
    if model in (1, 2, 3):
        distorted = (k[..., 0].abs() > 1e-7) if model == 2 else None
        if model in (2, 3):
            radius = torch.hypot(x, y)
            theta = torch.atan2(radius, z)
            scale = _radial(theta, k, model) / radius.clamp_min(1e-30)
            nx, ny = x * scale, y * scale
            valid &= theta <= p[..., 35]
            if model == 2:
                nx, ny = (
                    torch.where(distorted, nx, x / z),
                    torch.where(distorted, ny, y / z),
                )
        else:
            nx, ny = _distort(x / z, y / z, k, model)
    elif model == 4:
        den = z + p[..., 25] * norm
        valid &= den.abs() > 1e-12
        nx, ny = _distort(x / den, y / den, k, model)
    elif model == 5:
        den = (
            p[..., 27] * torch.sqrt(p[..., 28] * (x * x + y * y) + z * z)
            + (1 - p[..., 27]) * z
        )
        valid &= den > 1e-12
        nx, ny = x / den, y / den
    else:
        x, y, z = x / norm, y / norm, z / norm
        z1 = z + p[..., 25]
        z2 = z1 + p[..., 26] * torch.sqrt(x * x + y * y + z1 * z1)
        den = z2 + p[..., 27] / (1 - p[..., 27]) * torch.sqrt(x * x + y * y + z2 * z2)
        valid &= den > 1e-12
        nx, ny = x / den, y / den
    cross = p[..., 23]
    if model == 2:
        cross = torch.where(distorted, cross, 0.0)
    uv = torch.stack(
        (
            p[..., 2] * nx + cross * ny + p[..., 4],
            p[..., 24] * nx + p[..., 3] * ny + p[..., 5],
        ),
        -1,
    )
    return uv, valid & torch.isfinite(uv).all(-1)


def _unproject(pixels, data, model):
    p = data.unsqueeze(-2)
    k = p[..., 29:35]
    du, dv = pixels[..., 0] - p[..., 4], pixels[..., 1] - p[..., 5]
    cross = p[..., 23]
    if model == 2:
        cross = torch.where(k[..., 0].abs() > 1e-7, cross, 0.0)
    det = p[..., 2] * p[..., 3] - cross * p[..., 24]
    x = (p[..., 3] * du - cross * dv) / det
    y = (p[..., 2] * dv - p[..., 24] * du) / det
    valid = torch.isfinite(pixels).all(-1)
    if model in (1, 4):
        target_x, target_y = x, y
        x, y = x.clone(), y.clone()
        for _ in range(20):
            fx, fy, a, b, c, d = _distort(x, y, k, model, jacobian=True)
            determinant = a * d - b * c
            safe = torch.where(determinant.abs() > 1e-20, determinant, 1.0)
            ex, ey = fx - target_x, fy - target_y
            x = x - ((d * ex - b * ey) / safe).clamp(-0.5, 0.5)
            y = y - ((a * ey - c * ex) / safe).clamp(-0.5, 0.5)
        fx, fy = _distort(x, y, k, model)
        valid &= torch.hypot(fx - target_x, fy - target_y) < 2e-6
    r2 = x * x + y * y
    if model in (2, 3):
        radius = torch.sqrt(r2)
        # This application deliberately excludes the rear hemisphere.
        upper = torch.minimum(p[..., 35], torch.full_like(radius, math.pi / 2 - 1e-6))
        lo, hi = torch.zeros_like(radius), upper.expand_as(radius)
        radial_valid = radius <= _radial(upper, k, model)
        for _ in range(40 if pixels.dtype == torch.float64 else 28):
            mid = (lo + hi) / 2
            below = _radial(mid, k, model) < radius
            lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
        theta = (lo + hi) / 2
        scale = torch.sin(theta) / radius.clamp_min(1e-30)
        ray = torch.stack((x * scale, y * scale, torch.cos(theta)), -1)
        if model == 2:
            distorted = k[..., 0].abs() > 1e-7
            ray = torch.where(
                distorted[..., None],
                ray,
                _unit(torch.stack((x, y, torch.ones_like(x)), -1)),
            )
            radial_valid = radial_valid | ~distorted
        valid &= radial_valid
    elif model == 5:
        a, b = p[..., 27], p[..., 28]
        disc = 1 - (2 * a - 1) * b * r2
        valid &= disc >= 0
        z = (1 - a * a * b * r2) / (a * torch.sqrt(disc.clamp_min(0)) + 1 - a)
        ray = _unit(torch.stack((x, y, z), -1))
    elif model == 6:
        a = p[..., 27] / (1 - p[..., 27])
        disc = 1 + (1 - a * a) * r2
        valid &= disc >= -1e-12
        s = (a + torch.sqrt(disc.clamp_min(0))) / (1 + r2)
        valid &= s > 1e-12
        qx, qy, qz = s * x, s * y, s - a
        for offset in (p[..., 26], p[..., 25]):
            disc = 1 - offset * offset * (qx * qx + qy * qy)
            valid &= disc >= -1e-12
            s = offset * qz + torch.sqrt(disc.clamp_min(0))
            valid &= s > 1e-12
            qx, qy, qz = s * qx, s * qy, s * qz - offset
        ray = _unit(torch.stack((qx, qy, qz), -1))
    elif model == 4:
        xi = p[..., 25]
        disc = 1 + (1 - xi * xi) * r2
        valid &= disc >= -1e-10
        den = xi + torch.sqrt(disc.clamp_min(0))
        valid &= den.abs() > 1e-12
        ray = _unit(torch.stack((x, y, 1 - xi * (r2 + 1) / den), -1))
    else:
        ray = _unit(torch.stack((x, y, torch.ones_like(x)), -1))
    uv, ok = _project(ray, data, model)
    tolerance = 1e-7 if pixels.dtype == torch.float64 else 2e-3
    valid &= ok & (torch.linalg.vector_norm(uv - pixels, dim=-1) < tolerance)
    return ray, valid & torch.isfinite(ray).all(-1) & (ray[..., 2] > 0)


def geometry(data, points, inverse=False):
    """Broadcast camera batches, including batches containing different models."""
    context = (
        torch.autocast(device_type=points.device.type, enabled=False)
        if points.device.type in ("cpu", "cuda")
        else contextlib.nullcontext()
    )
    with context:
        dtype = (
            torch.float64
            if points.dtype == torch.float64 or data.dtype == torch.float64
            else torch.float32
        )
        points, data = points.to(dtype), data.to(dtype)
        shape = torch.broadcast_shapes(data.shape[:-1], points.shape[:-2])
        count = points.shape[-2]
        cameras = data.expand(*shape, NATIVE_CAMERA_SIZE).reshape(
            -1, NATIVE_CAMERA_SIZE
        )
        inputs = points.expand(*shape, count, points.shape[-1]).reshape(
            len(cameras), count, points.shape[-1]
        )
        out = points.new_zeros((len(cameras), count, 3 if inverse else 2))
        valid = torch.zeros(
            (len(cameras), count), device=points.device, dtype=torch.bool
        )
        for model in cameras[:, 22].unique().tolist():
            if model not in range(1, 7):
                raise ValueError(f"Invalid native x-nav camera tag {model}")
            ids = torch.where(cameras[:, 22] == model)[0]
            selected = cameras[ids]
            result, ok = (_unproject if inverse else _project)(
                inputs[ids], selected, int(model)
            )
            if not inverse:
                bearings, invertible = _unproject(result, selected, int(model))
                ok &= invertible & (
                    torch.linalg.vector_norm(bearings - _unit(inputs[ids]), dim=-1)
                    < 1e-5
                )
            uv = inputs[ids] if inverse else result
            rect = selected[:, None, 36:40]
            size = selected[:, None, :2]
            ok &= ((uv >= rect[..., :2]) & (uv <= rect[..., 2:])).all(-1)
            ok &= ((uv >= 0) & (uv <= size - 1)).all(-1)
            out[ids], valid[ids] = torch.nan_to_num(result), ok
        return out.reshape(*shape, count, out.shape[-1]), valid.reshape(*shape, count)
