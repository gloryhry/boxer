"""Unfused polynomial ATen dispatch for immutable CUDA history observations.

The current thread disables executor optimizations while running these programs;
other model threads keep their settings. Every eager ATen operation and the 28
float32 inverse iterations are retained. CPU/float64/other models use the reference.
"""

import math

import torch


@torch.jit.script
def radial(theta: torch.Tensor, k: torch.Tensor, zero: torch.Tensor):
    result = zero + k[..., 0] * (theta**2)
    for i in range(1, 6):
        result = result + k[..., i] * (theta ** (i + 2))
    return theta + result


@torch.jit.script
def unit(x: torch.Tensor):
    return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-30)


@torch.jit.script
def project(points: torch.Tensor, data: torch.Tensor, zero: torch.Tensor):
    x, y, z = points.unbind(-1)
    p = data.unsqueeze(-2)
    k = p[..., 29:35]
    norm = torch.linalg.vector_norm(points, dim=-1)
    valid = torch.isfinite(points).all(-1) & (norm > 1e-12) & (z > 0)
    distorted = k[..., 0].abs() > 1e-7
    radius = torch.hypot(x, y)
    theta = torch.atan2(radius, z)
    scale = radial(theta, k, zero) / radius.clamp_min(1e-30)
    nx, ny = x * scale, y * scale
    valid &= theta <= p[..., 35]
    nx, ny = torch.where(distorted, nx, x / z), torch.where(distorted, ny, y / z)
    cross = torch.where(distorted, p[..., 23], 0.0)
    uv = torch.stack(
        (
            p[..., 2] * nx + cross * ny + p[..., 4],
            p[..., 24] * nx + p[..., 3] * ny + p[..., 5],
        ),
        -1,
    )
    return uv, valid & torch.isfinite(uv).all(-1)


@torch.jit.script
def unproject(pixels: torch.Tensor, data: torch.Tensor, zero: torch.Tensor):
    p = data.unsqueeze(-2)
    k = p[..., 29:35]
    du, dv = pixels[..., 0] - p[..., 4], pixels[..., 1] - p[..., 5]
    cross = torch.where(k[..., 0].abs() > 1e-7, p[..., 23], 0.0)
    det = p[..., 2] * p[..., 3] - cross * p[..., 24]
    x = (p[..., 3] * du - cross * dv) / det
    y = (p[..., 2] * dv - p[..., 24] * du) / det
    valid = torch.isfinite(pixels).all(-1)
    r2 = x * x + y * y
    radius = torch.sqrt(r2)
    upper = torch.minimum(p[..., 35], torch.full_like(radius, math.pi / 2 - 1e-6))
    lo, hi = torch.zeros_like(radius), upper.expand_as(radius)
    radial_valid = radius <= radial(upper, k, zero)
    for _ in range(28):
        mid = (lo + hi) / 2
        below = radial(mid, k, zero) < radius
        lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
    theta = (lo + hi) / 2
    scale = torch.sin(theta) / radius.clamp_min(1e-30)
    ray = torch.stack((x * scale, y * scale, torch.cos(theta)), -1)
    distorted = k[..., 0].abs() > 1e-7
    ray = torch.where(
        distorted[..., None], ray, unit(torch.stack((x, y, torch.ones_like(x)), -1))
    )
    radial_valid = radial_valid | ~distorted
    valid &= radial_valid
    uv, ok = project(ray, data, zero)
    valid &= ok & (torch.linalg.vector_norm(uv - pixels, dim=-1) < 2e-3)
    return ray, valid & torch.isfinite(ray).all(-1) & (ray[..., 2] > 0)


_zeros = {}


def _zero(reference):
    key = reference.device, reference.dtype
    if key not in _zeros:
        # Default blocking CPU->CUDA copy makes this immutable scalar ready
        # before publishing it for a later job's tracking stream.
        _zeros[key] = torch.tensor(0.0, device=reference.device, dtype=reference.dtype)
    return _zeros[key]


def project_polynomial(points, data):
    with torch.jit.optimized_execution(False):
        return project(points, data, _zero(points))


def unproject_polynomial(pixels, data):
    with torch.jit.optimized_execution(False):
        return unproject(pixels, data, _zero(pixels))
