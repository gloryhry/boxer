"""PCD/PLY readers and room-labeled PLY export for the potential-field worker."""

import colorsys
import struct
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


def _lzf_decompress(data, expected):
    out = bytearray()
    pos = 0
    while pos < len(data):
        ctrl = data[pos]
        pos += 1
        if ctrl < 32:
            length = ctrl + 1
            if pos + length > len(data):
                raise ValueError("PCD LZF 字面量越界")
            out.extend(data[pos:pos + length])
            pos += length
        else:
            length = ctrl >> 5
            ref = len(out) - ((ctrl & 31) << 8) - 1
            if length == 7:
                length += data[pos]
                pos += 1
            ref -= data[pos]
            pos += 1
            if ref < 0:
                raise ValueError("PCD LZF 引用越界")
            for _ in range(length + 2):
                out.append(out[ref])
                ref += 1
        if len(out) > expected:
            raise ValueError("PCD LZF 解压数据超过声明大小")
    if len(out) != expected:
        raise ValueError("PCD LZF 解压长度与声明不符")
    return bytes(out)

def read_pcd(path):
    with Path(path).open("rb") as f:
        header = {}
        for _ in range(100):
            line = f.readline()
            if not line:
                raise ValueError("PCD 缺少 DATA 头")
            parts = line.decode("ascii").strip().split()
            if not parts or parts[0].startswith("#"):
                continue
            header[parts[0].upper()] = parts[1:]
            if parts[0].upper() == "DATA":
                break
        else:
            raise ValueError("PCD 头过长")
        names = header.get("FIELDS", header.get("FIELD"))
        if not names:
            raise ValueError("PCD 缺少 FIELDS")
        sizes = list(map(int, header["SIZE"]))
        types = header["TYPE"]
        counts = list(map(int, header.get("COUNT", ["1"] * len(names))))
        n = int(header.get("POINTS", [str(int(header["WIDTH"][0]) * int(header.get("HEIGHT", ["1"])[0]))])[0])
        if n <= 0 or not (len(names) == len(sizes) == len(types) == len(counts)):
            raise ValueError("PCD 字段定义或点数无效")
        fields = [(name, np.dtype(f"<{dict(F='f', I='i', U='u')[typ]}{size}"), count)
                  for name, size, typ, count in zip(names, sizes, types, counts)]
        mode = header["DATA"][0].lower()
        result = {}
        if mode == "ascii":
            values = np.loadtxt(f, ndmin=2)
            if values.shape != (n, sum(counts)):
                raise ValueError("PCD ASCII 行数/字段数与头不一致")
            offset = 0
            for name, dtype, count in fields:
                result[name] = values[:, offset:offset + count].astype(dtype).squeeze(axis=1) if count == 1 else values[:, offset:offset + count].astype(dtype)
                offset += count
        elif mode == "binary":
            dtype = np.dtype([(name, dtype) if count == 1 else (name, dtype, (count,)) for name, dtype, count in fields])
            values = np.frombuffer(f.read(), dtype=dtype)
            if len(values) != n:
                raise ValueError("PCD 二进制点数与头不一致")
            result = {name: values[name] for name in names}
        elif mode == "binary_compressed":
            packed, expanded = struct.unpack("<II", f.read(8))
            expected = n * sum(size * count for size, count in zip(sizes, counts))
            if expected > 2 * 1024**3:
                raise ValueError("PCD 解压数据超过 2 GB，请先分割或下采样")
            if expanded != expected:
                raise ValueError("PCD 压缩头大小与字段定义不一致")
            compressed = f.read(packed)
            if len(compressed) != packed:
                raise ValueError("PCD 压缩数据不完整")
            raw = _lzf_decompress(compressed, expanded)
            offset = 0
            for name, dtype, count in fields:
                result[name] = np.frombuffer(raw, dtype=dtype, count=n * count, offset=offset).reshape(n, count)
                if count == 1:
                    result[name] = result[name][:, 0]
                offset += n * count * dtype.itemsize
        else:
            raise ValueError(f"不支持的 PCD 编码：{mode}")
        return result


def read_cloud(path):
    path = Path(path)
    if path.suffix.lower() == ".pcd":
        fields = read_pcd(path)
    elif path.suffix.lower() == ".ply":
        points = PlyData.read(str(path))["vertex"].data
        fields = {name: points[name] for name in points.dtype.names}
    else:
        raise ValueError("只支持 PCD 和 PLY 点云")
    if not all(name in fields for name in ("x", "y", "z")):
        raise ValueError("点云必须包含 x、y、z 字段")
    xyz = np.column_stack([fields[name] for name in ("x", "y", "z")]).astype(np.float64)
    normals = None
    for names in [("nx", "ny", "nz"), ("normal_x", "normal_y", "normal_z")]:
        if all(name in fields for name in names):
            normals = np.column_stack([fields[name] for name in names]).astype(np.float64)
            break
    valid = np.isfinite(xyz).all(axis=1)
    if valid.sum() < 3:
        raise ValueError("有效点不足 3 个")
    return xyz[valid], None if normals is None else normals[valid]


def estimate_normals(xyz, k=24, log=print):
    log(f"估计法向量：{len(xyz):,} 个点，邻域 k={k}")
    k = min(k, len(xyz))
    tree = cKDTree(xyz)
    normals = np.empty_like(xyz)
    for start in range(0, len(xyz), 8192):
        block = xyz[start:start + 8192]
        indices = tree.query(block, k=k, workers=-1)[1]
        neighbors = xyz[indices]
        centered = neighbors - neighbors.mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", centered, centered) / k
        normals[start:start + len(block)] = np.linalg.eigh(covariance)[1][:, :, 0]
    return normals


def room_color(label):
    palette = [(86, 190, 164), (239, 169, 101), (124, 155, 233), (220, 135, 164),
               (170, 157, 221), (208, 194, 107), (100, 188, 205), (184, 202, 143)]
    if label < 0:
        return (134, 147, 158)
    if label < len(palette):
        return palette[label]
    return tuple(round(255 * v) for v in colorsys.hsv_to_rgb((label * .61803398875) % 1, .48, .9))


def save_colored_ply(path, xyz, labels, normals=None):
    dtype = [(name, "f4") for name in ("x", "y", "z")]
    if normals is not None:
        dtype += [(name, "f4") for name in ("nx", "ny", "nz")]
    dtype += [(name, "u1") for name in ("red", "green", "blue")] + [("label", "i4")]
    vertices = np.empty(len(xyz), dtype=dtype)
    for i, name in enumerate(("x", "y", "z")):
        vertices[name] = xyz[:, i]
    if normals is not None:
        for i, name in enumerate(("nx", "ny", "nz")):
            vertices[name] = normals[:, i]
    for label in np.unique(labels):
        keep = labels == label
        for name, value in zip(("red", "green", "blue"), room_color(int(label))):
            vertices[name][keep] = value
    vertices["label"] = labels
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(path))
