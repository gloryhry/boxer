#!/usr/bin/env python3
"""Check that the local PyTorch build can actually execute CUDA kernels."""

import json

import torch
import torch.nn.functional as F


def main():
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is unavailable. Run nvidia-smi in a local terminal and ensure "
            "this process has access to /dev/nvidia*."
        )

    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    with torch.inference_mode():
        matrix = torch.ones((256, 256), device=device, dtype=dtype)
        product = matrix @ matrix
        torch.testing.assert_close(product.float(), torch.full_like(product.float(), 256))
        query = torch.randn((1, 4, 64, 64), device=device, dtype=dtype)
        attention = F.scaled_dot_product_attention(query, query, query)
        assert attention.is_cuda and torch.isfinite(attention).all().item()
        image = torch.randn((1, 3, 32, 32), device=device, dtype=dtype)
        kernel = torch.randn((8, 3, 3, 3), device=device, dtype=dtype)
        convolution = F.conv2d(image, kernel)
        assert convolution.is_cuda and torch.isfinite(convolution).all().item()
    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device),
                "compute_capability": list(torch.cuda.get_device_capability(device)),
                "precision": str(dtype),
                "vram_free_mib": free_bytes // 2**20,
                "vram_total_mib": total_bytes // 2**20,
                "cuda_matmul_attention_conv2d": "PASS",
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
