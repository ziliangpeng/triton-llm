#!/usr/bin/env python3
"""Download SmolLM2 weights from HuggingFace Hub and save as .npy files.

No PyTorch dependency — uses ``safetensors`` directly for safe tensor loading.

Usage:
    python scripts/download_smollm2_weights.py [variant] [output_dir]

    variant    — SmolLM2-135M (default), SmolLM2-360M, or SmolLM2-1.7B
    output_dir — Directory to save .npy files (default: ./weights)
"""

import json
import os
import sys

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open


def download_weights(variant: str = "SmolLM2-135M", output_dir: str = "./weights"):
    """Download SmolLM2 weights from HF and save as .npy files.

    Parameters
    ----------
    variant : str
        Model variant name (e.g. ``SmolLM2-135M``).
    output_dir : str
        Directory where ``.npy`` files will be saved.
    """
    hf_name = f"HuggingFaceTB/{variant}"
    os.makedirs(output_dir, exist_ok=True)

    # Determine shard filenames
    try:
        index_path = hf_hub_download(hf_name, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)
        shards = list(sorted(set(index["weight_map"].values())))
    except Exception:
        shards = ["model.safetensors"]

    count = 0
    for shard in shards:
        local_path = hf_hub_download(hf_name, shard)
        print(f"  Loading {shard} ...")
        with safe_open(local_path, framework="np", device="cpu") as f:
            for key in f.keys():
                arr = f.get_tensor(key)
                safe_name = key.replace(".", "_")
                np.save(os.path.join(output_dir, f"{safe_name}.npy"), arr)
                count += 1

    print(f"Downloaded {count} tensors to {output_dir}/")


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "SmolLM2-135M"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "./weights"
    print(f"Downloading {variant} weights to {out_dir}/")
    download_weights(variant, out_dir)


if __name__ == "__main__":
    main()
