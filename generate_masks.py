"""
Generate mask PNGs from the PIE-Bench mapping_file.json.

PIE-Bench provides per-image editing masks as RLE-encoded arrays inside
`mapping_file.json`. This script decodes them and saves each mask as a
512×512 grayscale PNG (white = edit region, black = background) under the
specified output directory, mirroring the PIE-Bench category structure.

Usage:
    python generate_masks.py \
        --mapping_file ./data/PIE-Bench/mapping_file.json \
        --output_dir ./data/mask
"""

import json
import os
import argparse
import numpy as np
from PIL import Image


def mask_decode(encoded_mask, image_shape=(512, 512)):
    """Decode an RLE mask from mapping_file.json into a binary array.

    The RLE format is a flat list of [start_index, run_length, ...] pairs
    over a row-flattened image of shape ``image_shape``.

    Boundary pixels are set to 1 (edit region) to avoid annotation errors
    at image edges, following the original PIE-Bench evaluation protocol.
    """
    length = image_shape[0] * image_shape[1]
    mask_array = np.zeros(length, dtype=np.uint8)

    for i in range(0, len(encoded_mask), 2):
        start = encoded_mask[i]
        run_len = encoded_mask[i + 1]
        end = min(start + run_len, length)
        if start < length:
            mask_array[start:end] = 1

    mask_array = mask_array.reshape(image_shape[0], image_shape[1])

    # Avoid annotation errors at the boundary.
    mask_array[0, :] = 1
    mask_array[-1, :] = 1
    mask_array[:, 0] = 1
    mask_array[:, -1] = 1

    return mask_array


def main():
    parser = argparse.ArgumentParser(
        description="Generate mask PNGs from PIE-Bench mapping_file.json"
    )
    parser.add_argument(
        "--mapping_file",
        type=str,
        default="./data/PIE-Bench/mapping_file.json",
        help="Path to PIE-Bench mapping_file.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/mask",
        help="Output directory for mask PNGs",
    )
    args = parser.parse_args()

    with open(args.mapping_file, "r") as f:
        annotations = json.load(f)

    print(f"Loaded {len(annotations)} entries from {args.mapping_file}")

    for image_id, item in annotations.items():
        image_path = item["image_path"]  # e.g. "0_random_140/000000000000.jpg"
        category_dir = os.path.dirname(image_path)

        mask_array = mask_decode(item["mask"])
        # Convert binary {0,1} to grayscale {0,255} for PNG storage.
        mask_array = (mask_array * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_array, mode="L")

        save_dir = os.path.join(args.output_dir, category_dir)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"{image_id}.png")
        mask_img.save(save_path)

    print(f"Generated {len(annotations)} masks in {args.output_dir}")


if __name__ == "__main__":
    main()
