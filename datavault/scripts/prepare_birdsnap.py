"""Prepare Birdsnap dataset into ImageFolder train/test splits.

Prerequisites:
    1. wget https://thomasberg.org/datasets/birdsnap/1.1/birdsnap.tgz
    2. tar xzf birdsnap.tgz
    3. python get_birdsnap.py  (requires python2, downloads images)

Usage:
    python scripts/prepare_birdsnap.py --input /path/to/birdsnap/download/images --output /path/to/birdsnap
"""

import argparse
import os
import random
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def is_valid_image(path):
    try:
        img = Image.open(path)
        img.verify()
        return img.format != "GIF"
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to birdsnap/download/images")
    parser.add_argument("--output", type=str, required=True, help="Output path for train/test splits")
    parser.add_argument("--test-per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    for split in ["train", "test"]:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    species_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    print(f"Found {len(species_dirs)} species")

    for species_dir in tqdm(species_dirs, desc="Processing species"):
        species = species_dir.name
        images = [f for f in species_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        images = [f for f in images if is_valid_image(f)]

        if len(images) < args.test_per_class + 1:
            continue

        random.shuffle(images)
        test_images = images[: args.test_per_class]
        train_images = images[args.test_per_class :]

        for split, split_images in [("train", train_images), ("test", test_images)]:
            dest = output_dir / split / species
            dest.mkdir(parents=True, exist_ok=True)
            for img_path in split_images:
                shutil.copy2(img_path, dest / img_path.name)

    for split in ["train", "test"]:
        n = sum(1 for _ in (output_dir / split).rglob("*") if _.is_file())
        print(f"{split}: {n} images")


if __name__ == "__main__":
    main()
