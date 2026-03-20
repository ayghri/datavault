"""Prepare Facebook Hateful Memes dataset into ImageFolder format.

Prerequisites:
    Download from https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset
    and extract to a folder containing train.jsonl, dev.jsonl, and image files.

Usage:
    python scripts/prepare_hatefulmemes.py --input /path/to/hatefulmemes --output /path/to/hatefulmemes
"""

import argparse
import json
import os
import shutil
from pathlib import Path

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    for split_name, jsonl_file in [("train", "train.jsonl"), ("test", "dev.jsonl")]:
        jsonl_path = input_dir / jsonl_file
        if not jsonl_path.exists():
            print(f"Skipping {split_name}: {jsonl_path} not found")
            continue

        with open(jsonl_path) as f:
            entries = [json.loads(line) for line in f]

        for label in ["0", "1"]:
            (output_dir / split_name / label).mkdir(parents=True, exist_ok=True)

        count = 0
        for entry in tqdm(entries, desc=f"Processing {split_name}"):
            img_path = input_dir / entry["img"]
            label = str(entry["label"])
            if img_path.exists():
                shutil.copy2(img_path, output_dir / split_name / label / img_path.name)
                count += 1

        print(f"{split_name}: {count} images")


if __name__ == "__main__":
    main()
