"""Prepare Facebook Hateful Memes dataset into ImageFolder format.

Prerequisites:
    Download from https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset
    and extract to ROOT_DIR/datasets/hatefulmemes/ (should contain train.jsonl, dev.jsonl, and image files).

Usage:
    python -m datavault.prepare.hatefulmemes --root_dir ROOT_DIR
"""

import argparse
import json
import os
import shutil
from pathlib import Path

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="Project root directory")
    args = parser.parse_args()

    data_path = Path(args.root_dir) / "datasets" / "hatefulmemes"
    input_dir = data_path
    output_dir = data_path

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
