"""Prepare FER2013 dataset for torchvision.

Prerequisites:
    Download fer2013.csv from Kaggle:
    https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
    Place it at ROOT_DIR/datasets/fer2013/fer2013.csv

Usage:
    python -m embedata.prepare.fer2013 --root_dir ROOT_DIR

Splits fer2013.csv into train_split.csv (Usage=Training) and
test_split.csv (Usage=PrivateTest or PublicTest), which is what
the MyFER2013 class in registry.py expects.
"""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="Project root directory")
    args = parser.parse_args()

    output_dir = Path(args.root_dir) / "datasets" / "fer2013"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = []
    test_rows = []
    header = None

    with open(output_dir / "fer2013.csv") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        for row in reader:
            if row["Usage"] == "Training":
                train_rows.append(row)
            else:
                test_rows.append(row)

    for name, rows in [("train_split.csv", train_rows), ("test_split.csv", test_rows)]:
        with open(output_dir / name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        print(f"{name}: {len(rows)} samples")

    print(f"\nFiles written to {output_dir}")
    print("Update md5sums in registry.py MyFER2013 if needed.")


if __name__ == "__main__":
    main()
