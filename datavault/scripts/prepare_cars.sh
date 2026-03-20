#!/bin/bash
# Prepare Stanford Cars dataset for torchvision.datasets.StanfordCars
#
# Requires manual Kaggle download:
#   1. Download from https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset
#   2. Place this script's --input to the extracted folder
#
# Usage: bash scripts/prepare_cars.sh /path/to/kaggle_download /path/to/output

set -e

INPUT="${1:?Usage: prepare_cars.sh INPUT_DIR OUTPUT_DIR}"
OUTPUT="${2:?Usage: prepare_cars.sh INPUT_DIR OUTPUT_DIR}"

mkdir -p "$OUTPUT/stanford_cars"

echo "Downloading devkit..."
wget -q --show-progress https://github.com/pytorch/vision/files/11644847/car_devkit.tgz -O "$OUTPUT/car_devkit.tgz" || \
    curl -L https://github.com/pytorch/vision/files/11644847/car_devkit.tgz -o "$OUTPUT/car_devkit.tgz"
tar xzf "$OUTPUT/car_devkit.tgz" -C "$OUTPUT/stanford_cars/"
rm "$OUTPUT/car_devkit.tgz"

echo "Downloading test annotations..."
wget -q --show-progress "https://raw.githubusercontent.com/nguyentruonglau/stanford-cars/main/labeldata/cars_test_annos_withlabels.mat" \
    -O "$OUTPUT/stanford_cars/cars_test_annos_withlabels.mat" || \
    curl -L "https://raw.githubusercontent.com/nguyentruonglau/stanford-cars/main/labeldata/cars_test_annos_withlabels.mat" \
    -o "$OUTPUT/stanford_cars/cars_test_annos_withlabels.mat"

echo "Copying images from Kaggle download..."
cp -r "$INPUT/cars_train" "$OUTPUT/stanford_cars/" 2>/dev/null || true
cp -r "$INPUT/cars_test" "$OUTPUT/stanford_cars/" 2>/dev/null || true

echo "Done. Structure:"
ls "$OUTPUT/stanford_cars/"
