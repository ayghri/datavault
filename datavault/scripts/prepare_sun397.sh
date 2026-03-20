#!/bin/bash
# Download and prepare SUN397 dataset for torchvision.
# Torchvision's SUN397 has download=True but it's slow.
# This script downloads and sets up the partition files.
#
# Usage: bash scripts/prepare_sun397.sh /path/to/datasets

set -e

DEST="${1:-.}/sun397"
mkdir -p "$DEST"

if [ -d "$DEST/SUN397" ] && [ -f "$DEST/SUN397/Training_01.txt" ]; then
    echo "SUN397 already prepared at $DEST"
    exit 0
fi

echo "Downloading SUN397..."
wget -q --show-progress http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz -O "$DEST/SUN397.tar.gz" || \
    curl -L http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz -o "$DEST/SUN397.tar.gz"

echo "Extracting SUN397..."
tar xzf "$DEST/SUN397.tar.gz" -C "$DEST"
rm "$DEST/SUN397.tar.gz"

echo "Downloading partition files..."
wget -q --show-progress https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip -O "$DEST/Partitions.zip" || \
    curl -L https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip -o "$DEST/Partitions.zip"

echo "Extracting partitions..."
unzip -q -o "$DEST/Partitions.zip" -d "$DEST/SUN397/"
rm "$DEST/Partitions.zip"

echo "Done. Contents:"
ls "$DEST/SUN397/" | head -20
