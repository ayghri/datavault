#!/bin/bash
# Download and prepare EuroSAT dataset for torchvision.datasets.EuroSAT
# Torchvision expects: {root}/eurosat/2750/{class_folders}
#
# Usage: bash scripts/download_eurosat.sh /buckets/datasets/torchvision
#        Then use: EuroSAT(root="/buckets/datasets/torchvision", download=False)

set -e

ROOT="${1:-.}"
DEST="$ROOT/eurosat"
mkdir -p "$DEST"

if [ -d "$DEST/2750" ]; then
    echo "EuroSAT already exists at $DEST/2750, skipping."
    exit 0
fi

URL="https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip"

echo "Downloading EuroSAT to $DEST ..."
wget -q --show-progress "$URL" -O "$DEST/EuroSAT.zip" || curl -L "$URL" -o "$DEST/EuroSAT.zip"

echo "Extracting..."
unzip -q "$DEST/EuroSAT.zip" -d "$DEST"
rm "$DEST/EuroSAT.zip"

# Rename to match torchvision expected layout: {root}/eurosat/2750/
if [ -d "$DEST/EuroSAT_RGB" ]; then
    mv "$DEST/EuroSAT_RGB" "$DEST/2750"
elif [ -d "$DEST/EuroSAT" ]; then
    mv "$DEST/EuroSAT" "$DEST/2750"
fi

echo "Done. Classes:"
ls "$DEST/2750/"
