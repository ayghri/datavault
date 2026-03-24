"""Prepare UCF101 dataset: extract middle frames from videos into ImageFolder format.

Prerequisites:
    pip install av pyunpack patool

Usage:
    python -m embedata.prepare.ucf101 --root_dir ROOT_DIR [--download]

Downloads/expects raw data at ROOT_DIR/datasets/ucf101/.
Writes frames to ROOT_DIR/datasets/ucf101/{train,val}/.
"""

import argparse
import os
import shutil
import urllib.request
from pathlib import Path

from tqdm import tqdm


def download_and_extract(input_dir):
    input_dir = Path(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    video_url = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
    splits_url = "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"

    for url, name in [(video_url, "UCF101.rar"), (splits_url, "splits.zip")]:
        dest = input_dir / name
        if not dest.exists():
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(url, dest)

    from pyunpack import Archive

    rar_path = input_dir / "UCF101.rar"
    if not (input_dir / "UCF-101").exists():
        print("Extracting UCF101.rar...")
        Archive(str(rar_path)).extractall(str(input_dir))

    import zipfile

    zip_path = input_dir / "splits.zip"
    if not (input_dir / "ucfTrainTestlist").exists():
        print("Extracting splits...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(input_dir)


def extract_middle_frame(video_path):
    import av

    try:
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        total = stream.frames
        target = total // 2

        for i, frame in enumerate(container.decode(video=0)):
            if i == target:
                return frame.to_image()

        # Fallback: return last frame
        container.seek(0)
        last = None
        for frame in container.decode(video=0):
            last = frame
        return last.to_image() if last else None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="Project root directory")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.root_dir) / "datasets" / "ucf101"
    input_dir = data_path
    output_dir = data_path

    if args.download:
        download_and_extract(input_dir)

    # Read split files
    splits_dir = input_dir / "ucfTrainTestlist"
    splits = {}
    for split_name, filename in [("train", "trainlist01.txt"), ("val", "testlist01.txt")]:
        split_file = splits_dir / filename
        videos = []
        with open(split_file) as f:
            for line in f:
                videos.append(line.strip().split()[0])
        splits[split_name] = videos

    video_root = input_dir / "UCF-101"

    for split_name, video_list in splits.items():
        print(f"\nProcessing {split_name} ({len(video_list)} videos)...")
        for video_rel in tqdm(video_list):
            video_path = video_root / video_rel
            if not video_path.exists():
                continue

            action_class = video_rel.split("/")[0]
            frame_name = video_path.stem + ".jpg"

            dest_dir = output_dir / split_name / action_class
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / frame_name

            if dest_path.exists():
                continue

            frame = extract_middle_frame(video_path)
            if frame is not None:
                frame.save(dest_path)

    for split in ["train", "val"]:
        n = sum(1 for _ in (output_dir / split).rglob("*.jpg"))
        print(f"{split}: {n} images")


if __name__ == "__main__":
    main()
