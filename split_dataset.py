"""
split_dataset.py
----------------
Reads all images from data/raw (fresh/ and rotten/ subfolders),
splits them 70% train / 15% val / 15% test using stratified sampling,
and COPIES them into data/split/{train,val,test}/{fresh,rotten}.

Run from the project root:
    python split_dataset.py
"""

import os
import shutil
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────────────────
RAW_DIR   = os.path.join("data", "raw")
SPLIT_DIR = os.path.join("data", "split")
CLASSES   = ["fresh", "rotten"]
SPLITS    = ["train", "val", "test"]
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15   # of total  →  val/(val+test) = 0.50
RANDOM_STATE = 42
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# ──────────────────────────────────────────────────────────────────────


def collect_images(class_dir: str) -> list:
    """Return sorted list of absolute image paths inside class_dir."""
    imgs = []
    for fname in sorted(os.listdir(class_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in IMG_EXTENSIONS:
            imgs.append(os.path.join(class_dir, fname))
    return imgs


def copy_files(file_list: list, dest_dir: str):
    """Copy every file in file_list to dest_dir."""
    os.makedirs(dest_dir, exist_ok=True)
    for src in file_list:
        fname = os.path.basename(src)
        dst = os.path.join(dest_dir, fname)
        # Avoid overwriting if already copied
        if not os.path.exists(dst):
            shutil.copy2(src, dst)


def main():
    print("=" * 55)
    print("  Fruit Freshness — Dataset Splitter")
    print("=" * 55)

    all_counts = {}

    for cls in CLASSES:
        src_dir = os.path.join(RAW_DIR, cls)
        if not os.path.isdir(src_dir):
            raise FileNotFoundError(
                f"[ERROR] Raw class folder not found: {src_dir}\n"
                f"  Make sure images are inside data/raw/fresh/ and data/raw/rotten/"
            )

        imgs = collect_images(src_dir)
        if len(imgs) == 0:
            raise ValueError(f"[ERROR] No images found in {src_dir}")

        print(f"\n[{cls.upper()}] Total images found: {len(imgs)}")

        # First split: train vs (val + test)
        train_imgs, valtest_imgs = train_test_split(
            imgs, test_size=(1 - TRAIN_RATIO), random_state=RANDOM_STATE
        )

        # Second split: val vs test (equal halves of the 30% remainder)
        val_imgs, test_imgs = train_test_split(
            valtest_imgs, test_size=0.50, random_state=RANDOM_STATE
        )

        splits_map = {
            "train": train_imgs,
            "val"  : val_imgs,
            "test" : test_imgs,
        }

        all_counts[cls] = {}
        for split_name, split_imgs in splits_map.items():
            dest = os.path.join(SPLIT_DIR, split_name, cls)
            copy_files(split_imgs, dest)
            count = len(split_imgs)
            all_counts[cls][split_name] = count
            print(f"  → {split_name:5s}: {count:5d} images  (copied to {dest})")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  Split Summary")
    print("=" * 55)

    total_fresh  = sum(all_counts["fresh"].values())
    total_rotten = sum(all_counts["rotten"].values())
    total_all    = total_fresh + total_rotten

    print(f"  Total fresh  images: {total_fresh}")
    print(f"  Total rotten images: {total_rotten}")
    print(f"  Grand total        : {total_all}")

    ratio = total_fresh / total_all * 100 if total_all else 0
    if abs(ratio - 50) > 10:
        print(
            f"\nWARNING: Dataset is imbalanced! "
            f"({ratio:.1f}% fresh / {100-ratio:.1f}% rotten). "
            f"Consider class_weight or oversampling."
        )
    else:
        print(f"\nClass balance looks acceptable ({ratio:.1f}% fresh).")

    print("\n  Per-split breakdown:")
    header = f"  {'Class':<8}" + "".join(f"  {s:<8}" for s in SPLITS)
    print(header)
    print("  " + "-" * (8 + 10 * len(SPLITS)))
    for cls in CLASSES:
        row = f"  {cls:<8}" + "".join(
            f"  {all_counts[cls][s]:<8}" for s in SPLITS
        )
        print(row)

    print("\nSplitting complete!\n")


if __name__ == "__main__":
    main()
