"""
Dataset Preparation Script for SmartYard
=========================================
Reads from already-downloaded local source datasets (no Roboflow API required),
filters and remaps classes to the 8 SmartYard target classes, merges all sources,
reshuffles, and writes the final dataset to data/processed/.

Target Classes (config/dataset.yaml):
  0: truck
  1: trailer
  2: container
  3: forklift
  4: person
  5: helmet
  6: safety_vest
  7: car

Source datasets (already downloaded to repo root):
  - Logistics-1/         : 20-class benchmark (Roboflow)
  - Truck-container-3/   : 2-class  (Container / Not_Container)
  - Yard-Management-System-1/ : 5-class (Container / IDs / Trailer-ID)
"""

import os
import shutil
import random
import cv2
from pathlib import Path
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────
RANDOM_SEED = 42
TARGET_CLASSES = ['truck', 'trailer', 'container', 'forklift',
                  'person', 'helmet', 'safety_vest', 'car']
IMAGE_SIZE = 640          # All images will be resized to this square
TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.20
TEST_SPLIT  = 0.10

# ── Source dataset root (relative to repo root) ────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parents[2]   # …/SmartYard/
OUTPUT_BASE  = REPO_ROOT / "data" / "processed"

# ── Per-dataset: local path + class index → target class index ─────────────────
# Key  = original class index in that dataset's data.yaml (0-based)
# Value= index in TARGET_CLASSES list above
# Classes not listed are simply dropped.

DATASET_CONFIGS = [
    {
        # ── LOGISTICS-1 (nc=20) ───────────────────────────────────────────────
        # Verified class list from Logistics-1/data.yaml:
        #  0:barcode  1:car  2:cardboard box  3:fire  4:forklift
        #  5:freight container  6:gloves  7:helmet  8:ladder
        #  9:license plate  10:person  11:qr code  12:road sign
        #  13:safety vest  14:smoke  15:traffic cone  16:traffic light
        #  17:truck  18:van  19:wood pallet
        "name": "Logistics-1",
        "path": REPO_ROOT / "Logistics-1",
        "splits": ["train", "valid", "test"],
        "index_map": {
            1:  7,   # car         → car
            4:  3,   # forklift    → forklift
            5:  2,   # freight container → container
            7:  5,   # helmet      → helmet
            10: 4,   # person      → person
            13: 6,   # safety vest → safety_vest
            17: 0,   # truck       → truck
            18: 7,   # van         → car  (van treated as car)
        }
    },
    {
        # ── TRUCK-CONTAINER-3 (nc=2) ──────────────────────────────────────────
        # Classes: 0:Container  1:Not_Container
        "name": "Truck-container-3",
        "path": REPO_ROOT / "Truck-container-3",
        "splits": ["train", "valid", "test"],
        "index_map": {
            0: 2,    # Container → container
            # 1: Not_Container → SKIP
        }
    },
    {
        # ── YARD-MANAGEMENT-SYSTEM-1 (nc=5) ───────────────────────────────────
        # Classes: 0:Container  1:Container-ID-horizontal  2:Container-ID-vertical
        #          3:Container-Logo  4:Trailer-ID
        # NOTE: "Trailer-ID" boxes mark the ID plate on a trailer chassis.
        #       We map them to the 'trailer' class as the best available proxy.
        "name": "Yard-Management-System-1",
        "path": REPO_ROOT / "Yard-Management-System-1",
        "splits": ["train", "valid", "test"],
        "index_map": {
            0: 2,    # Container  → container
            4: 1,    # Trailer-ID → trailer  ← FIXED (was "Chassis ID" which didn't exist)
        }
    },
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def prepare_output_dirs():
    """Create clean output directory structure."""
    for split in ["train", "val", "test"]:
        (OUTPUT_BASE / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_BASE / split / "labels").mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {OUTPUT_BASE}")


def collect_valid_pairs(dataset_cfg: dict) -> list:
    """
    Walk a source dataset, remap class indices, and collect
    (image_path, remapped_label_lines) tuples for images that have
    at least one annotation in our target class set.
    """
    index_map: dict = dataset_cfg["index_map"]
    ds_name: str    = dataset_cfg["name"]
    ds_path: Path   = dataset_cfg["path"]
    valid_pairs     = []

    for split in dataset_cfg["splits"]:
        img_dir = ds_path / split / "images"
        lbl_dir = ds_path / split / "labels"
        if not img_dir.exists():
            continue

        imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        print(f"  [{ds_name}] {split}: {len(imgs)} images")

        for img_path in tqdm(imgs, desc=f"  {ds_name}/{split}", leave=False):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            # Skip "_cleaned" artefacts from any previous preprocessing run
            if "_cleaned" in img_path.stem or "_cleaned" in lbl_path.stem:
                continue

            remapped_lines = []
            try:
                with open(lbl_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        src_cls = int(parts[0])
                        if src_cls in index_map:
                            parts[0] = str(index_map[src_cls])
                            remapped_lines.append(" ".join(parts))
            except Exception as e:
                print(f"    [WARN] Could not read {lbl_path}: {e}")
                continue

            if remapped_lines:          # only keep if ≥1 target annotation
                valid_pairs.append((img_path, remapped_lines))

    print(f"  [{ds_name}] -> {len(valid_pairs)} usable image-label pairs collected")
    return valid_pairs


def resize_and_save(img_path: Path, dst_path: Path):
    """Read, resize to IMAGE_SIZE×IMAGE_SIZE, and write image."""
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    if img.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    cv2.imwrite(str(dst_path), img)
    return True


def write_split(pairs: list, split_name: str):
    """Write image + label files for a given split."""
    img_out = OUTPUT_BASE / split_name / "images"
    lbl_out = OUTPUT_BASE / split_name / "labels"
    skipped = 0

    for idx, (img_path, label_lines) in enumerate(tqdm(pairs, desc=f"  Writing {split_name}")):
        # Use short zero-padded index to avoid Windows MAX_PATH issues with long Roboflow filenames
        stem = f"{split_name}_{idx:06d}"
        dst_img = img_out / f"{stem}.jpg"
        dst_lbl = lbl_out / f"{stem}.txt"

        if not resize_and_save(img_path, dst_img):
            skipped += 1
            continue
        with open(dst_lbl, "w") as f:
            f.write("\n".join(label_lines))

    print(f"  [{split_name}] {len(pairs) - skipped} written, {skipped} skipped (bad images)")


def print_class_summary(all_pairs: list):
    """Print per-class annotation counts across the full merged pool."""
    counts = {i: 0 for i in range(len(TARGET_CLASSES))}
    for _, lines in all_pairs:
        for line in lines:
            cls = int(line.split()[0])
            counts[cls] = counts.get(cls, 0) + 1
    total = sum(counts.values())
    print("\n  Class distribution in merged pool:")
    print(f"  {'Class':<5} {'Name':<15} {'Count':>8} {'%':>6}")
    print("  " + "-" * 38)
    for i, name in enumerate(TARGET_CLASSES):
        cnt = counts.get(i, 0)
        pct = (cnt / total * 100) if total else 0
        flag = "  [LOW]" if cnt < 500 else ""
        print(f"  {i:<5} {name:<15} {cnt:>8} {pct:>5.1f}%{flag}")
    print(f"  {'TOTAL':<5} {'':15} {total:>8}")


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" SmartYard Dataset Preparation Pipeline (Local Sources)")
    print("=" * 60)

    # 1. Collect valid pairs from all source datasets
    all_pairs = []
    for cfg in DATASET_CONFIGS:
        print(f"\n[STEP] Processing: {cfg['name']}")
        pairs = collect_valid_pairs(cfg)
        all_pairs.extend(pairs)

    print(f"\n[INFO] Total usable pairs across all datasets: {len(all_pairs)}")

    if not all_pairs:
        print("[ERROR] No valid pairs found. Check dataset paths.")
        return

    # 2. Print class distribution before splitting
    print_class_summary(all_pairs)

    # 3. Shuffle and split
    random.seed(RANDOM_SEED)
    random.shuffle(all_pairs)
    n = len(all_pairs)
    train_end = int(n * TRAIN_SPLIT)
    val_end   = train_end + int(n * VAL_SPLIT)

    splits = {
        "train": all_pairs[:train_end],
        "val":   all_pairs[train_end:val_end],
        "test":  all_pairs[val_end:],
    }

    print(f"\n[SPLIT] train={len(splits['train'])}  "
          f"val={len(splits['val'])}  "
          f"test={len(splits['test'])}")

    # 4. Prepare output dirs (clears old processed data)
    print("\n[STEP] Preparing output directories (clearing old data)...")
    if OUTPUT_BASE.exists():
        shutil.rmtree(OUTPUT_BASE)
    prepare_output_dirs()

    # 5. Write each split
    print("\n[STEP] Writing dataset to disk...")
    for split_name, pairs in splits.items():
        write_split(pairs, split_name)

    print("\n" + "=" * 60)
    print(f"[DONE] Dataset preparation complete!")
    print(f"    Output : {OUTPUT_BASE}")
    print(f"    Config : {REPO_ROOT / 'config' / 'dataset.yaml'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
