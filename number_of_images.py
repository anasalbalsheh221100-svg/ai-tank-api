from pathlib import Path

DATASET_ROOT = Path("split_dataset_gray_balanced")  # change if different
SPLITS = ["train", "validation", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


def main():
    if not DATASET_ROOT.exists():
        print(f"❌ Dataset folder not found: {DATASET_ROOT.resolve()}")
        return

    # get classes from train (or any split)
    classes_dir = DATASET_ROOT / "train"
    if not classes_dir.exists():
        print(f"❌ Missing split folder: {classes_dir.resolve()}")
        return

    classes = sorted([d.name for d in classes_dir.iterdir() if d.is_dir()])
    if not classes:
        print("❌ No class folders found in train/")
        return

    print(f"Dataset root: {DATASET_ROOT.resolve()}\n")
    print(f"{'Class':25} {'Train':>7} {'Val':>7} {'Test':>7} {'Total':>7}")
    print("-" * 60)

    grand = {s: 0 for s in SPLITS}

    for cls in classes:
        counts = {}
        total = 0
        for s in SPLITS:
            c = count_images(DATASET_ROOT / s / cls)
            counts[s] = c
            total += c
            grand[s] += c

        print(f"{cls:25} {counts['train']:7} {counts['validation']:7} {counts['test']:7} {total:7}")

    print("-" * 60)
    print(f"{'TOTAL':25} {grand['train']:7} {grand['validation']:7} {grand['test']:7} {(grand['train']+grand['validation']+grand['test']):7}")


if __name__ == "__main__":
    main()
