import random
import shutil
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance, ImageChops

# =========================
# SETTINGS
# =========================
INPUT_ROOT = Path("images")  # original dataset folder
OUTPUT_ROOT = Path("split_dataset_gray_balanced")

IMG_SIZE = (224, 224)

# Better for classes with only 50-70 images
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20

MIN_VAL = 5
MIN_TEST = 5

SEED = 42
TARGET_PER_CLASS = 500  # balance only train

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic"}

# =========================
# IMAGE HELPERS
# =========================
def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def to_gray(img_rgb: Image.Image) -> Image.Image:
    return ImageOps.grayscale(img_rgb)


def resize_img(img: Image.Image) -> Image.Image:
    return img.resize(IMG_SIZE, Image.BILINEAR)


def rotate_keep_size(img: Image.Image, angle: float) -> Image.Image:
    return img.rotate(angle, resample=Image.BILINEAR, expand=False)


def translate_keep_size(img: Image.Image, max_shift_px: int, fill=0) -> Image.Image:
    dx = random.randint(-max_shift_px, max_shift_px)
    dy = random.randint(-max_shift_px, max_shift_px)

    shifted = ImageChops.offset(img, dx, dy)

    w, h = img.size
    if dx > 0:
        shifted.paste(fill, (0, 0, dx, h))
    elif dx < 0:
        shifted.paste(fill, (w + dx, 0, w, h))

    if dy > 0:
        shifted.paste(fill, (0, 0, w, dy))
    elif dy < 0:
        shifted.paste(fill, (0, h + dy, w, h))

    return shifted


def augment_gray(img_gray: Image.Image) -> Image.Image:
    out = img_gray

    if random.random() < 0.8:
        out = rotate_keep_size(out, random.uniform(-25, 25))

    if random.random() < 0.5:
        out = ImageOps.mirror(out)

    if random.random() < 0.6:
        out = ImageEnhance.Brightness(out).enhance(random.uniform(0.75, 1.25))

    if random.random() < 0.6:
        out = ImageEnhance.Contrast(out).enhance(random.uniform(0.75, 1.35))

    if random.random() < 0.5:
        out = translate_keep_size(out, max_shift_px=12, fill=0)

    return out


# =========================
# SPLIT FUNCTION
# =========================
def split_paths(paths):
    paths = paths[:]   # important: do not modify original list
    random.shuffle(paths)

    n = len(paths)

    n_val = max(MIN_VAL, int(n * VAL_RATIO))
    n_test = max(MIN_TEST, int(n * TEST_RATIO))

    if n_val + n_test >= n:
        n_val = MIN_VAL
        n_test = MIN_TEST

    n_train = n - n_val - n_test

    if n_train < 1:
        n_train = 1
        remaining = n - n_train
        n_val = remaining // 2
        n_test = remaining - n_val

    train = paths[:n_train]
    val = paths[n_train:n_train + n_val]
    test = paths[n_train + n_val:]

    return train, val, test


# =========================
# DATA COLLECTION
# =========================
def collect_class_images(input_root: Path):
    classes = sorted([d for d in input_root.iterdir() if d.is_dir()])
    class_to_paths = {}

    print("\n=== CHECKING ORIGINAL CLASS FOLDERS ===\n")

    for cls in classes:
        imgs = [p for p in cls.rglob("*") if p.is_file() and is_image(p)]
        print(f"{cls.name}: found {len(imgs)} valid images")

        if imgs:
            class_to_paths[cls.name] = imgs
        else:
            print(f"[SKIPPED] {cls.name} -> no valid supported images found")

    return class_to_paths


# =========================
# WRITE PREPROCESSED IMAGES
# =========================
def write_preprocessed(split_name: str, cls_name: str, img_paths):
    out_dir = OUTPUT_ROOT / split_name / cls_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, p in enumerate(img_paths, start=1):
        try:
            img = load_rgb(p)
            img = to_gray(img)
            img = resize_img(img)

            save_path = out_dir / f"{p.stem}_{idx:04d}_base.jpg"
            img.save(save_path, quality=95)

        except Exception as e:
            print(f"[SKIP] {p} -> {e}")


# =========================
# COUNT HELPERS
# =========================
def count_images_in_dir(folder: Path):
    if not folder.exists():
        return 0
    return len([p for p in folder.rglob("*") if p.is_file() and is_image(p)])


def count_base_images_in_dir(folder: Path):
    if not folder.exists():
        return 0
    return len(list(folder.glob("*_base.jpg")))


def count_aug_images_in_dir(folder: Path):
    if not folder.exists():
        return 0
    return len(list(folder.glob("aug_*.jpg")))


# =========================
# BALANCE TRAIN ONLY
# =========================
def balance_train_class(train_class_dir: Path):
    base_imgs = sorted(train_class_dir.glob("*_base.jpg"))
    current = len(base_imgs)

    print(f"{train_class_dir.name}: train(base)={current}, target={TARGET_PER_CLASS}")

    if current == 0:
        print(f"{train_class_dir.name}: no base images found, skipped")
        return

    if current >= TARGET_PER_CLASS:
        print(f"{train_class_dir.name}: aug_added=0, train(final)={current}")
        return

    need = TARGET_PER_CLASS - current

    for i in range(need):
        src = random.choice(base_imgs)
        base_img = Image.open(src).convert("L")
        aug = augment_gray(base_img)
        aug.save(train_class_dir / f"aug_{i+1:05d}.jpg", quality=95)

    print(f"{train_class_dir.name}: aug_added={need}, train(final)={TARGET_PER_CLASS}")


# =========================
# SUMMARY PRINTING
# =========================
def print_final_summary(class_names):
    print("\n=== FINAL COUNTS AFTER AUGMENTATION ===\n")

    total_train = 0
    total_val = 0
    total_test = 0

    for cls_name in sorted(class_names):
        train_dir = OUTPUT_ROOT / "train" / cls_name
        val_dir = OUTPUT_ROOT / "validation" / cls_name
        test_dir = OUTPUT_ROOT / "test" / cls_name

        train_base = count_base_images_in_dir(train_dir)
        train_aug = count_aug_images_in_dir(train_dir)
        train_final = count_images_in_dir(train_dir)

        val_final = count_images_in_dir(val_dir)
        test_final = count_images_in_dir(test_dir)

        total_train += train_final
        total_val += val_final
        total_test += test_final

        print(
            f"{cls_name}: "
            f"train_base={train_base}, "
            f"train_aug={train_aug}, "
            f"train_final={train_final}, "
            f"val_final={val_final}, "
            f"test_final={test_final}"
        )

    print("\n=== TOTALS ===")
    print(f"TOTAL TRAIN: {total_train}")
    print(f"TOTAL VAL:   {total_val}")
    print(f"TOTAL TEST:  {total_test}")


# =========================
# MAIN
# =========================
def main():
    random.seed(SEED)

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    class_to_paths = collect_class_images(INPUT_ROOT)

    print("\n=== SPLITTING DATASET ===\n")

    for cls_name, paths in class_to_paths.items():
        train, val, test = split_paths(paths)

        print(
            f"{cls_name}: total={len(paths)}, "
            f"train={len(train)}, val={len(val)}, test={len(test)}"
        )

        write_preprocessed("train", cls_name, train)
        write_preprocessed("validation", cls_name, val)
        write_preprocessed("test", cls_name, test)

    print("\n=== BALANCING TRAIN SET ONLY ===\n")

    train_root = OUTPUT_ROOT / "train"
    for cls_dir in train_root.iterdir():
        if cls_dir.is_dir():
            balance_train_class(cls_dir)

    print_final_summary(class_to_paths.keys())

    print("\n✅ Dataset successfully prepared!")
    print("Saved at:", OUTPUT_ROOT.resolve())


if __name__ == "__main__":
    main()