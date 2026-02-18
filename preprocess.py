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

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

MIN_VAL = 5
MIN_TEST = 5

SEED = 42
TARGET_PER_CLASS = 500  # balance only train

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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
# SPLIT FUNCTION (STABLE)
# =========================
def split_paths(paths):
    random.shuffle(paths)
    n = len(paths)

    # compute base values
    n_val = max(MIN_VAL, int(n * VAL_RATIO))
    n_test = max(MIN_TEST, int(n * TEST_RATIO))

    # if total exceeds n, adjust safely
    if n_val + n_test >= n:
        n_val = MIN_VAL
        n_test = MIN_TEST

    n_train = n - n_val - n_test

    # final safety adjustment
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

    for cls in classes:
        imgs = [p for p in cls.rglob("*") if p.is_file() and is_image(p)]
        if imgs:
            class_to_paths[cls.name] = imgs

    return class_to_paths


# =========================
# WRITE PREPROCESSED IMAGES
# =========================
def write_preprocessed(split_name: str, cls_name: str, img_paths):
    out_dir = OUTPUT_ROOT / split_name / cls_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in img_paths:
        try:
            img = load_rgb(p)
            img = resize_img(to_gray(img))
            save_path = out_dir / f"{p.stem}_base.jpg"
            img.save(save_path, quality=95)
        except Exception as e:
            print(f"[SKIP] {p} -> {e}")


# =========================
# BALANCE TRAIN ONLY
# =========================
def balance_train_class(train_class_dir: Path):
    base_imgs = sorted(train_class_dir.glob("*_base.jpg"))
    current = len(base_imgs)

    print(f"{train_class_dir.name}: train(real)={current}, target={TARGET_PER_CLASS}")

    if current >= TARGET_PER_CLASS:
        return

    need = TARGET_PER_CLASS - current

    for i in range(need):
        src = random.choice(base_imgs)
        base_img = Image.open(src).convert("L")
        aug = augment_gray(base_img)
        aug.save(train_class_dir / f"aug_{i+1:05d}.jpg", quality=95)


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

        print(f"{cls_name}: total={len(paths)}, "
              f"train={len(train)}, val={len(val)}, test={len(test)}")

        write_preprocessed("train", cls_name, train)
        write_preprocessed("validation", cls_name, val)
        write_preprocessed("test", cls_name, test)

    print("\n=== BALANCING TRAIN SET ===\n")

    train_root = OUTPUT_ROOT / "train"
    for cls_dir in train_root.iterdir():
        if cls_dir.is_dir():
            balance_train_class(cls_dir)

    print("\n✅ Dataset successfully prepared!")
    print("Saved at:", OUTPUT_ROOT.resolve())


if __name__ == "__main__":
    main()
