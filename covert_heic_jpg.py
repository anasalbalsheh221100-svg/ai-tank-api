from pathlib import Path
from PIL import Image
import pillow_heif

# enable HEIC support
pillow_heif.register_heif_opener()

INPUT_ROOT = Path("images")
DELETE_HEIC_AFTER_CONVERT = False

converted = 0
skipped = 0
failed = 0

for p in INPUT_ROOT.rglob("*"):
    if p.is_file() and p.suffix.lower() == ".heic":
        jpg_path = p.with_suffix(".jpg")

        if jpg_path.exists():
            print(f"[SKIP] JPG already exists: {jpg_path}")
            skipped += 1
            continue

        try:
            img = Image.open(p).convert("RGB")
            img.save(jpg_path, "JPEG", quality=95)
            print(f"[OK] {p.name} -> {jpg_path.name}")
            converted += 1

            if DELETE_HEIC_AFTER_CONVERT:
                p.unlink()

        except Exception as e:
            print(f"[FAIL] {p} -> {e}")
            failed += 1

print("\n=== DONE ===")
print("Converted:", converted)
print("Skipped:", skipped)
print("Failed:", failed)