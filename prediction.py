# import json
# import numpy as np
# from pathlib import Path
# import tensorflow as tf
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from PIL import Image, ImageDraw
# import matplotlib.pyplot as plt
# from pillow_heif import register_heif_opener
# register_heif_opener()
#
#
# # =========================
# # PROJECT PATHS
# # =========================
# PROJECT_ROOT = Path(__file__).resolve().parent
# MODELS_DIR = PROJECT_ROOT / "models"
# RESULTS_DIR = PROJECT_ROOT / "results"
# DATASET_ROOT = PROJECT_ROOT / "split_dataset_gray_balanced"
#
# IMG_SIZE = (224, 224)
# IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
#
# # Visualization defaults
# HIGHLIGHT_CONF_THRESH = 0.80
# GRID_CONF_THRESH = 0.70
# GRID_SIZE = 10
# WINDOW_DIV = 4
#
#
# # =========================
# # FIND LATEST BEST MODEL
# # =========================
# def find_latest_best_model(models_dir: Path) -> Path:
#     # Prefer timestamped best models
#     candidates = sorted(models_dir.glob("improved_gray_balanced_best_*.keras"))
#     if candidates:
#         return candidates[-1]
#
#     # Fallback: old naming style
#     fallback = models_dir / "improved_gray_balanced_best.keras"
#     if fallback.exists():
#         return fallback
#
#     raise FileNotFoundError(
#         f"No best model found in: {models_dir}\n"
#         f"Expected: improved_gray_balanced_best_YYYYMMDD_HHMMSS.keras"
#     )
#
#
# # =========================
# # FIND METRICS.JSON
# # =========================
# def find_metrics_json(results_dir: Path) -> Path:
#     # Prefer your current fixed path
#     fixed = results_dir / "improved_gray_balanced" / "metrics.json"
#     if fixed.exists():
#         return fixed
#
#     # Otherwise search any metrics.json inside results/*
#     candidates = sorted(results_dir.rglob("metrics.json"))
#     if candidates:
#         return candidates[-1]
#
#     raise FileNotFoundError(f"metrics.json not found inside: {results_dir}")
#
#
# # =========================
# # LOAD CLASS NAMES
# # =========================
# def load_class_names() -> list:
#     metrics_path = find_metrics_json(RESULTS_DIR)
#     data = json.loads(metrics_path.read_text(encoding="utf-8"))
#
#     if "class_names" in data and isinstance(data["class_names"], list) and len(data["class_names"]) > 0:
#         return data["class_names"]
#
#     # fallback: dataset folders (train)
#     train_root = DATASET_ROOT / "train"
#     if train_root.exists():
#         return sorted([p.name for p in train_root.iterdir() if p.is_dir()])
#
#     raise FileNotFoundError("Cannot load class names (metrics.json missing class_names and dataset missing).")
#
#
# # =========================
# # PREPROCESS (MATCH TRAINING)
# # =========================
# def preprocess_any_image_to_model_input(file) -> np.ndarray:
#
#     img = Image.open(file)
#     img = img.convert("L").convert("RGB")
#     img = img.resize(IMG_SIZE)
#
#     x = np.array(img).astype(np.float32)
#     x = preprocess_input(x)
#     x = np.expand_dims(x, axis=0)
#
#     return x
#
#
# def preprocess_pil_crop_to_model_input(crop: Image.Image) -> np.ndarray:
#     crop = crop.convert("L").convert("RGB").resize(IMG_SIZE)
#     x = np.array(crop).astype(np.float32)
#     x = preprocess_input(x)
#     x = np.expand_dims(x, axis=0)
#     return x
#
#
# # =========================
# # PREDICT (TOP-K)
# # =========================
# def predict_single(model, class_names, image_path: str, top_k: int = 3):
#     image_path = Path(image_path)
#     if not image_path.exists():
#         raise FileNotFoundError(f"Image not found: {image_path}")
#
#     x = preprocess_any_image_to_model_input(image_path)
#     probs = model.predict(x, verbose=0)[0]
#
#     pred_idx = int(np.argmax(probs))
#     pred_class = class_names[pred_idx]
#     pred_conf = float(probs[pred_idx])
#
#     top_idx = np.argsort(probs)[::-1][:top_k]
#
#     print("\n=== Prediction ===")
#     print("Image:", image_path.name)
#     print(f"Predicted class: {pred_class}")
#     print(f"Confidence: {pred_conf:.4f}")
#
#     print(f"\nTop-{top_k}:")
#     for i in top_idx:
#         print(f"  - {class_names[i]}: {float(probs[i]):.4f}")
#
#     return pred_idx, pred_conf, probs
#
#
# # =========================
# # VISUALIZATION 1: HIGHLIGHT (SLIDING WINDOW)
# # =========================
# def highlight_regions(model, class_names, image_path: str,
#                       conf_thresh: float = HIGHLIGHT_CONF_THRESH,
#                       window_div: int = WINDOW_DIV,
#                       stride_ratio: float = 0.5):
#     image_path = Path(image_path)
#     original = Image.open(image_path).convert("RGB")
#     w, h = original.size
#
#     full_probs = model.predict(preprocess_any_image_to_model_input(image_path), verbose=0)[0]
#     main_class = int(np.argmax(full_probs))
#     main_conf = float(full_probs[main_class])
#     print(f"\nMain prediction: {class_names[main_class]} ({main_conf:.2%})")
#
#     overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
#     draw = ImageDraw.Draw(overlay)
#
#     window = max(32, min(w, h) // window_div)
#     stride = max(16, int(window * stride_ratio))
#
#     regions = 0
#     for y in range(0, max(1, h - window + 1), stride):
#         for x in range(0, max(1, w - window + 1), stride):
#             crop = original.crop((x, y, x + window, y + window))
#             crop_x = preprocess_pil_crop_to_model_input(crop)
#             probs = model.predict(crop_x, verbose=0)[0]
#
#             c = int(np.argmax(probs))
#             conf = float(probs[c])
#
#             if c == main_class and conf >= conf_thresh:
#                 regions += 1
#                 draw.rectangle([x, y, x + window, y + window],
#                                outline=(255, 0, 0, 220), width=3)
#                 draw.rectangle([x, y, x + window, y + window],
#                                fill=(255, 0, 0, 70))
#
#     result = Image.alpha_composite(original.convert("RGBA"), overlay)
#
#     plt.figure(figsize=(10, 6))
#     plt.imshow(result)
#     plt.title(f"Highlighted regions for {class_names[main_class]} | regions={regions}")
#     plt.axis("off")
#     plt.tight_layout()
#     out_path = PROJECT_ROOT / "object_highlighted.png"
#     plt.savefig(out_path, dpi=250)
#     plt.show()
#
#     print("Saved:", out_path)
#     return out_path
#
#
# # =========================
# # VISUALIZATION 2: GRID OVERLAY
# # =========================
# def grid_overlay(model, class_names, image_path: str,
#                  grid_size: int = GRID_SIZE,
#                  conf_thresh: float = GRID_CONF_THRESH):
#     image_path = Path(image_path)
#     original = Image.open(image_path).convert("RGB")
#     w, h = original.size
#
#     cell_w = max(1, w // grid_size)
#     cell_h = max(1, h // grid_size)
#
#     overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
#     draw = ImageDraw.Draw(overlay)
#
#     palette = [
#         (255, 0, 0),
#         (0, 120, 255),
#         (160, 0, 255),
#         (0, 200, 0),
#         (0, 150, 80),
#         (255, 165, 0),
#         (0, 0, 0)
#     ]
#
#     for r in range(grid_size):
#         for c in range(grid_size):
#             x1, y1 = c * cell_w, r * cell_h
#             x2 = w if c == grid_size - 1 else x1 + cell_w
#             y2 = h if r == grid_size - 1 else y1 + cell_h
#
#             cell = original.crop((x1, y1, x2, y2))
#             cell_x = preprocess_pil_crop_to_model_input(cell)
#             probs = model.predict(cell_x, verbose=0)[0]
#
#             cls = int(np.argmax(probs))
#             conf = float(probs[cls])
#
#             if conf >= conf_thresh:
#                 col = palette[cls % len(palette)]
#                 draw.rectangle([x1, y1, x2, y2], fill=col + (110,))
#
#     result = Image.alpha_composite(original.convert("RGBA"), overlay)
#
#     plt.figure(figsize=(10, 6))
#     plt.imshow(result)
#     plt.title(f"Grid overlay | grid={grid_size} | thresh={conf_thresh}")
#     plt.axis("off")
#     plt.tight_layout()
#     out_path = PROJECT_ROOT / "object_segmentation.png"
#     plt.savefig(out_path, dpi=250)
#     plt.show()
#
#     print("Saved:", out_path)
#     return out_path
#
#
# # =========================
# # BATCH PREDICTION (FOLDER)
# # =========================
# def predict_folder(model, class_names, folder_path: str, top_k: int = 3):
#     folder = Path(folder_path)
#     if not folder.exists():
#         raise FileNotFoundError(f"Folder not found: {folder}")
#
#     images = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
#     print(f"\nFound {len(images)} images in: {folder}")
#
#     for img_path in images:
#         predict_single(model, class_names, str(img_path), top_k=top_k)
#
#
# # =========================
# # MAIN
# # =========================
# if __name__ == "__main__":
#     model_path = find_latest_best_model(MODELS_DIR)
#     metrics_path = find_metrics_json(RESULTS_DIR)
#
#     print("Loading model:", model_path)
#     print("Using metrics:", metrics_path)
#
#     model = tf.keras.models.load_model(model_path)
#     class_names = load_class_names()
#
#     print("\nSelect mode:")
#     print("1) Predict one image")
#     print("2) Predict folder (batch)")
#     mode = input("Enter 1/2: ").strip()
#
#     if mode == "2":
#         folder = input("Enter folder path: ").strip().strip('"')
#         predict_folder(model, class_names, folder, top_k=3)
#     else:
#         img_path = input("Enter image path: ").strip().strip('"')
#         predict_single(model, class_names, img_path, top_k=3)
#
#         do_vis = input("Visualize highlight + grid? (y/n): ").strip().lower()
#         if do_vis == "y":
#             highlight_regions(model, class_names, img_path)
#             grid_overlay(model, class_names, img_path)
