import io
import json
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = FastAPI(title="AI Tank Museum Guide API")

# =========================
# PATHS
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "improved_gray_balanced"

IMG_SIZE = (224, 224)

# =========================
# LOAD MODEL
# =========================
def find_latest_best_model():
    candidates = sorted(MODELS_DIR.glob("improved_gray_balanced_best_*.keras"))
    if not candidates:
        raise FileNotFoundError("No trained model found in models folder.")
    return candidates[-1]

model_path = find_latest_best_model()
model = tf.keras.models.load_model(model_path)

# =========================
# LOAD CLASS NAMES
# =========================
metrics_path = RESULTS_DIR / "metrics.json"
metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))
class_names = metrics_data["class_names"]

# =========================
# PREPROCESS
# =========================
def preprocess_image(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("L").convert("RGB")
    img = img.resize(IMG_SIZE)

    x = np.array(img).astype(np.float32)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    return x

# =========================
# PREDICT ENDPOINT
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    x = preprocess_image(image_bytes)
    probs = model.predict(x, verbose=0)[0]

    pred_index = int(np.argmax(probs))
    confidence = float(probs[pred_index])
    tank_name = class_names[pred_index]

    return {
        "tank_name": tank_name,
        "confidence": confidence
    }
