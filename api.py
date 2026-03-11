import io
import json
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from sqlmodel import SQLModel, select
from database import engine, get_session
from models import Tank


app = FastAPI(title="AI Tank Museum Guide API")

# create database tables
SQLModel.metadata.create_all(engine)


# =========================
# PATHS
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "improved_gray_balanced"

IMG_SIZE = (224, 224)


# =========================
# FIND MODEL
# =========================
def find_latest_best_model():
    candidates = sorted(MODELS_DIR.glob("improved_gray_balanced_best_*.keras"))
    if not candidates:
        raise FileNotFoundError("No trained model found in models folder.")
    return candidates[-1]


# =========================
# LAZY LOAD MODEL
# =========================
model = None


def get_model():
    global model

    if model is None:
        model_path = find_latest_best_model()
        print("Loading model:", model_path)

        model = tf.keras.models.load_model(model_path)

    return model


# =========================
# LOAD CLASS NAMES
# =========================
metrics_path = RESULTS_DIR / "metrics.json"

if metrics_path.exists():
    metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))
    class_names = metrics_data["class_names"]
else:
    print("WARNING: metrics.json not found")
    class_names = ["unknown"]


# =========================
# IMAGE PREPROCESSING
# =========================
def preprocess_image(image_bytes: bytes):

    img = Image.open(io.BytesIO(image_bytes))

    # match training preprocessing
    img = img.convert("L").convert("RGB")
    img = img.resize(IMG_SIZE)

    x = np.array(img).astype(np.float32)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)

    return x


# =========================
# ROOT TEST
# =========================
@app.get("/")
def root():
    return {"message": "Tank API running"}


# =========================
# HEALTH CHECK
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}


# =========================
# PREDICT ENDPOINT
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        # check file type
        if not file.content_type.startswith("image"):
            return {"error": "File must be an image"}

        image_bytes = await file.read()

        if not image_bytes:
            return {"error": "Empty file"}

        # preprocess image
        x = preprocess_image(image_bytes)

        # load model
        model = get_model()

        # predict
        probs = model.predict(x, verbose=0)[0]

        pred_index = int(np.argmax(probs))
        confidence = float(probs[pred_index])

        # debug logs
        print("Predicted index:", pred_index)
        print("Class names:", class_names)

        if pred_index < len(class_names):
            tank_name = class_names[pred_index]
        else:
            tank_name = "unknown"

        # =========================
        # DATABASE QUERY
        # =========================
        tank = None

        with next(get_session()) as session:

            tank = session.exec(
                select(Tank).where(Tank.name == tank_name)
            ).first()

        return {
            "tank_name": tank_name,
            "confidence": confidence,
            "country": tank.country if tank else None,
            "year": tank.year if tank else None,
            "description": tank.description if tank else None
        }

    except Exception as e:

        print("Prediction error:", e)

        return {
            "error": str(e)
        }