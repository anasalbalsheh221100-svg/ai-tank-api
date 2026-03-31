import io
import json
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Depends
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from sqlmodel import SQLModel, Session, select
from database import engine, get_session
from models import Tank
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
import os
from openai import OpenAI
from pydantic import BaseModel


# =========================
# ENV + OPENAI
# =========================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# APP
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

SQLModel.metadata.create_all(engine)

# =========================
# GLOBAL STATE
# =========================
current_tank = {}
chat_history = []

# =========================
# MODEL
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "improved_gray_balanced"

model = None
IMG_SIZE = (224, 224)

def get_model():
    global model
    if model is None:
        model_path = sorted(MODELS_DIR.glob("*.keras"))[-1]
        model = tf.keras.models.load_model(model_path)
    return model


print("MODEL LOADED SUCCESSFULLY")

# =========================
# CLASS NAMES
# =========================
metrics_path = RESULTS_DIR / "metrics.json"

if metrics_path.exists():
    class_names = json.loads(metrics_path.read_text())["class_names"]
else:
    class_names = ["unknown"]

# =========================
# PREPROCESS (FIXED)
# =========================
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale
    img = img.resize(IMG_SIZE)

    arr = np.array(img)
    arr = np.stack((arr,)*3, axis=-1)   # important

    arr = arr.astype(np.float32)
    arr = preprocess_input(arr)

    return np.expand_dims(arr, axis=0)

# =========================
# GPT FUNCTION
# =========================
def ask_gpt(messages):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return response.choices[0].message.content

# =========================
# CHAT MODEL
# =========================
class ChatRequest(BaseModel):
    message: str

# =========================
# PREDICT
# =========================
@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    session: Session = Depends(get_session)
):
    global current_tank, chat_history

    print("START PREDICT")

    # =========================
    # READ IMAGE (SAFE)
    # =========================
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    file.file.seek(0)

    try:
        img = Image.open(file.file)
        img = img.convert("RGB")
    except Exception as e:
        print("IMAGE ERROR:", e)
        return {"error": "Invalid image file"}

    # =========================
    # PREPROCESS
    # =========================
    img = img.resize(IMG_SIZE)

    arr = np.array(img).astype(np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    # =========================
    # PREDICT
    # =========================
    model = get_model()
    probs = model.predict(arr)[0]

    print("PROBS:", probs)
    print("MAX:", np.max(probs))

    idx = int(np.argmax(probs))
    confidence = float(probs[idx])

    # =========================
    # 🔥 THRESHOLD (الأهم)
    # =========================
    if confidence < 0.8:
        tank_name = "Unknown try another image"
    else:
        tank_name = class_names[idx]

    # =========================
    # DATABASE
    # =========================
    tank = None
    if tank_name != "Unknown":
        tank = session.exec(
            select(Tank).where(Tank.name.ilike(f"%{tank_name}%"))
        ).first()

    # =========================
    # CURRENT STATE
    # =========================
    current_tank = {
        "name": tank.name if tank else tank_name,
        "description": tank.description if tank else ""
    }

    # =========================
    # CHAT RESET
    # =========================
    chat_history = [
        {
            "role": "system",
            "content": f"""
You are a military museum guide.
Explain in SIMPLE and SHORT way (max 4-5 lines).

Tank: {current_tank['name']}
Description: {current_tank['description']}
"""
        }
    ]

    # =========================
    # GPT (SAFE)
    # =========================
    if tank_name == "Unknown":
        gpt_text = "This does not look like a tank. Please try another image."
    else:
        try:
            gpt_text = ask_gpt(chat_history)
        except:
            gpt_text = "AI is currently unavailable."

    print("DONE PREDICT")

    # =========================
    # RESPONSE
    # =========================
    return {
        "tank_name": current_tank["name"],
        "confidence": round(confidence, 4),
        "country": tank.country if tank else None,
        "year": tank.year if tank else None,
        "description": current_tank["description"],
        "gpt_explanation": gpt_text
    }


# =========================
# CHAT
# =========================
@app.post("/chat")
def chat(req: ChatRequest):
    global chat_history, current_tank

    message = req.message

    # 🔥 أهم سطر
    context = f"""
Current tank: {current_tank.get("name")}
Description: {current_tank.get("description")}
"""

    chat_history.append({
        "role": "user",
        "content": context + "\nUser: " + message
    })

    reply = ask_gpt(chat_history)

    chat_history.append({"role": "assistant", "content": reply})

    return {"reply": reply}

# =========================
# TEST
# =========================
@app.get("/test")
def test():
    return {"working": True}