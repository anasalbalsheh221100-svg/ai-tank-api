import io
import json
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Depends
import tensorflow as tf
from PIL import Image, ImageFile
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
# PATHS
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "improved_gray_balanced"

model = None
IMG_SIZE = (224, 224)

# =========================
# LOAD MODEL (FIXED)
# =========================
def get_model():
    global model

    if model is None:
        model_files = list(MODELS_DIR.glob("*.h5"))  # 🔥 مهم

        if not model_files:
            raise Exception("NO MODEL FILE FOUND IN models/")

        model_path = model_files[-1]

        print("LOADING MODEL:", model_path)

        model = tf.keras.models.load_model(model_path, compile=False)

    return model

# =========================
# CLASS NAMES (SAFE)
# =========================
metrics_path = RESULTS_DIR / "metrics.json"

if metrics_path.exists():
    class_names = json.loads(metrics_path.read_text())["class_names"]
    print("CLASS NAMES LOADED")
else:
    class_names = ["unknown"]
    print("WARNING: metrics.json NOT FOUND")

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
# PREDICT (FIXED + DEBUG)
# =========================
@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    session: Session = Depends(get_session)
):
    global current_tank, chat_history

    print("START PREDICT")

    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        file.file.seek(0)

        # =========================
        # READ IMAGE
        # =========================
        img = Image.open(file.file)
        img = img.convert("RGB")
        print("IMAGE LOADED")

        # =========================
        # PREPROCESS
        # =========================
        img = img.resize(IMG_SIZE)
        arr = np.array(img).astype(np.float32)
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)

        print("IMAGE PREPROCESSED")

        # =========================
        # MODEL
        # =========================
        model = get_model()
        print("MODEL READY")

        probs = model.predict(arr)[0]
        print("PREDICTION DONE")

        idx = int(np.argmax(probs))
        confidence = float(probs[idx])

        # =========================
        # THRESHOLD
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
        # GPT
        # =========================
        if tank_name == "Unknown":
            gpt_text = "This does not look like a tank. Please try another image."
        else:
            try:
                gpt_text = ask_gpt(chat_history)
            except Exception as e:
                print("GPT ERROR:", e)
                gpt_text = "AI is currently unavailable."

        print("DONE PREDICT")

        return {
            "tank_name": current_tank["name"],
            "confidence": round(confidence, 4),
            "country": tank.country if tank else None,
            "year": tank.year if tank else None,
            "description": current_tank["description"],
            "gpt_explanation": gpt_text
        }

    except Exception as e:
        print("ERROR:", str(e))
        return {"error": str(e)}

# =========================
# CHAT
# =========================
@app.post("/chat")
def chat(req: ChatRequest):
    global chat_history, current_tank

    message = req.message

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