import io
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from openai import OpenAI
from PIL import Image, ImageFile
from pydantic import BaseModel
from sqlmodel import SQLModel, Session, select
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from database import engine, get_session
from models import Tank


# =========================
# ENV
# =========================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# APP
# =========================
app = FastAPI(
    docs_url="/docs",
    redoc_url="/redoc"
)

# =========================
# CORS
# =========================
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5000",
    "http://localhost:60252",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5000",
    "http://127.0.0.1:60252",
    "https://ai-tank-api-13cc.onrender.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# =========================
# PREFLIGHT
# =========================
@app.options("/{path:path}")
async def options_handler(path: str, request: Request):
    origin = request.headers.get("origin", "")
    headers = {
        "Access-Control-Allow-Methods": "DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Max-Age": "86400",
    }

    if origin in origins:
        headers["Access-Control-Allow-Origin"] = origin

    return Response(status_code=204, headers=headers)

# =========================
# DB
# =========================
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
# LOAD MODEL
# =========================
def get_model():
    global model

    if model is None:
        model_files = list(MODELS_DIR.glob("*.keras")) or list(MODELS_DIR.glob("*.h5"))

        if not model_files:
            raise Exception("NO MODEL FILE FOUND")

        model_path = model_files[-1]
        print("LOADING MODEL:", model_path)

        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            safe_mode=False
        )

    return model

# =========================
# CLASS NAMES
# =========================
metrics_path = RESULTS_DIR / "metrics.json"

if metrics_path.exists():
    class_names = json.loads(metrics_path.read_text())["class_names"]
else:
    class_names = ["unknown"]

# =========================
# GPT
# =========================
def ask_gpt(messages):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        print("GPT ERROR:", e)
        return "GPT error"

# =========================
# CHAT MODEL
# =========================
class ChatRequest(BaseModel):
    message: str

# =========================
# HEALTH / TEST
# =========================
@app.get("/")
def root():
    return {"message": "AI Tank API is running"}

@app.get("/test")
def test():
    return {"working": True}

# =========================
# PREDICT
# =========================
@app.post("/predict")
def predict(file: UploadFile = File(...), session: Session = Depends(get_session)):
    global current_tank, chat_history

    try:
        print("PREDICT REQUEST RECEIVED")
        print("FILENAME:", file.filename)
        print("CONTENT TYPE:", file.content_type)

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        file.file.seek(0)

        img = Image.open(file.file).convert("RGB")
        img = img.resize(IMG_SIZE)

        arr = np.array(img).astype(np.float32)
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)

        loaded_model = get_model()
        probs = loaded_model.predict(arr)[0]

        idx = int(np.argmax(probs))
        confidence = float(probs[idx])

        tank_name = class_names[idx] if confidence >= 0.8 else "Unknown"

        tank = None
        if tank_name != "Unknown":
            tank = session.exec(
                select(Tank).where(Tank.name.ilike(f"%{tank_name}%"))
            ).first()

        current_tank = {
            "name": tank.name if tank else tank_name,
            "description": tank.description if tank else ""
        }

        chat_history = [
            {
                "role": "system",
                "content": f"""
You are a military museum guide.
Explain simply and clearly for normal visitors.

Tank: {current_tank['name']}
Description: {current_tank['description']}
"""
            }
        ]

        gpt_text = ask_gpt(chat_history) if tank_name != "Unknown" else "Try another image"

        return {
            "tank_name": current_tank["name"],
            "confidence": round(confidence, 4),
            "country": tank.country if tank else None,
            "year": tank.year if tank else None,
            "description": current_tank["description"],
            "gpt_explanation": gpt_text
        }

    except Exception as e:
        print("PREDICT ERROR:", e)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# =========================
# CHAT
# =========================
@app.post("/chat")
def chat(req: ChatRequest):
    global chat_history, current_tank

    try:
        print("CHAT REQUEST RECEIVED:", req.message)

        context = f"""
Tank: {current_tank.get("name")}
Description: {current_tank.get("description")}
"""

        chat_history.append({
            "role": "user",
            "content": context + "\nUser: " + req.message
        })

        reply = ask_gpt(chat_history)

        chat_history.append({"role": "assistant", "content": reply})

        return {"reply": reply}

    except Exception as e:
        print("CHAT ERROR:", e)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )