import json
import os
from pathlib import Path
from typing import Optional

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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


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
    "https://ai-tank-api-l3cc.onrender.com",
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

model = None
class_names = ["unknown"]


# =========================
# PATHS
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "improved_gray_balanced"

MODEL_PATH = MODELS_DIR / "model_fixed.keras"
METRICS_PATH = RESULTS_DIR / "metrics.json"

IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.80


# =========================
# HELPERS
# =========================
def load_class_names():
    global class_names

    if METRICS_PATH.exists():
        try:
            data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            names = data.get("class_names", [])
            if isinstance(names, list) and len(names) > 0:
                class_names = names
                print("CLASS NAMES LOADED:", class_names)
                return
        except Exception as e:
            print("FAILED TO LOAD CLASS NAMES:", e)

    class_names = ["unknown"]
    print("USING FALLBACK CLASS NAMES:", class_names)


def get_model():
    global model

    if model is None:
        if not MODEL_PATH.exists():
            raise Exception(f"MODEL NOT FOUND: {MODEL_PATH}")

        print("LOADING MODEL:", MODEL_PATH)
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            safe_mode=False
        )

    return model


def preprocess_uploaded_image(file_obj) -> np.ndarray:
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    img = Image.open(file_obj).convert("RGB")
    img = img.convert("L").convert("RGB")
    img = img.resize(IMG_SIZE)

    arr = np.array(img).astype(np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    return arr


def ask_gpt(messages):
    if client is None:
        return "OpenAI key not configured."

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=120,
            temperature=0.6,
        )
        return response.choices[0].message.content
    except Exception as e:
        print("GPT ERROR:", e)
        return "GPT error"


def find_tank_in_db(session: Session, tank_name: str) -> Optional[Tank]:
    return session.exec(
        select(Tank).where(Tank.name.ilike(f"%{tank_name}%"))
    ).first()


def build_system_prompt(tank_name: str, description: str) -> str:
    return f"""
You are a friendly military museum guide for normal visitors, not experts.

Rules:
- Keep the reply short and easy.
- Use simple everyday language.
- Maximum 3 short paragraphs.
- Prefer 50 to 90 words.
- Do not give long technical details unless the user asks.
- Focus on:
  1. what it is
  2. why it is important
  3. one interesting fact
- If the user asks a simple question, answer simply.
- If the user asks for more detail, then expand.
- Avoid difficult military terms unless you explain them simply.
- Sound natural, warm, and clear.

Tank: {tank_name}
Description: {description}
""".strip()


# =========================
# CHAT MODEL
# =========================
class ChatRequest(BaseModel):
    message: str


# =========================
# STARTUP
# =========================
@app.on_event("startup")
def startup_event():
    load_class_names()
    try:
        get_model()
    except Exception as e:
        print("MODEL LOAD ON STARTUP FAILED:", e)


# =========================
# HEALTH / TEST
# =========================
@app.get("/")
def root():
    return {
        "message": "AI Tank API is running",
        "model_exists": MODEL_PATH.exists(),
        "metrics_exists": METRICS_PATH.exists(),
    }


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

        if not file.filename:
            return JSONResponse(
                status_code=400,
                content={"error": "No file name provided"}
            )

        file.file.seek(0)
        arr = preprocess_uploaded_image(file.file)

        loaded_model = get_model()
        probs = loaded_model.predict(arr, verbose=0)[0]

        idx = int(np.argmax(probs))
        confidence = float(probs[idx])

        if idx >= len(class_names):
            raise Exception(
                f"Predicted class index {idx} is out of range. "
                f"class_names length = {len(class_names)}"
            )

        predicted_name = class_names[idx]
        tank_name = predicted_name if confidence >= CONFIDENCE_THRESHOLD else "Unknown"

        tank = None
        if tank_name != "Unknown":
            tank = find_tank_in_db(session, tank_name)

        current_tank = {
            "name": tank.name if tank else tank_name,
            "description": tank.description if tank else ""
        }

        chat_history = [
            {
                "role": "system",
                "content": build_system_prompt(
                    current_tank["name"],
                    current_tank["description"]
                )
            }
        ]

        gpt_text = ask_gpt(chat_history) if tank_name != "Unknown" else "Please try another image."

        return {
            "tank_name": current_tank["name"],
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

        if not req.message.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "Message cannot be empty"}
            )

        if not current_tank:
            return JSONResponse(
                status_code=400,
                content={"error": "No tank context yet. Use /predict first."}
            )

        context = f"""
Tank: {current_tank.get("name")}
Description: {current_tank.get("description")}
""".strip()

        chat_history.append({
            "role": "user",
            "content": context + "\nUser: " + req.message
        })

        reply = ask_gpt(chat_history)

        chat_history.append({
            "role": "assistant",
            "content": reply
        })

        return {"reply": reply}

    except Exception as e:
        print("CHAT ERROR:", e)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )