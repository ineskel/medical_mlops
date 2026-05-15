from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import torchvision.transforms as transforms
import io

app = FastAPI(title="Medical MLOps API")

# --- Load models ---
# We'll add real models later, for now just the skeleton
models = {}

def load_models():
    # Will load mobilenet and resnet weights here in week 3
    pass

@app.on_event("startup")
def startup_event():
    load_models()

# --- Health check ---
@app.get("/")
def root():
    return {"status": "running", "models_loaded": list(models.keys())}

# --- Predict endpoint ---
@app.post("/predict/{model_name}")
async def predict(model_name: str, file: UploadFile = File(...)):
    if model_name not in ["mobilenet", "resnet"]:
        return {"error": f"Unknown model '{model_name}'. Choose 'mobilenet' or 'resnet'."}
    
    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Placeholder response until real weights are loaded
    return {
        "model": model_name,
        "prediction": "placeholder — model not loaded yet",
        "confidence": 0.0
    }

# --- A/B test endpoint ---
import random

@app.post("/predict/ab")
async def predict_ab(file: UploadFile = File(...)):
    # Randomly routes to mobilenet or resnet (50/50)
    chosen = random.choice(["mobilenet", "resnet"])
    return {
        "routed_to": chosen,
        "prediction": "placeholder",
        "confidence": 0.0
    }