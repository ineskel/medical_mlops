from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as tv_models
import numpy as np
import io, os, time, random, sys

# ── Path setup so logger import works from any working directory ───────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.logger import log_request

app = FastAPI(title="Medical MLOps API — ChestMNIST")

LABEL_NAMES = [
    'Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule',
    'Pneumonia','Pneumothorax','Consolidation','Edema','Emphysema',
    'Fibrosis','Pleural_Thickening','Hernia'
]
NUM_CLASSES = 14

TUNED_THRESHOLDS_MV2 = {
    'Atelectasis':0.70,'Cardiomegaly':0.90,'Effusion':0.65,
    'Infiltration':0.55,'Mass':0.80,'Nodule':0.60,
    'Pneumonia':0.85,'Pneumothorax':0.80,'Consolidation':0.75,
    'Edema':0.90,'Emphysema':0.80,'Fibrosis':0.75,
    'Pleural_Thickening':0.75,'Hernia':0.90
}

class GrayscaleTo3Ch(nn.Module):
    def forward(self, x): return x.expand(-1, 3, -1, -1)

def build_model(backbone):
    if backbone == 'mobilenet_v2':
        m = tv_models.mobilenet_v2(weights=None)
        m.classifier = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    else:
        m = tv_models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return nn.Sequential(GrayscaleTo3Ch(), m)

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

_models = {}

def _load(backbone, filename):
    path = os.path.join(ROOT, "models", filename)
    if not os.path.exists(path):
        print(f"[WARN] {path} not found — run scripts/restore_checkpoints.py")
        return None
    m = build_model(backbone)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    print(f"[OK] {backbone} loaded")
    return m

@app.on_event("startup")
def startup():
    _models["mobilenet"] = _load("mobilenet_v2", "mobilenet_v2_finetuned_best.pth")
    _models["resnet"]    = _load("resnet18",     "resnet18_finetuned_best.pth")

def infer(model, image, thresholds=None):
    tensor = get_transform()(image).unsqueeze(0)
    t0     = time.perf_counter()
    with torch.no_grad():
        probs = torch.sigmoid(model(tensor)).squeeze().numpy()
    ms  = (time.perf_counter() - t0) * 1000
    thr = [thresholds[n] for n in LABEL_NAMES] if thresholds else [0.5]*NUM_CLASSES
    preds = {
        name: {"probability": round(float(probs[i]), 4),
               "positive":    bool(probs[i] >= thr[i]),
               "threshold":   round(thr[i], 2)}
        for i, name in enumerate(LABEL_NAMES)
    }
    return {
        "predictions": preds,
        "positives":   [n for n, v in preds.items() if v["positive"]],
        "latency_ms":  round(ms, 2),
    }

@app.get("/")
def root():
    return {"status": "running",
            "models_loaded": [k for k, v in _models.items() if v]}

@app.get("/health")
def health():
    return {"status": "ok",
            "models": {k: v is not None for k, v in _models.items()}}

@app.post("/predict/ab")
async def predict_ab(file: UploadFile = File(...)):
    available = [k for k, v in _models.items() if v]
    if not available:
        raise HTTPException(503, "No models loaded.")
    chosen    = random.choice(available)
    image     = Image.open(io.BytesIO(await file.read())).convert("L")
    result    = infer(_models[chosen], image)
    thr_mode  = "tuned" if chosen == "mobilenet" else "fixed_0.5"

    log_request(
        model_name=chosen,
        image_name=file.filename or "unknown",
        predictions=result["predictions"],
        positives=result["positives"],
        latency_ms=result["latency_ms"],
        threshold_mode=thr_mode,
        source="api_ab",
    )
    return {"routed_to": chosen, "ab_test": True, **result}

@app.post("/predict/{model_name}")
async def predict(model_name: str, file: UploadFile = File(...),
                  tuned_thresholds: bool = True):
    if model_name not in _models:
        raise HTTPException(400, "Unknown model. Use 'mobilenet' or 'resnet'.")
    if _models[model_name] is None:
        raise HTTPException(503, "Checkpoint missing. Run restore_checkpoints.py")

    image    = Image.open(io.BytesIO(await file.read())).convert("L")
    thr      = TUNED_THRESHOLDS_MV2 if (tuned_thresholds and model_name == "mobilenet") else None
    thr_mode = "tuned" if thr else "fixed_0.5"
    result   = infer(_models[model_name], image, thr)

    log_request(
        model_name=model_name,
        image_name=file.filename or "unknown",
        predictions=result["predictions"],
        positives=result["positives"],
        latency_ms=result["latency_ms"],
        threshold_mode=thr_mode,
        source="api",
    )
    return {"model": model_name, "threshold_mode": thr_mode, **result}