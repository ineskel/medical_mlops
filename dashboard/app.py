"""
dashboard/app.py — ChestMNIST Multi-Site MLOps Monitor
Inspired by Arize AI / NannyML / CheXstray architecture.
Novel contribution: Drift Fingerprint radar chart per hospital.
3 tabs: Fleet Overview | Site Inspector | Audit Log
Prerequisites:
    uvicorn src.api:app --port 8000   (terminal 1)
    streamlit run dashboard/app.py    (terminal 2)

Email alerts: set ALERT_EMAIL_FROM, ALERT_EMAIL_TO, ALERT_EMAIL_PASSWORD in env
or in a .env file at project root.
"""
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import requests
import time, io, os, math, json, smtplib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from email.mime.text import MIMEText
from PIL import Image as PILImage, ImageEnhance
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
IMG_ROOT = ROOT / "test_images"
API_URL  = os.environ.get("API_URL", "https://yasmine0421-chestmnist-api.hf.space")
LOG_FILE = ROOT / "logs" / "inference_log.jsonl"
LOG_FILE.parent.mkdir(exist_ok=True)

# ── Email config (set these as env vars or in .env) ───────────────────────────
ALERT_EMAIL_FROM     = os.environ.get("ALERT_EMAIL_FROM", "")
ALERT_EMAIL_TO       = os.environ.get("ALERT_EMAIL_TO", "")
ALERT_EMAIL_PASSWORD = os.environ.get("ALERT_EMAIL_PASSWORD", "")
EMAIL_ENABLED        = bool(ALERT_EMAIL_FROM and ALERT_EMAIL_TO and ALERT_EMAIL_PASSWORD)

# ── MLflow model registry config ──────────────────────────────────────────────
MODEL_NAME = "ChestMNIST-MobileNetV2"

# ── Hospital config ────────────────────────────────────────────────────────────
HOSPITALS = {
    "A": {
        "label":       "Hospital A — Algiers Central",
        "subtitle":    "Reference scanner (clean)",
        "folder":      "hospital_A",
        "drift_type":  "None — P(X,Y) stable",
        "shift_class": "Reference",
        "color":       "#2ecc71",
        "interval":    2,
        "perturb":     "none",
        "lat": 36.7372, "lon": 3.0865,
        "city": "Algiers",
        "equipment": "Siemens SOMATOM (2022) — Top-tier, well-maintained",
        "volume":    "~120 scans/day",
        "notes":     "Reference site. Clean scanner, stable patient demographics. "
                     "Model was originally trained on data from this site.",
        "risk":      "Low",
    },
    "B": {
        "label":       "Hospital B — Oran Regional",
        "subtitle":    "Aging scanner (brightness↓)",
        "folder":      "hospital_B",
        "drift_type":  "Covariate shift — incremental P(X)↓",
        "shift_class": "Covariate / Incremental",
        "color":       "#f39c12",
        "interval":    3,
        "perturb":     "brightness",
        "lat": 35.6969, "lon": -0.6331,
        "city": "Oran",
        "equipment": "GE Definium (2014) — Aging detector panel, brightness drift",
        "volume":    "~80 scans/day",
        "notes":     "Scanner detector panel degrading. Brightness decreases ~3% per month. "
                     "Maintenance scheduled Q3 2026 but not yet performed.",
        "risk":      "Medium",
    },
    "C": {
        "label":       "Hospital C — Constantine Rural",
        "subtitle":    "Low-res scanner (sudden drop)",
        "folder":      "hospital_C",
        "drift_type":  "Covariate shift — sudden (img≥5)",
        "shift_class": "Covariate / Sudden",
        "color":       "#e74c3c",
        "interval":    1,
        "perturb":     "resolution",
        "lat": 36.3650, "lon": 6.6147,
        "city": "Constantine",
        "equipment": "Philips DigitalDiagnost (2009) — Very old, resolution issues",
        "volume":    "~40 scans/day",
        "notes":     "Oldest scanner in the network. Sudden firmware bug introduced "
                     "resolution degradation after image #5 in each batch cycle. "
                     "High priority for replacement.",
        "risk":      "High",
    },
    "D": {
        "label":       "Hospital D — Annaba Pediatric",
        "subtitle":    "Rare-class prevalence shift",
        "folder":      "hospital_D",
        "drift_type":  "Label shift — gradual P(Y)↑ rare",
        "shift_class": "Label / Gradual",
        "color":       "#9b59b6",
        "interval":    5,
        "perturb":     "noise",
        "lat": 36.9000, "lon": 7.7667,
        "city": "Annaba",
        "equipment": "Canon CXDI-Elite (2018) — Good hardware, noisy environment",
        "volume":    "~60 scans/day",
        "notes":     "Pediatric hospital — patient demographics differ significantly "
                     "from training data. Rare pathologies (Hernia, Emphysema) appear "
                     "~3× more than in training distribution. Label shift growing over time.",
        "risk":      "Medium-High",
    },
}

LABEL_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

# ── Detector parameters ────────────────────────────────────────────────────────
PHT_DELTA   = 0.002
PHT_LAMBDA  = 4.0
PHT_WARNING = 2.0
ADWIN_DELTA = 0.002
SMOOTH_WIN  = 5
WARMUP_IMGS = 20

# ── Risk colors ────────────────────────────────────────────────────────────────
RISK_COLOR = {
    "Low": "#2ecc71", "Medium": "#f39c12",
    "Medium-High": "#e67e22", "High": "#e74c3c",
}

# ── Live perturbation ──────────────────────────────────────────────────────────
def apply_perturbation(img_path: Path, perturb: str, img_count: int) -> bytes:
    img = PILImage.open(img_path).convert("RGB")
    if perturb == "brightness":
        factor = max(0.10, 1.0 - (img_count / 25) * 0.90)
        img = ImageEnhance.Brightness(img).enhance(factor)
    elif perturb == "resolution":
        if img_count >= 5:
            severity = min(img_count - 5, 20)
            low_res  = max(4, 28 - severity)
            img = img.resize((low_res, low_res), PILImage.BILINEAR).resize(
                (224, 224), PILImage.NEAREST)
    elif perturb == "noise":
        arr   = np.array(img, dtype=np.float32)
        sigma = min(img_count * 3.5, 90)
        arr   = np.clip(arr + np.random.normal(0, sigma, arr.shape), 0, 255).astype(np.uint8)
        img   = PILImage.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ── Signal helpers ─────────────────────────────────────────────────────────────
def smooth_signal(history: list, new_val: float, win: int = SMOOTH_WIN) -> float:
    recent = (history[-(win - 1):] if len(history) >= win else history) + [new_val]
    return sum(recent) / len(recent)

# ── PHT ───────────────────────────────────────────────────────────────────────
def pht_update(state: dict, x: float, lam: float) -> dict:
    n    = state["n"] + 1
    mean = state["mean"] + (x - state["mean"]) / n
    U    = state["U"] + (x - mean - PHT_DELTA)
    M    = max(state["M"], U)
    val  = M - U
    warn = lam * 0.5
    status = "DRIFT" if val > lam else ("WARNING" if val > warn else "STABLE")
    return {"n": n, "mean": mean, "U": U, "M": M,
            "pht_val": round(val, 4), "status": status}

def fresh_pht():
    return {"n": 0, "mean": 0.0, "U": 0.0, "M": 0.0,
            "pht_val": 0.0, "status": "STABLE"}

# ── ADWIN ─────────────────────────────────────────────────────────────────────
def adwin_update(state: dict, x: float) -> dict:
    window = state["window"] + [x]
    if len(window) > 150:
        window = window[-150:]
    status, gap = "STABLE", 0.0
    if len(window) >= 10:
        n   = len(window)
        n0  = n // 2
        n1  = n - n0
        mu0 = sum(window[:n0]) / n0
        mu1 = sum(window[n0:]) / n1
        gap = abs(mu0 - mu1)
        m   = (n0 * n1) / (n0 + n1)
        eps = math.sqrt((1.0 / (2.0 * m)) * math.log(4.0 * n / ADWIN_DELTA))
        if gap >= eps:
            status = "DRIFT"
        elif gap >= eps * 0.6:
            status = "WARNING"
    return {"window": window, "status": status, "gap": round(gap, 4)}

def fresh_adwin():
    return {"window": [], "status": "STABLE", "gap": 0.0}

# ── Drift Fingerprint ─────────────────────────────────────────────────────────
def compute_drift_fingerprint(hid: str) -> dict:
    hst = st.session_state.hospitals[hid]
    scores = {}
    err_hist = hst["raw_error_hist"][-30:] if hst["raw_error_hist"] else []
    scores["Signal\nVolatility"] = min(1.0, float(np.std(err_hist)) / 0.2) if len(err_hist) > 3 else 0.0
    pht_hist = hst["pht_history"][-20:] if hst["pht_history"] else []
    if len(pht_hist) > 5:
        slope = (pht_hist[-1] - pht_hist[0]) / max(len(pht_hist), 1)
        scores["PHT\nSlope"] = min(1.0, max(0.0, slope / (hst["pht_lambda"] / 10)))
    else:
        scores["PHT\nSlope"] = 0.0
    scores["ADWIN\nGap"] = min(1.0, hst["adwin"]["gap"] / 0.15)
    conf_hist = hst["conf_history"][-20:] if hst["conf_history"] else []
    if len(conf_hist) > 5:
        trend = conf_hist[0] - conf_hist[-1]
        scores["Confidence\nTrend↓"] = min(1.0, max(0.0, trend / 0.4))
    else:
        scores["Confidence\nTrend↓"] = 0.0
    lat = hst["latency_ms"] or 0
    scores["Latency\nStress"] = min(1.0, max(0.0, (lat - 80) / 400))
    return scores

def render_drift_fingerprint(hid: str, color: str):
    fp   = compute_drift_fingerprint(hid)
    cats = list(fp.keys())
    vals = list(fp.values())
    cats_closed = cats + [cats[0]]
    vals_closed = vals + [vals[0]]
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)
    fill_color = f"rgba({r},{g},{b},0.3)"
    fig = go.Figure(go.Scatterpolar(
        r=vals_closed, theta=cats_closed, fill="toself",
        fillcolor=fill_color, line=dict(color=color, width=2), name=f"Hosp {hid}",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1],
                tickvals=[0.25, 0.5, 0.75, 1.0],
                ticktext=["Low", "Med", "High", "Max"],
                gridcolor="#333", linecolor="#333"),
            angularaxis=dict(gridcolor="#333", linecolor="#444"),
            bgcolor="#0e1117",
        ),
        paper_bgcolor="#0e1117", font=dict(color="white", size=10),
        margin=dict(l=40, r=40, t=30, b=30), height=230, showlegend=False,
    )
    return fig

# ── Recalibration ─────────────────────────────────────────────────────────────
def recalibrate_thresholds(hid: str) -> dict:
    hst    = st.session_state.hospitals[hid]
    recent = hst.get("recent_probs", [])
    if len(recent) < 10:
        return {}
    arr = np.array(recent)
    return {name: round(float(np.percentile(arr[:, i], 70)), 3)
            for i, name in enumerate(LABEL_NAMES)}

# ── Email alert ───────────────────────────────────────────────────────────────
def send_drift_email(hid: str, detector: str, pht_val: float, adwin_gap: float,
                     img_count: int, drift_type: str):
    if not EMAIL_ENABLED:
        st.toast("⚠️ Email disabled — set env vars", icon="📧")
        return
    try:
        cfg = HOSPITALS[hid]
        subject = f"🔴 DRIFT ALARM — Hospital {hid} ({cfg['city']}) [{detector}]"
        body = f"""
ChestMNIST MLOps Monitor — Drift Alert
=======================================
Hospital  : {cfg['label']}
Drift Type: {drift_type}
Detector  : {detector}
Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Metrics:
  PHT value  : {pht_val:.4f}  (alarm threshold: {cfg.get('pht_lambda', PHT_LAMBDA)})
  ADWIN gap  : {adwin_gap:.4f}
  Image count: {img_count}

Action required:
  → Review Site Inspector in the dashboard
  → Consider threshold recalibration
  → Flag for retraining if drift persists

Dashboard: https://medicalmlops-bxmlrkzynprnjtmb2pkvjz.streamlit.app/
MLflow UI: mlflow ui --backend-store-uri sqlite:///{ROOT}/experiments/mlflow.db
        """.strip()
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"]    = ALERT_EMAIL_FROM
        msg["To"]      = ALERT_EMAIL_TO
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(ALERT_EMAIL_FROM, ALERT_EMAIL_PASSWORD)
            server.sendmail(ALERT_EMAIL_FROM, ALERT_EMAIL_TO, msg.as_string())
    except Exception as e:
        st.toast(f"📧 Email failed: {e}", icon="❌")

# ── MLflow helpers ────────────────────────────────────────────────────────────
def get_mlflow_client():
    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{ROOT}/experiments/mlflow.db")
    return mlflow.tracking.MlflowClient()

def register_threshold_version(hid: str, thresholds: dict, trigger: str,
                                pht_val: float, adwin_gap: float) -> str | None:
    """
    Creates a new model version in the registry with recalibrated thresholds
    as tags. Same .pth weights, new decision boundaries = new deployable version.
    Returns the version number as string, or None on failure.
    """
    try:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{ROOT}/experiments/mlflow.db")
        client = get_mlflow_client()

        # Find the current production run to link to
        try:
            current_mv = client.get_model_version_by_alias(MODEL_NAME, "production")
            source_uri = current_mv.source
            base_run_id = current_mv.run_id
        except Exception:
            # No production alias yet — find any version
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            if not versions:
                return None
            source_uri  = versions[0].source
            base_run_id = versions[0].run_id

        # Log recalibration as a monitoring run to get a run_id
        mlflow.set_experiment("mlops-monitoring")
        with mlflow.start_run(run_name=f"recal_hosp_{hid}_{trigger}") as run:
            mlflow.set_tag("hospital_id",  hid)
            mlflow.set_tag("event",        "THRESHOLD_RECALIBRATION")
            mlflow.set_tag("trigger",      trigger)
            mlflow.log_metric("pht_val",   pht_val)
            mlflow.log_metric("adwin_gap", adwin_gap)
            for name, val in thresholds.items():
                mlflow.log_param(f"thr_{name}", val)
            recal_run_id = run.info.run_id

        # Create new model version with thresholds as tags
        mv = client.create_model_version(
            name=MODEL_NAME,
            source=source_uri,
            run_id=base_run_id,
            description=(
                f"Threshold recalibration for Hospital {hid}. "
                f"Trigger: {trigger}. PHT={pht_val:.4f} ADWIN={adwin_gap:.4f}. "
                f"Recal run: {recal_run_id[:8]}. "
                f"Weights unchanged — only decision boundaries updated."
            ),
        )
        version = mv.version

        # Wait for READY
        for _ in range(10):
            mv_status = client.get_model_version(MODEL_NAME, version)
            if mv_status.status == "READY":
                break
            time.sleep(0.5)

        # Tag all thresholds on the version
        for name, val in thresholds.items():
            client.set_model_version_tag(MODEL_NAME, version, f"thr_{name}", str(val))
        client.set_model_version_tag(MODEL_NAME, version, "hospital",      hid)
        client.set_model_version_tag(MODEL_NAME, version, "trigger",       trigger)
        client.set_model_version_tag(MODEL_NAME, version, "recal_run_id",  recal_run_id[:8])

        # Set staging alias
        client.set_registered_model_alias(MODEL_NAME, "staging", version)

        return str(version)
    except Exception as e:
        return None

def promote_to_production(version: str) -> bool:
    try:
        client = get_mlflow_client()
        client.set_registered_model_alias(MODEL_NAME, "production", version)
        client.set_model_version_tag(MODEL_NAME, version, "promoted_at",
                                     datetime.now().isoformat())
        return True
    except Exception:
        return False

def log_drift_mlflow(hid, pht_val, adwin_gap, img_count, drift_type, detector):
    try:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{ROOT}/experiments/mlflow.db")
        mlflow.set_experiment("mlops-monitoring")
        with mlflow.start_run(run_name=f"drift_{detector}_hosp_{hid}"):
            mlflow.set_tag("hospital_id", hid)
            mlflow.set_tag("drift_type",  drift_type)
            mlflow.set_tag("detector",    detector)
            mlflow.set_tag("event",       "DRIFT_DETECTED")
            mlflow.log_param("pht_lambda", PHT_LAMBDA)
            mlflow.log_param("smooth_win", SMOOTH_WIN)
            mlflow.log_metric("pht_value",   pht_val)
            mlflow.log_metric("adwin_gap",   adwin_gap)
            mlflow.log_metric("image_count", img_count)
    except Exception:
        pass

def log_action_mlflow(hid, action, details):
    try:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{ROOT}/experiments/mlflow.db")
        mlflow.set_experiment("mlops-monitoring")
        with mlflow.start_run(run_name=f"action_{action}_hosp_{hid}"):
            mlflow.set_tag("hospital_id", hid)
            mlflow.set_tag("action",      action)
            mlflow.set_tag("event",       "OPERATOR_ACTION")
            for k, v in details.items():
                mlflow.log_param(k, v)
    except Exception:
        pass

# ── Auto-recalibration on drift ───────────────────────────────────────────────
def auto_recalibrate_on_drift(hid: str, detector: str, pht_val: float,
                               adwin_gap: float, drift_type: str):
    """
    Called automatically when drift is first detected.
    Recalibrates thresholds, registers new staging version, sends email.
    """
    hst = st.session_state.hospitals[hid]

    # Only auto-recalibrate once per drift event (same guard as drift_logged)
    key = f"{hid}_autorecal"
    if key in st.session_state.drift_logged:
        return
    st.session_state.drift_logged.add(key)

    new_thr = recalibrate_thresholds(hid)
    if not new_thr:
        # Not enough probs yet — use flat 0.4 as fallback for injected demo
        new_thr = {name: 0.4 for name in LABEL_NAMES}

    hst["calibrated_thr"] = new_thr
    if not new_thr:
        return

    hst["calibrated_thr"] = new_thr

    # Register in MLflow registry as staging version
    version = register_threshold_version(
        hid, new_thr, trigger=detector,
        pht_val=pht_val, adwin_gap=adwin_gap
    )
    if version is None:
        st.toast("⚠️ Registry failed — check MLflow DB path", icon="🔴")
    hst["staging_version"] = version

    # Log action
    log_action_mlflow(hid, "auto_threshold_recalibrated", {
        "trigger": detector, "pht_val": pht_val,
        "adwin_gap": adwin_gap, "method": "percentile_70",
        "registry_version": version or "failed",
    })
    st.session_state.actions_taken.append({
        "ts": datetime.now().isoformat(), "hospital": hid,
        "action": "auto_recalibrated",
        "trigger": detector,
        "registry_version": version or "—",
    })

    # Send email alert
    send_drift_email(hid, detector, pht_val, adwin_gap,
                     hst["img_count"], drift_type)

    ts = datetime.now().strftime("%H:%M:%S")
    ver_str = f"v{version}" if version else "registry unavailable"
    st.session_state.event_log.insert(
        0, f"[{ts}] 🔧 Hosp {hid} AUTO-RECALIBRATED → {ver_str} staging")

# ── Helpers ───────────────────────────────────────────────────────────────────
def status_icon(s):
    return {"STABLE": "🟢", "WARNING": "🟡", "DRIFT": "🔴"}.get(s, "⚪")

def combined_status(a, b):
    rank = {"STABLE": 0, "WARNING": 1, "DRIFT": 2}
    return a if rank.get(a, 0) >= rank.get(b, 0) else b

# ── Session state ──────────────────────────────────────────────────────────────
def init_state():
    if "initialized" in st.session_state:
        return
    st.session_state.initialized   = True
    st.session_state.running       = False
    st.session_state.event_log     = []
    st.session_state.drift_logged  = set()
    st.session_state.actions_taken = []
    st.session_state.hospitals     = {}
    now = time.time()
    for i, (hid, cfg) in enumerate(HOSPITALS.items()):
        imgs   = sorted((IMG_ROOT / cfg["folder"]).glob("*.png"))
        lam    = PHT_LAMBDA * 2 if hid == "A" else PHT_LAMBDA
        warmup = 8 if hid == "C" else WARMUP_IMGS
        st.session_state.hospitals[hid] = {
            "images":            imgs,
            "img_idx":           0,
            "last_tick":         now - (i / len(HOSPITALS)) * cfg["interval"],
            "pht":               fresh_pht(),
            "adwin":             fresh_adwin(),
            "pht_lambda":        lam,
            "warmup_imgs":       warmup,
            "raw_error_hist":    [],
            "conf_history":      [],
            "pht_history":       [],
            "adwin_gap_history": [],
            "smooth_history":    [],
            "recent_probs":      [],
            "img_count":         0,
            "last_conf":         None,
            "last_img_path":     None,
            "last_img_bytes":    None,
            "last_positives":    [],
            "last_probs":        {},
            "latency_ms":        None,
            "last_error":        None,
            "calibrated_thr":    None,
            "staging_version":   None,
            "warmup_done":       False,
            "warmup_errors":     [],
        }

# ── Inference log ──────────────────────────────────────────────────────────────
def log_inference(hid, img_name, conf, positives, latency_ms,
                  pht_val, pht_status, adwin_status):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps({
            "ts": datetime.now().isoformat(), "hospital": hid,
            "image": img_name, "max_conf": round(conf, 4),
            "positives": positives, "latency_ms": round(latency_ms, 2),
            "pht_val": round(pht_val, 4),
            "pht_status": pht_status, "adwin_status": adwin_status,
        }) + "\n")

# ── API ────────────────────────────────────────────────────────────────────────
def call_api_bytes(img_bytes: bytes, filename: str) -> dict | None:
    try:
        resp = requests.post(
            f"{API_URL}/predict/mobilenet",
            files={"file": (filename, img_bytes, "image/png")},
            params={"tuned_thresholds": "true"},
            timeout=15,
        )
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None

def check_api_health() -> bool:
    try:
        h = requests.get(f"{API_URL}/health", timeout=3).json()
        return bool(h.get("models", {}).get("mobilenet", False))
    except Exception:
        return False

# ── Core tick ──────────────────────────────────────────────────────────────────
def do_tick(hid: str) -> bool:
    try:
        cfg = HOSPITALS[hid]
        hst = st.session_state.hospitals[hid]
        now = time.time()
        if now - hst["last_tick"] < cfg["interval"]:
            return False
        if not hst["images"]:
            hst["last_error"] = "No images found"
            return False
        img_path  = hst["images"][hst["img_idx"] % len(hst["images"])]
        img_count = hst["img_count"]
        img_bytes = apply_perturbation(img_path, cfg["perturb"], img_count)
        result    = call_api_bytes(img_bytes, img_path.name)
        tick_done = time.time()
        if result is None:
            hst["last_tick"]  = tick_done
            hst["last_error"] = "API timeout"
            return False
        hst["last_error"] = None
        probs      = [result["predictions"][lbl]["probability"] for lbl in LABEL_NAMES]
        max_conf   = max(probs)
        latency_ms = result.get("latency_ms", 0.0)
        raw_error  = 1.0 - max_conf

        if img_count < hst["warmup_imgs"]:
            hst["warmup_errors"].append(raw_error)
            hst["conf_history"].append(round(max_conf, 4))
            hst["pht_history"].append(0.0)
            hst["adwin_gap_history"].append(0.0)
            hst["smooth_history"].append(round(raw_error, 4))
            hst["img_count"]     += 1
            hst["img_idx"]       += 1
            hst["last_tick"]      = tick_done
            hst["last_conf"]      = round(max_conf, 4)
            hst["last_img_path"]  = img_path
            hst["last_img_bytes"] = img_bytes
            hst["last_positives"] = result.get("positives", [])
            hst["last_probs"]     = {LABEL_NAMES[i]: round(probs[i], 4) for i in range(14)}
            hst["latency_ms"]     = round(latency_ms, 1)
            ts = datetime.now().strftime("%H:%M:%S")
            st.session_state.event_log.insert(
                0, f"[{ts}] ⏳ Hosp {hid} #{img_count+1:02d} WARMUP conf={max_conf:.3f}")
            st.session_state.event_log = st.session_state.event_log[:80]
            return True

        if not hst["warmup_done"] and hst["warmup_errors"]:
            baseline           = sum(hst["warmup_errors"]) / len(hst["warmup_errors"])
            hst["pht"]["mean"] = baseline
            hst["warmup_done"] = True

        smooth_err = smooth_signal(hst["raw_error_hist"], raw_error)
        hst["raw_error_hist"].append(raw_error)
        if len(hst["raw_error_hist"]) > 200:
            hst["raw_error_hist"] = hst["raw_error_hist"][-200:]
        hst["pht"]   = pht_update(hst["pht"], smooth_err, hst["pht_lambda"])
        hst["adwin"] = adwin_update(hst["adwin"], smooth_err)
        hst["recent_probs"].append(probs)
        if len(hst["recent_probs"]) > 30:
            hst["recent_probs"] = hst["recent_probs"][-30:]
        hst["conf_history"].append(round(max_conf, 4))
        hst["pht_history"].append(hst["pht"]["pht_val"])
        hst["adwin_gap_history"].append(hst["adwin"]["gap"])
        hst["smooth_history"].append(round(smooth_err, 4))
        hst["img_count"]     += 1
        hst["img_idx"]       += 1
        hst["last_tick"]      = tick_done
        hst["last_conf"]      = round(max_conf, 4)
        hst["last_img_path"]  = img_path
        hst["last_img_bytes"] = img_bytes
        hst["last_positives"] = result.get("positives", [])
        hst["last_probs"]     = {LABEL_NAMES[i]: round(probs[i], 4) for i in range(14)}
        hst["latency_ms"]     = round(latency_ms, 1)

        pht_status   = hst["pht"]["status"]
        adwin_status = hst["adwin"]["status"]
        status       = combined_status(pht_status, adwin_status)
        ts           = datetime.now().strftime("%H:%M:%S")

        log_inference(hid, img_path.name, max_conf, hst["last_positives"],
                      latency_ms, hst["pht"]["pht_val"], pht_status, adwin_status)

        for detector, det_status, key in [
            ("PHT",   pht_status,   f"{hid}_pht"),
            ("ADWIN", adwin_status, f"{hid}_adwin"),
        ]:
            if det_status == "DRIFT" and key not in st.session_state.drift_logged:
                log_drift_mlflow(hid, hst["pht"]["pht_val"], hst["adwin"]["gap"],
                                 hst["img_count"], cfg["drift_type"], detector)
                st.session_state.drift_logged.add(key)
                # ── AUTO-RECALIBRATE + EMAIL on first drift ────────────────
                auto_recalibrate_on_drift(
                    hid, detector, hst["pht"]["pht_val"],
                    hst["adwin"]["gap"], cfg["drift_type"]
                )

        if status in ("DRIFT", "WARNING"):
            msg = (f"[{ts}] {status_icon(status)} Hosp {hid} #{hst['img_count']:02d} "
                   f"PHT:{status_icon(pht_status)}({hst['pht']['pht_val']:.2f}) "
                   f"ADWIN:{status_icon(adwin_status)}({hst['adwin']['gap']:.3f}) "
                   f"err={smooth_err:.3f}")
        else:
            msg = (f"[{ts}] ✅ Hosp {hid} #{hst['img_count']:02d} "
                   f"conf={max_conf:.3f} err={smooth_err:.3f} lat={latency_ms:.0f}ms")
        st.session_state.event_log.insert(0, msg)
        st.session_state.event_log = st.session_state.event_log[:80]
        return True
    except Exception as e:
        st.session_state.hospitals[hid]["last_error"] = str(e)
        return False

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Fleet Overview
# ═══════════════════════════════════════════════════════════════════════════════
@st.fragment(run_every=2)
def render_fleet_overview():
    total_imgs  = sum(st.session_state.hospitals[h]["img_count"] for h in HOSPITALS)
    n_drift     = sum(1 for h in HOSPITALS if combined_status(
        st.session_state.hospitals[h]["pht"]["status"],
        st.session_state.hospitals[h]["adwin"]["status"]) == "DRIFT")
    n_warn      = sum(1 for h in HOSPITALS if combined_status(
        st.session_state.hospitals[h]["pht"]["status"],
        st.session_state.hospitals[h]["adwin"]["status"]) == "WARNING")
    lats = [st.session_state.hospitals[h]["latency_ms"]
            for h in HOSPITALS if st.session_state.hospitals[h]["latency_ms"]]
    avg_latency = np.mean(lats) if lats else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Images Processed", f"{total_imgs:,}")
    k2.metric("🔴 Active Drift Alarms", n_drift,
              delta=f"+{n_drift}" if n_drift else None, delta_color="inverse")
    k3.metric("🟡 Warnings", n_warn)
    k4.metric("Avg Latency", f"{avg_latency:.0f} ms")
    st.divider()

    st.markdown("### 🏥 Site Status Table")
    rows = []
    for hid, cfg in HOSPITALS.items():
        hst    = st.session_state.hospitals[hid]
        pht_s  = hst["pht"]["status"]
        adw_s  = hst["adwin"]["status"]
        status = combined_status(pht_s, adw_s)
        rows.append({
            "Site":       f"{status_icon(status)} {cfg['label']}",
            "Status":     status,
            "Shift Type": cfg["shift_class"],
            "Images":     hst["img_count"],
            "PHT":        f"{hst['pht']['pht_val']:.2f}",
            "ADWIN Gap":  f"{hst['adwin']['gap']:.3f}",
            "Last Conf":  f"{hst['last_conf']:.3f}" if hst["last_conf"] else "—",
            "Latency":    f"{hst['latency_ms']}ms" if hst["latency_ms"] else "—",
            "Calibrated": "✓" if hst["calibrated_thr"] else "—",
            "Registry":   f"v{hst['staging_version']} staging" if hst["staging_version"] else "—",
        })
    df = pd.DataFrame(rows)
    def color_status(val):
        return {"DRIFT":   "background-color:#ff4b4b;color:white",
                "WARNING": "background-color:#ffa500;color:black",
                "STABLE":  "background-color:#21c354;color:black"}.get(val, "")
    st.dataframe(df.style.map(color_status, subset=["Status"]),
                 use_container_width=True, hide_index=True)
    st.divider()

    st.markdown("### 📊 Confidence Heatmap — All Sites Over Time")
    max_len = max((len(st.session_state.hospitals[h]["conf_history"])
                   for h in HOSPITALS), default=1) or 1
    z = np.array([
        [v for v in st.session_state.hospitals[h]["conf_history"]] +
        [np.nan] * (max_len - len(st.session_state.hospitals[h]["conf_history"]))
        for h in HOSPITALS
    ], dtype=float)
    fig_heat = go.Figure(go.Heatmap(
        z=z, x=list(range(1, max_len + 1)),
        y=[f"Hosp {h}" for h in HOSPITALS],
        colorscale="RdYlGn", zmin=0.3, zmax=1.0,
        colorbar=dict(title="Max Conf"),
    ))
    fig_heat.update_layout(
        height=200, margin=dict(l=80, r=20, t=20, b=40),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white"), xaxis_title="Image #", uirevision="stable",
    )
    st.plotly_chart(fig_heat, use_container_width=True, key="heatmap_fleet")

    st.markdown("### 📈 Drift Detector Signals — All Sites")
    with st.expander("ℹ️ How to read these charts"):
        st.markdown(f"""
**PHT chart (top):** The Page-Hinkley statistic M_T − U_T accumulates when the
model's uncertainty (1 − max_confidence) rises above its baseline.
- 🟢 Below {PHT_WARNING}: stable
- 🟡 Between {PHT_WARNING}–{PHT_LAMBDA}: warning zone
- 🔴 Above {PHT_LAMBDA}: drift alarm → auto-recalibration fires

**ADWIN chart (bottom):** The gap between the mean of the first and second halves
of a sliding window. When this gap exceeds the Hoeffding bound ε_cut, drift is detected.

On first DRIFT detection, thresholds are **automatically recalibrated** and a new
model version is registered in the MLflow registry under alias `staging`.
        """)

    fig = make_subplots(rows=2, cols=1,
        subplot_titles=(
            f"PHT  M_T−U_T  (alarm > λ, Hospital A uses λ={PHT_LAMBDA*2}, others λ={PHT_LAMBDA})",
            "ADWIN  |μ_W0−μ_W1|",
        ),
        vertical_spacing=0.18,
    )
    max_x = 2
    for hid, cfg in HOSPITALS.items():
        hst  = st.session_state.hospitals[hid]
        pht  = hst["pht_history"]
        adw  = hst["adwin_gap_history"]
        xs   = list(range(1, len(pht) + 1))
        max_x = max(max_x, len(pht))
        fig.add_trace(go.Scatter(
            x=xs, y=pht, mode="lines", name=f"Hosp {hid}",
            line=dict(color=cfg["color"], width=2), legendgroup=hid,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=list(range(1, len(adw)+1)), y=adw, mode="lines",
            name=f"ADWIN-{hid}", line=dict(color=cfg["color"], width=2, dash="dash"),
            legendgroup=hid, showlegend=False,
        ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[1, max_x], y=[PHT_LAMBDA]*2, mode="lines",
        line=dict(color="#e74c3c", dash="solid", width=1.5),
        name=f"λ alarm ({PHT_LAMBDA})",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[1, max_x], y=[PHT_WARNING]*2, mode="lines",
        line=dict(color="#f39c12", dash="dash", width=1.2),
        name=f"λ/2 warning ({PHT_WARNING})",
    ), row=1, col=1)
    fig.update_layout(
        height=420, margin=dict(l=50, r=20, t=50, b=20),
        legend=dict(orientation="h", y=-0.15, font=dict(size=11)),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white"), uirevision="stable",
    )
    fig.update_xaxes(gridcolor="#2d2d2d", title_text="Image #")
    fig.update_yaxes(gridcolor="#2d2d2d")
    st.plotly_chart(fig, use_container_width=True, key="pht_adwin_fleet")

    st.markdown("### 🚨 Alert Feed")
    alerts = [e for e in st.session_state.event_log
              if "DRIFT" in e or "WARNING" in e or "RECAL" in e or "✅" in e][:15]
    st.code("\n".join(alerts) if alerts else "No events yet. Start the simulation.",
            language=None)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Site Inspector
# ═══════════════════════════════════════════════════════════════════════════════
@st.fragment(run_every=None)
def render_site_static(hid: str):
    cfg = HOSPITALS[hid]
    st.markdown("### 🗺️ Site Location & Equipment Profile")
    map_col, info_col = st.columns([2, 1])
    with map_col:
        fig_map = go.Figure(go.Scattermap(
            lat=[cfg["lat"]], lon=[cfg["lon"]],
            mode="markers+text",
            marker=dict(size=18, color=cfg["color"], opacity=0.95),
            text=[f"Hosp {hid}"], textposition="top right",
            hovertext=[f"<b>{cfg['label']}</b><br>Risk: {cfg['risk']}<br>Equipment: {cfg['equipment']}"],
            hoverinfo="text",
        ))
        fig_map.update_layout(
            map=dict(style="carto-darkmatter",          # ← was mapbox=dict(
                    center=dict(lat=cfg["lat"], lon=cfg["lon"]), zoom=7),
            margin=dict(l=0, r=0, t=0, b=0), height=280, paper_bgcolor="#0e1117",
        )
        st.plotly_chart(fig_map, key=f"map_static_{hid}", use_container_width=True)
    with info_col:
        risk_color = RISK_COLOR.get(cfg["risk"], "#888")
        st.markdown(
            f"<div style='border:1px solid {cfg['color']};border-radius:8px;padding:14px'>"
            f"<b>📍 {cfg['city']}</b><br><br>"
            f"<b>Equipment:</b><br>{cfg['equipment']}<br><br>"
            f"<b>Volume:</b> {cfg['volume']}<br><br>"
            f"<b>Risk Level:</b> <span style='color:{risk_color};font-weight:bold'>{cfg['risk']}</span><br><br>"
            f"<b>Notes:</b><br><small>{cfg['notes']}</small>"
            f"</div>", unsafe_allow_html=True,
        )

@st.fragment(run_every=2)
def render_site_live(hid: str):
    cfg = HOSPITALS[hid]
    hst = st.session_state.hospitals[hid]
    pht_s  = hst["pht"]["status"]
    adw_s  = hst["adwin"]["status"]
    status = combined_status(pht_s, adw_s)

    st.markdown(f"#### {status_icon(status)} Status: **{status}**")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Images",   hst["img_count"])
    m2.metric("Max Conf", f"{hst['last_conf']:.3f}" if hst["last_conf"] else "—")
    m3.metric(f"PHT {status_icon(pht_s)}", f"{hst['pht']['pht_val']:.2f}",
              help=f"λ={hst['pht_lambda']} (alarm threshold)")
    m4.metric(f"ADWIN {status_icon(adw_s)}", f"{hst['adwin']['gap']:.3f}")
    m5.metric("Latency",  f"{hst['latency_ms']}ms" if hst["latency_ms"] else "—")
    st.divider()

    col_img, col_pred = st.columns([1, 2])
    with col_img:
        st.markdown("**Latest received image**")
        if hst["last_img_bytes"]:
            st.image(hst["last_img_bytes"],
                     caption=f"#{hst['img_count']} · {cfg['perturb']}",
                     use_container_width=True)
            if cfg["perturb"] != "none":
                st.caption(f"⚠️ Live perturbation active: `{cfg['perturb']}`")
        else:
            st.markdown("_No image received yet_")
    with col_pred:
        st.markdown("**Per-class probabilities vs thresholds**")
        if hst["calibrated_thr"]:
            thr_label = "🔧 Auto-recalibrated thresholds (percentile-70)"
            thr_vals  = [hst["calibrated_thr"].get(n, 0.5) for n in LABEL_NAMES]
        else:
            thr_label = "Default thresholds (0.5)"
            thr_vals  = [0.5] * 14
        st.caption(f"🎯 {thr_label}")
        if hst["last_probs"]:
            values = [hst["last_probs"].get(n, 0) for n in LABEL_NAMES]
            colors = ["#e74c3c" if v >= t else "#3498db" for v, t in zip(values, thr_vals)]
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Bar(x=LABEL_NAMES, y=values,
                                      marker_color=colors, name="Probability"))
            fig_pred.add_trace(go.Scatter(
                x=LABEL_NAMES, y=thr_vals, mode="lines+markers",
                line=dict(color="#f39c12", dash="dash", width=1.5),
                name="Threshold", marker=dict(size=5),
            ))
            fig_pred.update_layout(
                height=280, margin=dict(l=20, r=20, t=10, b=80),
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font=dict(color="white", size=10),
                xaxis_tickangle=-45, legend=dict(orientation="h"),
                yaxis=dict(range=[0, 1]), uirevision=f"pred_{hid}",
            )
            st.plotly_chart(fig_pred, key=f"pred_bar_{hid}", use_container_width=True)
        else:
            st.markdown("_No predictions yet_")
    st.divider()

    chart_col, radar_col = st.columns([3, 1])
    with chart_col:
        st.markdown("**Drift detector history — this site**")
        if hst["smooth_history"]:
            fig_site = make_subplots(rows=2, cols=1,
                subplot_titles=("Smoothed Uncertainty Signal (1 − max_conf)",
                                f"PHT Statistic (alarm > {hst['pht_lambda']})"),
                vertical_spacing=0.22)
            xs = list(range(1, len(hst["smooth_history"]) + 1))
            fig_site.add_trace(go.Scatter(
                x=xs, y=hst["smooth_history"], mode="lines+markers",
                name="Smooth Error", line=dict(color=cfg["color"], width=2),
                marker=dict(size=4),
            ), row=1, col=1)
            if hst["warmup_errors"]:
                baseline = sum(hst["warmup_errors"]) / len(hst["warmup_errors"])
                fig_site.add_hline(y=baseline, line_dash="dash", line_color="gray",
                                   row=1, col=1, annotation_text="warmup baseline",
                                   annotation_font_color="gray")
            xs_pht = list(range(1, len(hst["pht_history"]) + 1))
            fig_site.add_trace(go.Scatter(
                x=xs_pht, y=hst["pht_history"], mode="lines",
                name="PHT", line=dict(color=cfg["color"], width=2),
            ), row=2, col=1)
            fig_site.add_hline(y=hst["pht_lambda"], line_color="#e74c3c",
                               line_dash="solid", row=2, col=1,
                               annotation_text="alarm", annotation_font_color="#e74c3c")
            fig_site.add_hline(y=hst["pht_lambda"] * 0.5, line_color="#f39c12",
                               line_dash="dash", row=2, col=1,
                               annotation_text="warning", annotation_font_color="#f39c12")
            for i, v in enumerate(hst["pht_history"]):
                if v > hst["pht_lambda"]:
                    fig_site.add_vline(x=i+1, line_color="#e74c3c",
                                       line_dash="dot", row=2, col=1)
                    break
            fig_site.update_layout(
                height=360, margin=dict(l=50, r=20, t=40, b=20),
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font=dict(color="white"), showlegend=False,
                uirevision=f"site_{hid}",
            )
            fig_site.update_xaxes(gridcolor="#2d2d2d", title_text="Image #")
            fig_site.update_yaxes(gridcolor="#2d2d2d")
            st.plotly_chart(fig_site, key=f"site_chart_{hid}", use_container_width=True)
        else:
            st.info("No data yet for this site.")
    with radar_col:
        st.markdown("**Drift Fingerprint**")
        st.plotly_chart(render_drift_fingerprint(hid, cfg["color"]),
                        key=f"radar_inspector_{hid}", use_container_width=True)
    st.divider()

@st.fragment(run_every=5)
def render_action_panel(hid: str):
    # ── Action panel ───────────────────────────────────────────────────────────
    cfg = HOSPITALS[hid]                              # ← add this
    hst = st.session_state.hospitals[hid]             # ← add this
    st.markdown("### ⚡ Operator Actions")
    st.caption("All actions logged to MLflow under `mlops-monitoring`.")

    col_a, col_b, col_c, col_d = st.columns(4)

    # ── Inject Drift ──────────────────────────────────────────────────────────
    with col_a:
        st.markdown("**💉 Inject Drift**")
        st.caption("Force PHT above alarm threshold for demo purposes.")
        if st.button(f"Inject Drift — Hosp {hid}", key=f"inject_{hid}",
                     type="primary", disabled=not hst["warmup_done"]):
            # Directly push PHT state above lambda to trigger the alarm
            lam = hst["pht_lambda"]
            hst["pht"]["M"]       = lam + 1.0
            hst["pht"]["U"]       = 0.0
            hst["pht"]["pht_val"] = round(lam + 1.0, 4)
            hst["pht"]["status"]  = "DRIFT"
            ts = datetime.now().strftime("%H:%M:%S")
            st.session_state.event_log.insert(
                0, f"[{ts}] 💉 Hosp {hid} DRIFT INJECTED (demo)")
            # Trigger the same pipeline as natural drift
            key_pht = f"{hid}_pht"
            if key_pht not in st.session_state.drift_logged:
                log_drift_mlflow(hid, hst["pht"]["pht_val"], hst["adwin"]["gap"],
                                 hst["img_count"], cfg["drift_type"], "INJECTED")
                st.session_state.drift_logged.add(key_pht)
                auto_recalibrate_on_drift(
                    hid, "INJECTED", hst["pht"]["pht_val"],
                    hst["adwin"]["gap"], cfg["drift_type"]
                )
            st.success("✓ Drift injected — auto-recalibration fired")

    # ── Manual Recalibrate ────────────────────────────────────────────────────
    with col_b:
        st.markdown("**🎯 Recalibrate**")
        st.caption("Manual threshold recalibration (operator-initiated).")
        if st.button(f"Recalibrate — Hosp {hid}", key=f"recal_{hid}",
                     disabled=len(hst["recent_probs"]) < 10):
            new_thr = recalibrate_thresholds(hid)
            hst["calibrated_thr"] = new_thr
            version = register_threshold_version(
                hid, new_thr, trigger="manual",
                pht_val=hst["pht"]["pht_val"],
                adwin_gap=hst["adwin"]["gap"]
            )
            hst["staging_version"] = version
            log_action_mlflow(hid, "threshold_recalibrated", {
                "n_samples": len(hst["recent_probs"]), "method": "percentile_70",
                "registry_version": version or "failed",
            })
            st.session_state.actions_taken.append({
                "ts": datetime.now().isoformat(), "hospital": hid,
                "action": "manual_recalibrated",
                "n_samples": len(hst["recent_probs"]),
                "registry_version": version or "—",
            })
            ver_str = f"→ registry v{version} staging" if version else ""
            st.success(f"✓ Thresholds recalibrated {ver_str}")
            if new_thr:
                with st.expander("New thresholds (vs default 0.5)"):
                    df_thr = pd.DataFrame({
                        "Pathology":     list(new_thr.keys()),
                        "New Threshold": list(new_thr.values()),
                        "Default":       [0.5] * len(new_thr),
                        "Δ":             [round(v - 0.5, 3) for v in new_thr.values()],
                    })
                    st.dataframe(df_thr, hide_index=True)

    # ── Promote to Production ─────────────────────────────────────────────────
    with col_c:
        st.markdown("**🚀 Promote to Production**")
        staging_ver = hst.get("staging_version")
        if staging_ver:
            st.caption(f"v{staging_ver} is in staging. Review thresholds above, then promote.")
            if st.button(f"Promote v{staging_ver} — Hosp {hid}",
                         key=f"promote_{hid}", type="primary"):
                ok = promote_to_production(staging_ver)
                if ok:
                    log_action_mlflow(hid, "promoted_to_production", {
                        "version": staging_ver, "hospital": hid,
                    })
                    st.session_state.actions_taken.append({
                        "ts": datetime.now().isoformat(), "hospital": hid,
                        "action": "promoted_to_production",
                        "version": staging_ver,
                    })
                    ts = datetime.now().strftime("%H:%M:%S")
                    st.session_state.event_log.insert(
                        0, f"[{ts}] 🚀 Hosp {hid} v{staging_ver} PROMOTED to production")
                    st.success(f"✓ v{staging_ver} promoted to production alias")
                    st.info("ℹ️ In production, the API would reload this version on next restart.")
                else:
                    st.error("Registry unavailable — check MLflow DB")
        else:
            st.caption("No staging version yet. Drift must be detected first.")
            st.button("Promote to Production", key=f"promote_{hid}", disabled=True)

    # ── Reset Detector ────────────────────────────────────────────────────────
    with col_d:
        st.markdown("**🔄 Reset Detector**")
        st.caption("Reset PHT/ADWIN after maintenance or model update.")
        if st.button(f"Reset Detector — Hosp {hid}", key=f"reset_det_{hid}",
                     disabled=hst["img_count"] == 0):
            lam = hst["pht_lambda"]
            hst["pht"]            = fresh_pht()
            hst["adwin"]          = fresh_adwin()
            hst["raw_error_hist"] = []
            hst["warmup_done"]    = False
            hst["warmup_errors"]  = []
            hst["staging_version"] = None
            st.session_state.drift_logged.discard(f"{hid}_pht")
            st.session_state.drift_logged.discard(f"{hid}_adwin")
            st.session_state.drift_logged.discard(f"{hid}_autorecal")
            hst["pht_lambda"] = lam
            log_action_mlflow(hid, "detector_reset", {"reason": "manual"})
            st.session_state.actions_taken.append({
                "ts": datetime.now().isoformat(),
                "hospital": hid, "action": "detector_reset",
            })
            st.success(f"✓ Detectors reset for Hospital {hid}")

def render_site_inspector():
    col_sel, col_info = st.columns([1, 3])
    with col_sel:
        hid = st.selectbox(
            "Select hospital site",
            options=list(HOSPITALS.keys()),
            format_func=lambda h: HOSPITALS[h]["label"],
            key="inspector_hid",
        )
    cfg = HOSPITALS[hid]
    with col_info:
        st.markdown(
            f"<div style='border-left:5px solid {cfg['color']};padding-left:12px'>"
            f"<h3>{cfg['label']}</h3>"
            f"<p style='color:gray'>{cfg['subtitle']} · {cfg['drift_type']}</p>"
            f"</div>", unsafe_allow_html=True,
        )
    render_site_static(hid)
    st.divider()
    render_site_live(hid)
    render_action_panel(hid)  # ← add this

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Audit Log
# ═══════════════════════════════════════════════════════════════════════════════
@st.fragment(run_every=2)
def render_audit_log():
    st.markdown("### 📋 Inference Log")
    records = []
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if "hospital" in r:
                        records.append(r)
                except Exception:
                    pass
    if records:
        df_log = pd.DataFrame(records)
        df_log["ts"] = pd.to_datetime(df_log["ts"])
        df_log = df_log.sort_values("ts", ascending=False)
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            hosp_filter = st.multiselect("Hospital", ["A","B","C","D"],
                                         default=["A","B","C","D"], key="audit_hosp")
        with col_f2:
            status_filter = st.multiselect("PHT Status",
                                           ["STABLE","WARNING","DRIFT"],
                                           default=["WARNING","DRIFT"], key="audit_status")
        with col_f3:
            n_rows = st.slider("Rows", 10, 500, 100, key="audit_rows")
        mask = df_log["hospital"].isin(hosp_filter)
        if status_filter:
            mask &= df_log["pht_status"].isin(status_filter)
        df_show = df_log[mask].head(n_rows)
        def color_pht(val):
            return {"DRIFT":   "background-color:#ff4b4b;color:white",
                    "WARNING": "background-color:#ffa500", "STABLE": ""}.get(val, "")
        st.dataframe(
            df_show[["ts","hospital","image","max_conf","latency_ms",
                      "pht_val","pht_status","adwin_status","positives"]]
            .style.map(color_pht, subset=["pht_status"]),
            use_container_width=True, hide_index=True,
        )
        st.caption(f"{len(df_log):,} total records · {len(df_show)} shown")
    else:
        st.info("No inference records yet.")
    st.divider()

    st.markdown("### ⚡ Operator Actions Log")
    if st.session_state.actions_taken:
        st.dataframe(pd.DataFrame(st.session_state.actions_taken),
                     use_container_width=True, hide_index=True)
    else:
        st.info("No operator actions taken yet.")
    st.divider()

    st.markdown("### 🔬 MLflow Drift Events")
    try:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{ROOT}/experiments/mlflow.db")
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name("mlops-monitoring")
        if exp:
            runs = client.search_runs(experiment_ids=[exp.experiment_id],
                                      order_by=["start_time DESC"], max_results=50)
            if runs:
                rows = [{"Run": r.info.run_name,
                    "Hospital":   r.data.tags.get("hospital_id","—"),
                    "Detector":   r.data.tags.get("detector","—"),
                    "Event":      r.data.tags.get("event","—"),
                    "Drift Type": r.data.tags.get("drift_type","—"),
                    "PHT":        str(round(r.data.metrics["pht_value"], 4)) if "pht_value" in r.data.metrics else "—",
                    "Images":     str(int(r.data.metrics["image_count"])) if "image_count" in r.data.metrics else "—",
                    "Run ID":     r.info.run_id[:8]} for r in runs]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("No drift events logged yet.")
        else:
            st.info("No `mlops-monitoring` experiment yet.")
    except Exception as e:
        st.warning(f"MLflow unavailable: {e}")
    st.divider()

    st.markdown("### 🏷️ Model Registry — Version History")
    try:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{ROOT}/experiments/mlflow.db")
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if versions:
            # Get current aliases
            try:
                prod_mv    = client.get_model_version_by_alias(MODEL_NAME, "production")
                prod_ver   = prod_mv.version
            except Exception:
                prod_ver   = None
            try:
                stage_mv   = client.get_model_version_by_alias(MODEL_NAME, "staging")
                stage_ver  = stage_mv.version
            except Exception:
                stage_ver  = None

            reg_rows = []
            for v in sorted(versions, key=lambda x: int(x.version), reverse=True):
                alias = ""
                if v.version == prod_ver:
                    alias = "🟢 production"
                elif v.version == stage_ver:
                    alias = "🟡 staging"
                reg_rows.append({
                    "Version":     f"v{v.version}",
                    "Alias":       alias,
                    "Status":      v.status,
                    "Hospital":    v.tags.get("hospital", "—"),
                    "Trigger":     v.tags.get("trigger", "—"),
                    "Description": (v.description or "")[:60],
                    "Created":     datetime.fromtimestamp(
                        v.creation_timestamp / 1000).strftime("%Y-%m-%d %H:%M"),
                })
            st.dataframe(pd.DataFrame(reg_rows), use_container_width=True, hide_index=True)
        else:
            st.info(f"No versions registered for `{MODEL_NAME}` yet.")
    except Exception as e:
        st.warning(f"Registry unavailable: {e}")

    st.caption("💡 Full MLflow UI: `mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db`")

# ═══════════════════════════════════════════════════════════════════════════════
# TICK FRAGMENTS
# ═══════════════════════════════════════════════════════════════════════════════
def make_tick_fragment(hid: str):
    @st.fragment(run_every=1)
    def _fragment():
        if st.session_state.running:
            do_tick(hid)
        st.html("<span style='display:none'></span>")
    return _fragment

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(page_title="ChestMNIST MLOps Monitor",
                       page_icon="🏥", layout="wide")
    init_state()

    if "tick_fragments" not in st.session_state:
        st.session_state.tick_fragments = {
            hid: make_tick_fragment(hid) for hid in HOSPITALS
        }
    for hid in HOSPITALS:
        st.session_state.tick_fragments[hid]()

    col_title, col_ctrl = st.columns([3, 1])
    with col_title:
        st.markdown("## 🏥 ChestMNIST Multi-Site MLOps Monitor")
        st.caption(
            "PHT (O(1)) + ADWIN (O(log n)) · Drift Fingerprint radar (novel) · "
            "Live perturbation · Auto-recalibration · MLflow Registry · "
            f"Email alerts {'✅' if EMAIL_ENABLED else '⚠️ disabled (set env vars)'} · "
            "F6-Score — ESI Algiers 2026"
        )
    with col_ctrl:
        api_ok = check_api_health()
        if api_ok:
            st.success("✅ API online")
        else:
            st.error("❌ API offline")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("▶", disabled=st.session_state.running or not api_ok,
                         help="Start", type="primary"):
                st.session_state.running = True
                st.rerun()
        with c2:
            if st.button("⏹", disabled=not st.session_state.running, help="Stop"):
                st.session_state.running = False
                st.rerun()
        with c3:
            if st.button("🔄", help="Reset all"):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()
    st.divider()

    tab1, tab2, tab3 = st.tabs([
        "🌐 Fleet Overview", "🔬 Site Inspector", "📋 Audit Log",
    ])
    with tab1:
        render_fleet_overview()
    with tab2:
        render_site_inspector()
    with tab3:
        render_audit_log()

if __name__ == "__main__":
    main()