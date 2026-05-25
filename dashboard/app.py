"""
dashboard/app.py — ChestMNIST Multi-Site MLOps Monitor
Inspired by Arize AI / NannyML / CheXstray architecture.
Novel contribution: Drift Fingerprint radar chart per hospital.
3 tabs: Fleet Overview | Site Inspector | Audit Log

Prerequisites:
    uvicorn src.api:app --port 8000   (terminal 1)
    streamlit run dashboard/app.py    (terminal 2)
"""

import streamlit as st
import requests
import time, io, os, math, json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from PIL import Image as PILImage, ImageEnhance
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
IMG_ROOT = ROOT / "test_images"
API_URL  = os.environ.get("API_URL", "http://localhost:8000")
LOG_FILE = ROOT / "logs" / "inference_log.jsonl"
LOG_FILE.parent.mkdir(exist_ok=True)

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
        # Map + static info
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
        "drift_type":  "Covariate shift — sudden (img≥20)",
        "shift_class": "Covariate / Sudden",
        "color":       "#e74c3c",
        "interval":    4,
        "perturb":     "resolution",
        "lat": 36.3650, "lon": 6.6147,
        "city": "Constantine",
        "equipment": "Philips DigitalDiagnost (2009) — Very old, resolution issues",
        "volume":    "~40 scans/day",
        "notes":     "Oldest scanner in the network. Sudden firmware bug introduced "
                     "resolution degradation after image #20 in each batch cycle. "
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
WARMUP_IMGS = 10

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
        if img_count >= 20:
            severity = min(img_count - 20, 20)
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

# ── PHT (O(1), slide 8) ───────────────────────────────────────────────────────
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

# ── ADWIN (slide 11) ──────────────────────────────────────────────────────────
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

# ── Drift Fingerprint (novel contribution) ────────────────────────────────────
def compute_drift_fingerprint(hid: str) -> dict:
    """
    5-axis drift fingerprint — not in Arize/NannyML.
    Returns scores in [0,1] for radar chart.
    Axes:
      signal_volatility  — std of recent error signal (captures noise/instability)
      pht_slope          — rate of PHT growth (captures accumulation speed)
      adwin_gap          — normalized ADWIN gap (captures distribution gap)
      conf_trend         — downward trend in confidence (captures model degradation)
      latency_stress     — latency above baseline (captures compute load shift)
    """
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
        trend = conf_hist[0] - conf_hist[-1]   # positive = confidence dropped
        scores["Confidence\nTrend↓"] = min(1.0, max(0.0, trend / 0.4))
    else:
        scores["Confidence\nTrend↓"] = 0.0

    lat = hst["latency_ms"] or 0
    scores["Latency\nStress"] = min(1.0, max(0.0, (lat - 80) / 400))

    return scores

def render_drift_fingerprint(hid: str, color: str):
    """Radar / spider chart — one per hospital."""
    fp   = compute_drift_fingerprint(hid)
    cats = list(fp.keys())
    vals = list(fp.values())
    cats_closed = cats + [cats[0]]
    vals_closed = vals + [vals[0]]

    # Convert #rrggbb to rgba() for semi-transparent fill — Plotly requires this
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)
    fill_color = f"rgba({r},{g},{b},0.3)"

    fig = go.Figure(go.Scatterpolar(
        r=vals_closed,
        theta=cats_closed,
        fill="toself",
        fillcolor=fill_color,
        line=dict(color=color, width=2),
        name=f"Hosp {hid}",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickvals=[0.25, 0.5, 0.75, 1.0],
                ticktext=["Low", "Med", "High", "Max"],
                gridcolor="#333", linecolor="#333",
            ),
            angularaxis=dict(gridcolor="#333", linecolor="#444"),
            bgcolor="#0e1117",
        ),
        paper_bgcolor="#0e1117",
        font=dict(color="white", size=10),
        margin=dict(l=40, r=40, t=30, b=30),
        height=230,
        showlegend=False,
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
        imgs = sorted((IMG_ROOT / cfg["folder"]).glob("*.png"))
        lam  = PHT_LAMBDA * 2 if hid == "A" else PHT_LAMBDA
        st.session_state.hospitals[hid] = {
            "images":            imgs,
            "img_idx":           0,
            "last_tick":         now - (i / len(HOSPITALS)) * cfg["interval"],
            "pht":               fresh_pht(),
            "adwin":             fresh_adwin(),
            "pht_lambda":        lam,
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

# ── MLflow ─────────────────────────────────────────────────────────────────────
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
            mlflow.log_param("pht_lambda",  PHT_LAMBDA)
            mlflow.log_param("smooth_win",  SMOOTH_WIN)
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

        if img_count < WARMUP_IMGS:
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
            baseline          = sum(hst["warmup_errors"]) / len(hst["warmup_errors"])
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
# TAB 1 — Fleet Overview (auto-refreshes every 2s)
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

    # ── Fleet status table ─────────────────────────────────────────────────────
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
        })
    df = pd.DataFrame(rows)

    def color_status(val):
        return {"DRIFT":   "background-color:#ff4b4b;color:white",
                "WARNING": "background-color:#ffa500;color:black",
                "STABLE":  "background-color:#21c354;color:black"}.get(val, "")

    st.dataframe(df.style.map(color_status, subset=["Status"]),
                 use_container_width=True, hide_index=True)

    st.divider()

    # ── Confidence heatmap ─────────────────────────────────────────────────────
    st.markdown("### 📊 Confidence Heatmap — All Sites Over Time")
    max_len = max((len(st.session_state.hospitals[h]["conf_history"])
                   for h in HOSPITALS), default=1) or 1
    z = np.array([
        [v for v in st.session_state.hospitals[h]["conf_history"]] +
        [np.nan] * (max_len - len(st.session_state.hospitals[h]["conf_history"]))
        for h in HOSPITALS
    ], dtype=float)
    fig_heat = go.Figure(go.Heatmap(
        z=z,
        x=list(range(1, max_len + 1)),
        y=[f"Hosp {h}" for h in HOSPITALS],
        colorscale="RdYlGn", zmin=0.3, zmax=1.0,
        colorbar=dict(title="Max Conf"),
    ))
    fig_heat.update_layout(
        height=200, margin=dict(l=80, r=20, t=20, b=40),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white"), xaxis_title="Image #",
        uirevision="stable",
    )
    st.plotly_chart(fig_heat, use_container_width=True, key="heatmap_fleet")

    # ── PHT + ADWIN chart ──────────────────────────────────────────────────────
    st.markdown("### 📈 Drift Detector Signals — All Sites")

    with st.expander("ℹ️ How to read these charts"):
        st.markdown(f"""
**PHT chart (top):** The Page-Hinkley statistic M_T − U_T accumulates when the
model's uncertainty (1 − max_confidence) rises above its baseline.
- 🟢 Below {PHT_WARNING}: stable
- 🟡 Between {PHT_WARNING}–{PHT_LAMBDA}: warning zone
- 🔴 Above {PHT_LAMBDA}: drift alarm

**ADWIN chart (bottom):** The gap between the mean of the first and second halves
of a sliding window. When this gap exceeds the Hoeffding bound ε_cut, drift is detected.
ADWIN is faster than PHT on *gradual* drift (Hospital B, D).

**Why 0.5 threshold on error signal?** It's not 0.5 — it's adaptive.
The model outputs per-class probabilities; we use `1 − max(probabilities)` as the
uncertainty signal. The PHT baseline is calibrated from the first {WARMUP_IMGS} images
(warmup phase), so each hospital's detector starts from its own natural operating point.
Thresholds are *per-hospital*, not global.
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

    # ── Alert feed ─────────────────────────────────────────────────────────────
    st.markdown("### 🚨 Alert Feed")
    alerts = [e for e in st.session_state.event_log
              if "DRIFT" in e or "WARNING" in e or "✅" in e][:15]
    st.code("\n".join(alerts) if alerts else "No events yet. Start the simulation.",
            language=None)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Site Inspector (auto-refreshes every 2s)
# ═══════════════════════════════════════════════════════════════════════════════
# ── Replace render_site_inspector with this split version ─────────────────────

@st.fragment(run_every=None)   # only reruns when hospital selectbox changes
def render_site_static(hid: str):
    """Map + static hospital info — never auto-refreshes, only on hid change."""
    cfg = HOSPITALS[hid]

    st.markdown("### 🗺️ Site Location & Equipment Profile")
    map_col, info_col = st.columns([2, 1])

    with map_col:
        # Only the SELECTED hospital as a single marker
        fig_map = go.Figure(go.Scattermapbox(
            lat=[cfg["lat"]],
            lon=[cfg["lon"]],
            mode="markers+text",
            marker=dict(size=18, color=cfg["color"], opacity=0.95),
            text=[f"Hosp {hid}"],
            textposition="top right",
            hovertext=[
                f"<b>{cfg['label']}</b><br>"
                f"Risk: {cfg['risk']}<br>"
                f"Equipment: {cfg['equipment']}"
            ],
            hoverinfo="text",
        ))
        fig_map.update_layout(
            mapbox=dict(
                style="carto-darkmatter",
                center=dict(lat=cfg["lat"], lon=cfg["lon"]),
                zoom=7,
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=280,
            paper_bgcolor="#0e1117",
        )
        st.plotly_chart(fig_map, key=f"map_static_{hid}", use_container_width=True)

    with info_col:
        risk_color = RISK_COLOR.get(cfg["risk"], "#888")
        st.markdown(
            f"<div style='border:1px solid {cfg['color']};border-radius:8px;padding:14px'>"
            f"<b>📍 {cfg['city']}</b><br><br>"
            f"<b>Equipment:</b><br>{cfg['equipment']}<br><br>"
            f"<b>Volume:</b> {cfg['volume']}<br><br>"
            f"<b>Risk Level:</b> "
            f"<span style='color:{risk_color};font-weight:bold'>{cfg['risk']}</span><br><br>"
            f"<b>Notes:</b><br><small>{cfg['notes']}</small>"
            f"</div>",
            unsafe_allow_html=True,
        )


@st.fragment(run_every=2)   # live metrics refresh every 2s
def render_site_live(hid: str):
    """Live metrics, image, predictions, drift charts — refreshes every 2s."""
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
            thr_label = "Recalibrated thresholds (percentile-70)"
            thr_vals  = [hst["calibrated_thr"].get(n, 0.5) for n in LABEL_NAMES]
        else:
            thr_label = "Default thresholds (0.5) — click Recalibrate to adapt"
            thr_vals  = [0.5] * 14
        st.caption(f"🎯 {thr_label}")

        if hst["last_probs"]:
            values = [hst["last_probs"].get(n, 0) for n in LABEL_NAMES]
            colors = ["#e74c3c" if v >= t else "#3498db"
                      for v, t in zip(values, thr_vals)]
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
                yaxis=dict(range=[0, 1]),
                uirevision=f"pred_{hid}",
            )
            st.plotly_chart(fig_pred, key=f"pred_bar_{hid}",
                            use_container_width=True)
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
                fig_site.add_hline(y=baseline, line_dash="dash",
                                   line_color="gray", row=1, col=1,
                                   annotation_text="warmup baseline",
                                   annotation_font_color="gray")
            xs_pht = list(range(1, len(hst["pht_history"]) + 1))
            fig_site.add_trace(go.Scatter(
                x=xs_pht, y=hst["pht_history"], mode="lines",
                name="PHT", line=dict(color=cfg["color"], width=2),
            ), row=2, col=1)
            fig_site.add_hline(y=hst["pht_lambda"], line_color="#e74c3c",
                               line_dash="solid", row=2, col=1,
                               annotation_text="alarm",
                               annotation_font_color="#e74c3c")
            fig_site.add_hline(y=hst["pht_lambda"] * 0.5, line_color="#f39c12",
                               line_dash="dash", row=2, col=1,
                               annotation_text="warning",
                               annotation_font_color="#f39c12")
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
            st.plotly_chart(fig_site, key=f"site_chart_{hid}",
                            use_container_width=True)
        else:
            st.info("No data yet for this site.")

    with radar_col:
        st.markdown("**Drift Fingerprint**")
        st.plotly_chart(render_drift_fingerprint(hid, cfg["color"]),
                        key=f"radar_inspector_{hid}",
                        use_container_width=True)

    st.divider()

    # ── Action panel ───────────────────────────────────────────────────────────
    st.markdown("### ⚡ Operator Actions")
    st.caption("All actions logged to MLflow under `mlops-monitoring`.")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**🔁 Request Retraining**")
        st.caption("Log a retraining request to MLflow.")
        if st.button(f"Request Retraining — Hosp {hid}",
                     key=f"retrain_{hid}",
                     type="primary" if status == "DRIFT" else "secondary",
                     disabled=hst["img_count"] == 0):
            log_action_mlflow(hid, "retrain_requested", {
                "trigger": status, "pht_val": hst["pht"]["pht_val"],
                "adwin_gap": hst["adwin"]["gap"], "img_count": hst["img_count"],
                "drift_type": cfg["drift_type"],
            })
            st.session_state.actions_taken.append({
                "ts": datetime.now().isoformat(), "hospital": hid,
                "action": "retrain_requested", "trigger": status,
                "img_count": hst["img_count"],
            })
            st.success(f"✓ Retraining request logged (hospital={hid})")

    with col_b:
        st.markdown("**🎯 Recalibrate Thresholds**")
        st.caption(
            "Adapts per-class thresholds using the 70th percentile "
            "of recent prediction probabilities as pseudo-labels."
        )
        if st.button(f"Recalibrate — Hosp {hid}", key=f"recal_{hid}",
                     disabled=len(hst["recent_probs"]) < 10):
            new_thr = recalibrate_thresholds(hid)
            hst["calibrated_thr"] = new_thr
            log_action_mlflow(hid, "threshold_recalibrated", {
                "n_samples": len(hst["recent_probs"]), "method": "percentile_70"})
            st.session_state.actions_taken.append({
                "ts": datetime.now().isoformat(), "hospital": hid,
                "action": "threshold_recalibrated",
                "n_samples": len(hst["recent_probs"]),
            })
            st.success(f"✓ Thresholds recalibrated ({len(hst['recent_probs'])} samples)")
            if new_thr:
                with st.expander("New thresholds (vs default 0.5)"):
                    df_thr = pd.DataFrame({
                        "Pathology": list(new_thr.keys()),
                        "New Threshold": list(new_thr.values()),
                        "Default": [0.5] * len(new_thr),
                        "Δ": [round(v - 0.5, 3) for v in new_thr.values()],
                    })
                    st.dataframe(df_thr, hide_index=True)

    with col_c:
        st.markdown("**🔄 Reset Detector**")
        st.caption("Reset PHT/ADWIN for this site after maintenance or model update.")
        if st.button(f"Reset Detector — Hosp {hid}", key=f"reset_det_{hid}",
                     disabled=hst["img_count"] == 0):
            lam = hst["pht_lambda"]
            hst["pht"]            = fresh_pht()
            hst["adwin"]          = fresh_adwin()
            hst["raw_error_hist"] = []
            hst["warmup_done"]    = False
            hst["warmup_errors"]  = []
            st.session_state.drift_logged.discard(f"{hid}_pht")
            st.session_state.drift_logged.discard(f"{hid}_adwin")
            hst["pht_lambda"] = lam
            log_action_mlflow(hid, "detector_reset", {"reason": "manual"})
            st.session_state.actions_taken.append({
                "ts": datetime.now().isoformat(),
                "hospital": hid, "action": "detector_reset",
            })
            st.success(f"✓ Detectors reset for Hospital {hid}")


# Replace the old render_site_inspector with this orchestrator (not a fragment itself)
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

    # Static map — only reruns when hid changes (run_every=None + unique key)
    render_site_static(hid)

    st.divider()

    # Live metrics — reruns every 2s independently
    render_site_live(hid)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Audit Log (auto-refreshes every 5s)
# ═══════════════════════════════════════════════════════════════════════════════
@st.fragment(run_every=5)
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
                                         default=["A","B","C","D"],
                                         key="audit_hosp")
        with col_f2:
            status_filter = st.multiselect("PHT Status",
                                           ["STABLE","WARNING","DRIFT"],
                                           default=["WARNING","DRIFT"],
                                           key="audit_status")
        with col_f3:
            n_rows = st.slider("Rows", 10, 500, 100, key="audit_rows")

        mask = df_log["hospital"].isin(hosp_filter)
        if status_filter:
            mask &= df_log["pht_status"].isin(status_filter)
        df_show = df_log[mask].head(n_rows)

        def color_pht(val):
            return {"DRIFT":   "background-color:#ff4b4b;color:white",
                    "WARNING": "background-color:#ffa500",
                    "STABLE":  ""}.get(val, "")

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
        exp    = client.get_experiment_by_name("mlops-monitoring")
        if exp:
            runs = client.search_runs(experiment_ids=[exp.experiment_id],
                                      order_by=["start_time DESC"], max_results=50)
            if runs:
                rows = [{"Run": r.info.run_name,
                         "Hospital": r.data.tags.get("hospital_id","—"),
                         "Detector": r.data.tags.get("detector","—"),
                         "Event":    r.data.tags.get("event","—"),
                         "Drift Type": r.data.tags.get("drift_type","—"),
                         "PHT":      r.data.metrics.get("pht_value","—"),
                         "Images":   r.data.metrics.get("image_count","—"),
                         "Run ID":   r.info.run_id[:8]} for r in runs]
                st.dataframe(pd.DataFrame(rows),
                             use_container_width=True, hide_index=True)
            else:
                st.info("No drift events logged yet.")
        else:
            st.info("No `mlops-monitoring` experiment yet.")
    except Exception as e:
        st.warning(f"MLflow unavailable: {e}")
    st.caption("💡 Full MLflow UI: `mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db`")


# ═══════════════════════════════════════════════════════════════════════════════
# TICK FRAGMENTS (invisible, background, independent per hospital)
# ═══════════════════════════════════════════════════════════════════════════════
def make_tick_fragment(hid: str):
    @st.fragment(run_every=1)
    def _fragment():
        if st.session_state.running:
            do_tick(hid)
        # Emit a zero-width invisible HTML element — occupies no space
        st.html("<span style='display:none'></span>")
    return _fragment

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(page_title="ChestMNIST MLOps Monitor",
                       page_icon="🏥", layout="wide")
    init_state()

    # Background tick fragments (one per hospital, hidden via zero-height container)
    if "tick_fragments" not in st.session_state:
        st.session_state.tick_fragments = {
            hid: make_tick_fragment(hid) for hid in HOSPITALS
        }
    for hid in HOSPITALS:
        st.session_state.tick_fragments[hid]()

    # ── Header ─────────────────────────────────────────────────────────────────
    col_title, col_ctrl = st.columns([3, 1])
    with col_title:
        st.markdown("## 🏥 ChestMNIST Multi-Site MLOps Monitor")
        st.caption(
            "PHT (O(1)) + ADWIN (O(log n)) · Drift Fingerprint radar (novel) · "
            "Live perturbation · Adaptive recalibration · MLflow · "
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
        "🌐 Fleet Overview",
        "🔬 Site Inspector",
        "📋 Audit Log",
    ])
    with tab1:
        render_fleet_overview()
    with tab2:
        render_site_inspector()
    with tab3:
        render_audit_log()

if __name__ == "__main__":
    main()