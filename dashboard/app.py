"""
dashboard/app.py  —  ChestMNIST Multi-Site MLOps Monitor
Fix: use sliding window mean confidence (smoothed signal) for PHT/ADWIN.
"""
import streamlit as st
import requests
import time, io, os, math
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image as PILImage, ImageEnhance
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT     = Path(__file__).parent.parent
IMG_ROOT = ROOT / "test_images"
API_URL  = os.environ.get("API_URL", "http://localhost:8000")
LOG_FILE = ROOT / "logs" / "inference_log.jsonl"
LOG_FILE.parent.mkdir(exist_ok=True)

HOSPITALS = {
    "A": {
        "label": "Hospital A", "subtitle": "Reference — clean scanner",
        "folder": "hospital_A", "drift_type": "None — P(X,Y) stable",
        "color": "#2ecc71", "interval": 2, "perturb": "none",
    },
    "B": {
        "label": "Hospital B", "subtitle": "Brightness degradation (incremental)",
        "folder": "hospital_B", "drift_type": "Covariate shift — incremental P(X)↓",
        "color": "#f39c12", "interval": 3, "perturb": "brightness",
    },
    "C": {
        "label": "Hospital C", "subtitle": "Resolution drop at img 20",
        "folder": "hospital_C", "drift_type": "Covariate shift — sudden (img≥20)",
        "color": "#e74c3c", "interval": 4, "perturb": "resolution",
    },
    "D": {
        "label": "Hospital D", "subtitle": "Rare-class + Gaussian noise",
        "folder": "hospital_D", "drift_type": "Label shift — gradual P(Y)↑ rare",
        "color": "#9b59b6", "interval": 5, "perturb": "noise",
    },
}

# ── PHT & ADWIN params — tuned for smoothed signal ────────────────────────────
PHT_DELTA    = 0.002   # smaller = more sensitive
PHT_LAMBDA   = 3.0     # alarm on smoothed signal (much lower)
PHT_WARNING  = 1.5
ADWIN_DELTA  = 0.002
SMOOTH_WIN   = 5       # sliding window size for smoothing

LABEL_NAMES = [
    'Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule',
    'Pneumonia','Pneumothorax','Consolidation','Edema','Emphysema',
    'Fibrosis','Pleural_Thickening','Hernia'
]

# ── Perturbation (live, grows with img_count) ─────────────────────────────────
def apply_perturbation(img_path: Path, perturb: str, img_count: int) -> bytes:
    img = PILImage.open(img_path).convert("RGB")
    if perturb == "brightness":
        factor = max(0.10, 1.0 - (img_count / 25) * 0.90)
        img = ImageEnhance.Brightness(img).enhance(factor)
    elif perturb == "resolution":
        if img_count >= 20:
            severity = min(img_count - 20, 20)
            low_res  = max(4, 28 - severity)
            img = img.resize((low_res, low_res), PILImage.BILINEAR).resize((224, 224), PILImage.NEAREST)
    elif perturb == "noise":
        arr   = np.array(img, dtype=np.float32)
        sigma = min(img_count * 3.5, 90)
        arr   = np.clip(arr + np.random.normal(0, sigma, arr.shape), 0, 255).astype(np.uint8)
        img   = PILImage.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ── Smoothed signal: sliding window mean of error_sig ────────────────────────
def get_smooth_signal(history: list, new_val: float, win: int = SMOOTH_WIN) -> float:
    """Returns mean of last `win` values including new_val."""
    recent = (history[-win+1:] if len(history) >= win else history) + [new_val]
    return sum(recent) / len(recent)

# ── PHT on smoothed signal ────────────────────────────────────────────────────
def pht_update(state: dict, x: float) -> dict:
    n    = state["n"] + 1
    mean = state["mean"] + (x - state["mean"]) / n
    U    = state["U"] + (x - mean - PHT_DELTA)
    M    = max(state["M"], U)
    val  = M - U
    status = "DRIFT" if val > PHT_LAMBDA else ("WARNING" if val > PHT_WARNING else "STABLE")
    return {"n": n, "mean": mean, "U": U, "M": M, "pht_val": round(val, 4), "status": status}

def fresh_pht():
    return {"n": 0, "mean": 0.0, "U": 0.0, "M": 0.0, "pht_val": 0.0, "status": "STABLE"}

# ── ADWIN (Hoeffding-based, slide 11) ────────────────────────────────────────
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

def status_icon(s):
    return {"STABLE": "🟢", "WARNING": "🟡", "DRIFT": "🔴"}[s]

def combined_status(a, b):
    rank = {"STABLE": 0, "WARNING": 1, "DRIFT": 2}
    return a if rank[a] >= rank[b] else b

# ── Session state ──────────────────────────────────────────────────────────────
def init_state():
    if "initialized" in st.session_state:
        return
    st.session_state.initialized  = True
    st.session_state.running      = False
    st.session_state.event_log    = []
    st.session_state.drift_logged = set()
    st.session_state.hospitals    = {}
    now = time.time()
    for i, (hid, cfg) in enumerate(HOSPITALS.items()):
        imgs = sorted((IMG_ROOT / cfg["folder"]).glob("*.png"))
        st.session_state.hospitals[hid] = {
            "images":            imgs,
            "img_idx":           0,
            "last_tick":         now - (i / len(HOSPITALS)) * cfg["interval"],
            "pht":               fresh_pht(),
            "adwin":             fresh_adwin(),
            "raw_error_hist":    [],   # raw error_sig history for smoothing
            "conf_history":      [],
            "pht_history":       [],
            "adwin_gap_history": [],
            "smooth_history":    [],   # smoothed signal shown in chart
            "img_count":         0,
            "last_conf":         None,
            "last_img_path":     None,
            "last_positives":    [],
            "latency_ms":        None,
            "last_error":        None,
        }

# ── Log ────────────────────────────────────────────────────────────────────────
def log_inference(hid, img_name, conf, positives, latency_ms,
                  pht_val, pht_status, adwin_status):
    import json
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

# ── MLflow drift log ──────────────────────────────────────────────────────────
def log_drift_mlflow(hid, pht_val, adwin_gap, img_count, drift_type, detector):
    try:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{ROOT}/experiments/mlflow.db")
        mlflow.set_experiment("mlops-chestmnist")
        with mlflow.start_run(run_name=f"drift_{detector}_hospital_{hid}"):
            mlflow.set_tag("hospital_id", hid)
            mlflow.set_tag("drift_type",  drift_type)
            mlflow.set_tag("detector",    detector)
            mlflow.set_tag("event",       "DRIFT_DETECTED")
            mlflow.log_param("pht_lambda",  PHT_LAMBDA)
            mlflow.log_param("pht_delta",   PHT_DELTA)
            mlflow.log_param("smooth_win",  SMOOTH_WIN)
            mlflow.log_metric("pht_value",   pht_val)
            mlflow.log_metric("adwin_gap",   adwin_gap)
            mlflow.log_metric("image_count", img_count)
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

        # ── Smooth the signal before feeding detectors ────────────────────────
        smooth_err = get_smooth_signal(hst["raw_error_hist"], raw_error)
        hst["raw_error_hist"].append(raw_error)
        if len(hst["raw_error_hist"]) > 200:
            hst["raw_error_hist"] = hst["raw_error_hist"][-200:]

        hst["pht"]   = pht_update(hst["pht"],   smooth_err)
        hst["adwin"] = adwin_update(hst["adwin"], smooth_err)

        hst["conf_history"].append(round(max_conf, 4))
        hst["pht_history"].append(hst["pht"]["pht_val"])
        hst["adwin_gap_history"].append(hst["adwin"]["gap"])
        hst["smooth_history"].append(round(smooth_err, 4))
        hst["img_count"]     += 1
        hst["img_idx"]       += 1
        hst["last_tick"]      = tick_done
        hst["last_conf"]      = round(max_conf, 4)
        hst["last_img_path"]  = img_path
        hst["last_positives"] = result.get("positives", [])
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
                   f"smooth_err={smooth_err:.3f}")
        else:
            msg = (f"[{ts}] ✅ Hosp {hid} #{hst['img_count']:02d} "
                   f"conf={max_conf:.3f} smooth_err={smooth_err:.3f} lat={latency_ms:.0f}ms")

        st.session_state.event_log.insert(0, msg)
        st.session_state.event_log = st.session_state.event_log[:80]
        return True
    except Exception as e:
        st.session_state.hospitals[hid]["last_error"] = str(e)
        return False

# ── Hospital fragment ─────────────────────────────────────────────────────────
def make_hospital_fragment(hid: str):
    cfg = HOSPITALS[hid]
    @st.fragment(run_every=1)
    def _fragment():
        if st.session_state.running:
            do_tick(hid)
        hst          = st.session_state.hospitals[hid]
        pht_status   = hst["pht"]["status"]
        adwin_status = hst["adwin"]["status"]
        status       = combined_status(pht_status, adwin_status)

        st.markdown(
            f"<div style='border-left:4px solid {cfg['color']};padding-left:8px'>"
            f"<b>{cfg['label']}</b><br>"
            f"<small style='color:gray'>{cfg['subtitle']}</small></div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"{status_icon(status)} **{status}**")
        if hst.get("last_error"):
            st.caption(f"⚠️ _{hst['last_error']}_")

        m1, m2 = st.columns(2)
        m1.metric("Images", hst["img_count"])
        m2.metric("Conf",   f"{hst['last_conf']:.3f}" if hst["last_conf"] else "—")

        m3, m4 = st.columns(2)
        m3.metric(f"PHT {status_icon(pht_status)}",
                  f"{hst['pht']['pht_val']:.2f}",
                  help=f"Alarm > λ={PHT_LAMBDA}")
        m4.metric(f"ADWIN {status_icon(adwin_status)}",
                  f"{hst['adwin']['gap']:.3f}",
                  help="Hoeffding sub-window gap")

        st.metric("Latency", f"{hst['latency_ms']}ms" if hst["latency_ms"] else "—")
        st.caption(f"**Drift:** {cfg['drift_type']}")
        st.caption(f"**Interval:** {cfg['interval']}s · **Perturb:** `{cfg['perturb']}`")
        pos = hst["last_positives"]
        st.caption(
            "**Positives:** " + (", ".join(pos[:3]) + ("…" if len(pos) > 3 else ""))
            if pos else "**Positives:** No finding"
        )
        if hst["last_img_path"] and hst["last_img_path"].exists():
            try:
                perturbed = apply_perturbation(
                    hst["last_img_path"], cfg["perturb"], max(0, hst["img_count"] - 1)
                )
                st.image(perturbed,
                         caption=f"#{hst['img_count']} {hst['last_img_path'].name[:20]}",
                         width="stretch")
            except Exception:
                st.image(str(hst["last_img_path"]), width="stretch")
        else:
            st.markdown("_No image yet_")
    return _fragment

# ── Chart fragment ─────────────────────────────────────────────────────────────
@st.fragment(run_every=3)
def chart_fragment():
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "Smoothed Error Signal  1−conf  (5-image window mean)",
            f"PHT Statistic  M_T−U_T  (alarm > λ={PHT_LAMBDA})",
            "ADWIN Gap  |μ_W0−μ_W1|  (alarm when > ε_cut)",
        ),
        vertical_spacing=0.12,
    )
    max_x = 2
    for hid, cfg in HOSPITALS.items():
        hst   = st.session_state.hospitals[hid]
        smooth = hst["smooth_history"]
        pht    = hst["pht_history"]
        adw    = hst["adwin_gap_history"]
        xs     = list(range(1, len(smooth) + 1))
        max_x  = max(max_x, len(smooth))

        fig.add_trace(go.Scatter(x=xs, y=smooth, mode="lines+markers",
            name=cfg["label"], line=dict(color=cfg["color"], width=2),
            marker=dict(size=4), legendgroup=hid), row=1, col=1)
        fig.add_trace(go.Scatter(x=list(range(1, len(pht)+1)), y=pht,
            mode="lines", name=f"PHT-{hid}",
            line=dict(color=cfg["color"], width=2, dash="dot"),
            legendgroup=hid, showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=list(range(1, len(adw)+1)), y=adw,
            mode="lines", name=f"ADWIN-{hid}",
            line=dict(color=cfg["color"], width=2, dash="dash"),
            legendgroup=hid, showlegend=False), row=3, col=1)

    fig.add_trace(go.Scatter(x=[1, max_x], y=[PHT_WARNING]*2,
        mode="lines", line=dict(color="#f39c12", dash="dash", width=1.5),
        name=f"PHT Warning ({PHT_WARNING})"), row=2, col=1)
    fig.add_trace(go.Scatter(x=[1, max_x], y=[PHT_LAMBDA]*2,
        mode="lines", line=dict(color="#e74c3c", dash="solid", width=2),
        name=f"PHT Alarm ({PHT_LAMBDA})"), row=2, col=1)

    fig.update_layout(
        height=640,
        margin=dict(l=50, r=20, t=60, b=20),
        legend=dict(orientation="h", y=-0.10, font=dict(size=11)),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white", size=12),
        uirevision="stable",
    )
    fig.update_xaxes(gridcolor="#2d2d2d", title_text="Image #")
    fig.update_yaxes(gridcolor="#2d2d2d")
    st.plotly_chart(fig, use_container_width=True)

# ── Event log fragment ─────────────────────────────────────────────────────────
@st.fragment(run_every=1)
def log_fragment():
    st.markdown("**📋 Event Log**")
    text = "\n".join(st.session_state.event_log[:25]) or "Simulation not started."
    st.code(text, language=None)

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="ChestMNIST MLOps Monitor", page_icon="🏥", layout="wide")
    init_state()

    if "hospital_fragments" not in st.session_state:
        st.session_state.hospital_fragments = {
            hid: make_hospital_fragment(hid) for hid in HOSPITALS
        }

    col_title, col_model = st.columns([3, 1])
    with col_title:
        st.markdown("## 🏥 ChestMNIST Multi-Site MLOps Monitor")
        st.caption(
            "Live drift injection · PHT (O(1)) + ADWIN (O(log n)) · "
            "5-image smoothing window · 4 hospital profiles · F6-Score Team"
        )
    with col_model:
        st.markdown("**Serving:** MobileNetV2 · AUC=0.790")
        st.markdown("`POST /predict/mobilenet`")

    api_ok = check_api_health()
    c1, c2, c3, c4 = st.columns([2, 1, 1, 2])
    with c1:
        if api_ok:
            st.success("✅ API online — MobileNetV2 loaded")
        else:
            st.error("❌ API offline — run: `uvicorn src.api:app --port 8000`")
    with c2:
        if st.button("▶ Start", disabled=st.session_state.running or not api_ok,
                     use_container_width=True, type="primary"):
            st.session_state.running = True
            st.rerun()
    with c3:
        if st.button("⏹ Stop", disabled=not st.session_state.running,
                     use_container_width=True):
            st.session_state.running = False
            st.rerun()
    with c4:
        if st.button("🔄 Reset", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    st.divider()

    card_cols = st.columns(4)
    for i, hid in enumerate(HOSPITALS):
        with card_cols[i]:
            st.session_state.hospital_fragments[hid]()

    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.info("**PHT** O(1) memory\nFast on sudden drift\nSlow on gradual\nλ=3.0, δ=0.002")
    c2.warning("**ADWIN** O(log n)\nFast on both types\nHoeffding guarantees\nδ=0.002")
    c3.success("**Hosp B** brightness↓\nExpect ~20 imgs\nADWIN fires first")
    c4.error("**Hosp C** sudden blur\nExpect img 20+\nBoth fire quickly")

    chart_fragment()

    with st.expander("ℹ️ Methodology — live drift injection + smoothing"):
        st.markdown(f"""
**Signal pipeline:** raw image → live perturbation → API (MobileNetV2) → max_conf → error_sig = 1−conf → 5-image sliding window mean → PHT + ADWIN

**Why smoothing?** Raw per-image confidence varies ±0.3 due to natural image variance (multi-label, imbalanced).
Smoothing reduces noise so PHT/ADWIN accumulate genuine drift signal rather than per-image fluctuation.

| Hospital | Perturbation | Type | Mechanism |
|---|---|---|---|
| A | None | Reference | PHT/ADWIN should stay STABLE |
| B | Brightness ↓ | Covariate incremental | factor = max(0.10, 1.0 − img/25 × 0.90) |
| C | Resolution ↓ | Covariate sudden | 28→4px NEAREST after img 20 |
| D | Gaussian noise ↑ | Label + covariate | σ = min(img × 3.5, 90) |

**PHT params:** λ={PHT_LAMBDA} (alarm), warning={PHT_WARNING}, δ={PHT_DELTA}
**ADWIN params:** δ={ADWIN_DELTA} (Hoeffding confidence level)
        """)

    log_fragment()

if __name__ == "__main__":
    main()