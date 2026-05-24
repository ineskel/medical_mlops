"""
dashboard/app.py  —  ChestMNIST Multi-Site MLOps Monitor
Real-time inference stream with Page-Hinkley drift detection.

Prerequisites:
    uvicorn src.api:app --port 8000   (terminal 1)
    streamlit run dashboard/app.py    (terminal 2)
"""

import streamlit as st
import requests
import time
import os
import random
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image as PILImage
import io

# ── Config ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
IMG_ROOT = ROOT / "test_images"
API_URL  = os.environ.get("API_URL", "http://localhost:8000")
LOG_FILE = ROOT / "logs" / "inference_log.jsonl"
LOG_FILE.parent.mkdir(exist_ok=True)

# ── Hospital definitions ───────────────────────────────────────────────────────
HOSPITALS = {
    "A": {
        "label":      "Hospital A",
        "subtitle":   "Reference — clean scanner",
        "folder":     "hospital_A",
        "drift_type": "None — P(X,Y) stable",
        "color":      "#2ecc71",
        "interval":   2,
    },
    "B": {
        "label":      "Hospital B",
        "subtitle":   "Brightness degradation",
        "folder":     "hospital_B",
        "drift_type": "Covariate shift — incremental P(X)↓",
        "color":      "#f39c12",
        "interval":   3,
    },
    "C": {
        "label":      "Hospital C",
        "subtitle":   "Resolution drop at img 30",
        "folder":     "hospital_C",
        "drift_type": "Covariate shift — sudden (img≥30)",
        "color":      "#e74c3c",
        "interval":   4,
    },
    "D": {
        "label":      "Hospital D",
        "subtitle":   "Rare-class oversampling",
        "folder":     "hospital_D",
        "drift_type": "Label shift — gradual P(Y)↑ rare",
        "color":      "#9b59b6",
        "interval":   5,
    },
}

# ── Page-Hinkley (from course slides, O(1) memory) ────────────────────────────
PHT_DELTA   = 0.005
PHT_LAMBDA  = 50.0
PHT_WARNING = 25.0

LABEL_NAMES = [
    'Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule',
    'Pneumonia','Pneumothorax','Consolidation','Edema','Emphysema',
    'Fibrosis','Pleural_Thickening','Hernia'
]

# ── PHT update ─────────────────────────────────────────────────────────────────
def pht_update(state: dict, x: float) -> dict:
    n    = state["n"] + 1
    mean = state["mean"] + (x - state["mean"]) / n
    U    = state["U"] + (x - mean - PHT_DELTA)
    M    = max(state["M"], U)
    val  = M - U
    status = "DRIFT" if val > PHT_LAMBDA else ("WARNING" if val > PHT_WARNING else "STABLE")
    return {"n": n, "mean": mean, "U": U, "M": M, "pht_val": round(val, 3), "status": status}

def fresh_pht():
    return {"n": 0, "mean": 0.0, "U": 0.0, "M": 0.0, "pht_val": 0.0, "status": "STABLE"}

def status_icon(status):
    return {"STABLE": "🟢", "WARNING": "🟡", "DRIFT": "🔴"}[status]

# ── Session state init ─────────────────────────────────────────────────────────
def init_state():
    if "initialized" in st.session_state:
        return
    st.session_state.initialized   = True
    st.session_state.running       = False
    st.session_state.event_log     = []
    st.session_state.drift_logged  = set()
    st.session_state.rerun_count   = 0

    now = time.time()
    st.session_state.hospitals = {}
    for i, (hid, cfg) in enumerate(HOSPITALS.items()):
        imgs = sorted((IMG_ROOT / cfg["folder"]).glob("*.png"))
        st.session_state.hospitals[hid] = {
            "images":        imgs,
            "img_idx":       0,
            "last_tick":     now - cfg["interval"] + i * 0.5,  # stagger start
            "pht":           fresh_pht(),
            "conf_history":  [],
            "pht_history":   [],
            "img_count":     0,
            "last_conf":     None,
            "last_img_path": None,
            "last_positives": [],
            "latency_ms":    None,
        }

# ── Inference log (JSONL) ─────────────────────────────────────────────────────
def log_inference(hid, img_name, conf, positives, latency_ms, pht_val, status):
    import json
    entry = {
        "ts":         datetime.now().isoformat(),
        "hospital":   hid,
        "image":      img_name,
        "max_conf":   round(conf, 4),
        "positives":  positives,
        "latency_ms": round(latency_ms, 2),
        "pht_val":    round(pht_val, 3),
        "status":     status,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

# ── API call ───────────────────────────────────────────────────────────────────
def call_api(img_path: Path) -> dict | None:
    try:
        with open(img_path, "rb") as f:
            resp = requests.post(
                f"{API_URL}/predict/mobilenet",
                files={"file": (img_path.name, f, "image/png")},
                params={"tuned_thresholds": "true"},
                timeout=10,
            )
        if resp.status_code == 200:
            return resp.json()
    except requests.exceptions.ConnectionError:
        pass
    return None

# ── MLflow drift logging ───────────────────────────────────────────────────────
def log_drift_mlflow(hid, pht_val, img_count, drift_type):
    try:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{ROOT}/experiments/mlflow.db")
        mlflow.set_experiment("mlops-chestmnist")
        with mlflow.start_run(run_name=f"drift_hospital_{hid}"):
            mlflow.set_tag("hospital_id",  hid)
            mlflow.set_tag("drift_type",   drift_type)
            mlflow.set_tag("event",        "DRIFT_DETECTED")
            mlflow.log_param("pht_lambda", PHT_LAMBDA)
            mlflow.log_param("pht_delta",  PHT_DELTA)
            mlflow.log_metric("pht_value",   pht_val)
            mlflow.log_metric("image_count", img_count)
    except Exception:
        pass

# ── Tick one hospital ──────────────────────────────────────────────────────────
def tick_hospital(hid: str, now: float) -> bool:
    cfg = HOSPITALS[hid]
    hst = st.session_state.hospitals[hid]

    if now - hst["last_tick"] < cfg["interval"]:
        return False
    if not hst["images"]:
        return False

    img_path = hst["images"][hst["img_idx"] % len(hst["images"])]
    result   = call_api(img_path)
    if result is None:
        return False

    probs      = [result["predictions"][lbl]["probability"] for lbl in LABEL_NAMES]
    max_conf   = max(probs)
    error_sig  = 1.0 - max_conf
    latency_ms = result.get("latency_ms", 0.0)

    hst["pht"]           = pht_update(hst["pht"], error_sig)
    hst["conf_history"].append(round(max_conf, 4))
    hst["pht_history"].append(hst["pht"]["pht_val"])
    hst["img_count"]    += 1
    hst["img_idx"]      += 1
    hst["last_tick"]     = now
    hst["last_conf"]     = round(max_conf, 4)
    hst["last_img_path"] = img_path
    hst["last_positives"] = result.get("positives", [])
    hst["latency_ms"]    = round(latency_ms, 1)

    status = hst["pht"]["status"]
    ts     = datetime.now().strftime("%H:%M:%S")

    log_inference(hid, img_path.name, max_conf,
                  hst["last_positives"], latency_ms,
                  hst["pht"]["pht_val"], status)

    if status == "DRIFT":
        msg = f"[{ts}] 🔴 Hosp {hid} DRIFT — PHT={hst['pht']['pht_val']:.1f} > λ={PHT_LAMBDA:.0f}"
        if hid not in st.session_state.drift_logged:
            log_drift_mlflow(hid, hst["pht"]["pht_val"],
                             hst["img_count"], cfg["drift_type"])
            st.session_state.drift_logged.add(hid)
    elif status == "WARNING":
        msg = f"[{ts}] ⚠️  Hosp {hid} WARNING — PHT={hst['pht']['pht_val']:.1f}"
    else:
        msg = f"[{ts}] ✅ Hosp {hid} #{hst['img_count']:02d} conf={max_conf:.3f} lat={latency_ms:.0f}ms"

    st.session_state.event_log.insert(0, msg)
    st.session_state.event_log = st.session_state.event_log[:80]
    return True

# ── Chart ──────────────────────────────────────────────────────────────────────
def build_chart():
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            "Max Confidence per Image  (↓ = model less certain = drift signal rising)",
            "Page-Hinkley Statistic  M_T − U_T  (drift alarm when > λ=50)",
        ),
        vertical_spacing=0.20,
    )
    max_x = 2
    for hid, cfg in HOSPITALS.items():
        hst  = st.session_state.hospitals[hid]
        conf = hst["conf_history"]
        pht  = hst["pht_history"]
        xs   = list(range(1, len(conf) + 1))
        max_x = max(max_x, len(conf))

        fig.add_trace(go.Scatter(
            x=xs, y=conf, mode="lines+markers",
            name=f"{cfg['label']} ({cfg['subtitle']})",
            line=dict(color=cfg["color"], width=2),
            marker=dict(size=5),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=list(range(1, len(pht) + 1)), y=pht,
            mode="lines",
            name=f"PHT-{hid}",
            line=dict(color=cfg["color"], width=2, dash="dot"),
            showlegend=False,
        ), row=2, col=1)

    for threshold, label, color, dash in [
        (PHT_WARNING, f"Warning zone (λ/2={PHT_WARNING:.0f})", "#f39c12", "dash"),
        (PHT_LAMBDA,  f"Drift alarm (λ={PHT_LAMBDA:.0f})",     "#e74c3c", "solid"),
    ]:
        fig.add_trace(go.Scatter(
            x=[1, max_x], y=[threshold, threshold],
            mode="lines",
            line=dict(color=color, dash=dash, width=1.5),
            name=label,
        ), row=2, col=1)

    fig.update_layout(
        height=520,
        margin=dict(l=50, r=20, t=60, b=20),
        legend=dict(orientation="h", y=-0.15, font=dict(size=11)),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white", size=12),
    )
    fig.update_xaxes(gridcolor="#2d2d2d", title_text="Image #")
    fig.update_yaxes(gridcolor="#2d2d2d")
    return fig

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="ChestMNIST MLOps Monitor",
        page_icon="🏥",
        layout="wide",
    )
    init_state()

    # ── Header ────────────────────────────────────────────────────────────────
    col_title, col_model = st.columns([3, 1])
    with col_title:
        st.markdown("## 🏥 ChestMNIST Multi-Site MLOps Monitor")
        st.caption(
            "Live inference stream · Page-Hinkley drift detection (δ=0.005, λ=50) · "
            "4 hospital profiles · MLflow drift logging · F6-Score Team"
        )
    with col_model:
        st.markdown("**Serving model**")
        st.markdown("MobileNetV2 · AUC=0.790 · 58.6ms/img")
        st.markdown("`POST /predict/mobilenet` (tuned thresholds)")

    # ── API status + controls ─────────────────────────────────────────────────
    try:
        health = requests.get(f"{API_URL}/health", timeout=2).json()
        api_ok = health.get("models", {}).get("mobilenet", False)
    except Exception:
        api_ok = False

    c1, c2, c3, c4 = st.columns([2, 1, 1, 2])
    with c1:
        if api_ok:
            st.success("✅ API online — MobileNetV2 + ResNet18 loaded")
        else:
            st.error("❌ API offline — run: `uvicorn src.api:app --port 8000`")
    with c2:
        start = st.button("▶ Start", disabled=st.session_state.running or not api_ok,
                          use_container_width=True, type="primary")
    with c3:
        stop = st.button("⏹ Stop", disabled=not st.session_state.running,
                         use_container_width=True)
    with c4:
        reset = st.button("🔄 Reset", use_container_width=True)

    if start:
        st.session_state.running = True
        st.session_state.drift_logged = set()
        st.rerun()
    if stop:
        st.session_state.running = False
        st.rerun()
    if reset:
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.divider()

    # ── Hospital cards (4 columns) ────────────────────────────────────────────
    card_cols = st.columns(4)
    img_placeholders = {}

    for i, (hid, cfg) in enumerate(HOSPITALS.items()):
        hst    = st.session_state.hospitals[hid]
        status = hst["pht"]["status"]
        icon   = status_icon(status)

        with card_cols[i]:
            # Status header
            st.markdown(
                f"<div style='border-left:4px solid {cfg['color']};padding-left:8px'>"
                f"<b>{cfg['label']}</b><br>"
                f"<small style='color:gray'>{cfg['subtitle']}</small>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"{icon} **{status}**")

            # Metrics row
            m1, m2 = st.columns(2)
            m1.metric("Images", hst["img_count"])
            m2.metric("PHT", f"{hst['pht']['pht_val']:.1f}")

            m3, m4 = st.columns(2)
            m3.metric("Conf", f"{hst['last_conf']:.3f}" if hst["last_conf"] else "—")
            m4.metric("Lat", f"{hst['latency_ms']}ms" if hst["latency_ms"] else "—")

            # Drift type badge
            st.caption(f"**Drift:** {cfg['drift_type']}")
            st.caption(f"**Interval:** {cfg['interval']}s/image")

            # Positives
            pos = hst["last_positives"]
            if pos:
                st.caption("**Positives:** " + ", ".join(pos[:3]) + ("…" if len(pos) > 3 else ""))
            else:
                st.caption("**Positives:** No finding")

            # Latest image
            img_placeholders[hid] = st.empty()
            if hst["last_img_path"] and hst["last_img_path"].exists():
                img_placeholders[hid].image(
                    str(hst["last_img_path"]),
                    caption=hst["last_img_path"].name[:30],
                    use_container_width=True,
                )
            else:
                img_placeholders[hid].markdown("_No image yet_")

    st.divider()

    # ── Live chart ────────────────────────────────────────────────────────────
    chart_ph = st.empty()
    chart_ph.plotly_chart(build_chart(), use_container_width=True)

    # ── Methodology expander ──────────────────────────────────────────────────
    with st.expander("ℹ️ Drift detection — Page-Hinkley Test (course slides §5)"):
        st.markdown(r"""
**Signal:** $x_t = 1 - \max_k p_k(y|x_t)$ — model uncertainty per image.
High uncertainty → $x_t$ rises → PHT statistic accumulates → alarm fires.

$$U_T = \sum_{t=1}^{T}(x_t - \bar{x}_T - \delta) \qquad M_T = \max_{1\leq t\leq T} U_t \qquad \text{Drift if } M_T - U_T > \lambda$$

| Param | Value | Role |
|---|---|---|
| δ | 0.005 | Min detectable magnitude |
| λ | 50 | Alarm threshold |
| Warning | 25 | Early-alert zone |

**Hospital profiles map to course taxonomy (slide 5):**
- **A** → reference P(X,Y) stable — PHT flat
- **B** → covariate shift P(X) incremental (brightness) — PHT rises slowly  
- **C** → covariate shift P(X) sudden at image 30 (resolution) — PHT spikes  
- **D** → label shift P(Y) gradual (rare class prevalence) — PHT rises gradually

Drift events are logged to MLflow with `hospital_id` and `drift_type` tags.
Inference requests are logged to `logs/inference_log.jsonl`.
        """)

    # ── Event log ─────────────────────────────────────────────────────────────
    st.markdown("**📋 Event Log**")
    log_ph = st.empty()
    log_text = "\n".join(st.session_state.event_log[:25]) or "Simulation not started."
    log_ph.code(log_text, language=None)

    # ── Simulation loop ───────────────────────────────────────────────────────
    if st.session_state.running:
        now     = time.time()
        ticked  = False
        for hid in HOSPITALS:
            if tick_hospital(hid, now):
                ticked = True

        # Update chart and log
        chart_ph.plotly_chart(build_chart(), use_container_width=True)
        log_ph.code(
            "\n".join(st.session_state.event_log[:25]) or "Waiting…",
            language=None,
        )

        # Update images
        for hid in HOSPITALS:
            hst = st.session_state.hospitals[hid]
            if hst["last_img_path"] and hst["last_img_path"].exists():
                img_placeholders[hid].image(
                    str(hst["last_img_path"]),
                    caption=hst["last_img_path"].name[:30],
                    use_container_width=True,
                )

        time.sleep(1)
        st.rerun()

if __name__ == "__main__":
    main()