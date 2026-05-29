# MLOps Pipeline for Medical Model Deployment

End-to-end MLOps pipeline for training, versioning, deploying, and **monitoring** a medical imaging model on **ChestMNIST** (14-class multi-label chest X-ray classification).

**Team:** F6-Score · ESI Algiers · Advanced Machine Learning 2025–2026  
**Stack:** PyTorch · MedMNIST · MLflow · FastAPI · Streamlit · Python 3.11

---

## Results Summary

| Model            | Strategy       | Test AUC   | F1 tuned | ms/img   |
| ---------------- | -------------- | ---------- | -------- | -------- |
| MobileNetV2      | Linear probe   | 0.6852     | —        | —        |
| ResNet18         | Linear probe   | 0.6865     | —        | —        |
| MobileNetV2      | Full fine-tune | **0.7900** | 0.2360   | **58.6** |
| ResNet18         | Full fine-tune | 0.7820     | 0.2331   | 87.3     |
| Yang et al. SOTA | Full fine-tune | 0.7707     | —        | —        |

Both fine-tuned models exceed SOTA. MobileNetV2 is deployed (higher AUC, 1.5× faster, 5× fewer params).

---

## Project Structure

```
medical_mlops/
├── data/                          ← ChestMNIST auto-downloaded (gitignored)
├── notebooks/
│   ├── W2_data_exploration.ipynb
│   ├── W2_baseline_experiments.ipynb
│   ├── W3_finetuning.ipynb
│   └── W3_evaluation.ipynb
├── src/
│   ├── api.py                     ← FastAPI serving (3 endpoints)
│   ├── config.py
│   ├── data_loader.py
│   └── mlflow_setup.py
├── dashboard/
│   └── app.py                     ← Streamlit monitoring dashboard (4 hospitals)
├── scripts/
│   ├── export_test_images.py      ← Exports hospital simulation images
│   ├── restore_checkpoints.py     ← Restores .pth from MLflow artifacts
│   └── register_model.py          ← Registers best model in MLflow registry
├── experiments/
│   └── mlflow.db                  ← All experiment runs (committed)
├── models/                        ← .pth checkpoints (gitignored, restore via script)
├── test_images/                   ← Hospital simulation images (gitignored)
│   ├── hospital_A/                ← Reference — clean scanner
│   ├── hospital_B/                ← Covariate shift — brightness degradation
│   ├── hospital_C/                ← Covariate shift — sudden resolution drop
│   └── hospital_D/                ← Label shift — rare class prevalence
├── logs/
│   └── inference_log.jsonl        ← Live inference log (gitignored)
├── figures/                       ← All W2+W3 evaluation plots
├── reports/
│   ├── W2_baseline_report.pdf
│   └── W3_experiments_summary.pdf
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Setup

```bash
# 1. Clone and create environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Restore model checkpoints from MLflow
python scripts/restore_checkpoints.py

# 4. Register best model in MLflow registry
python scripts/register_model.py

# 5. Generate hospital simulation images
python scripts/export_test_images.py
```

**Email alerts** (optional): create a `.env` file at project root:

```
ALERT_EMAIL_FROM=yourgmail@gmail.com
ALERT_EMAIL_TO=yourgmail@gmail.com
ALERT_EMAIL_PASSWORD=your-gmail-app-password
```

Use a [Gmail App Password](https://support.google.com/accounts/answer/185833), not your real password.

---

## Dataset

ChestMNIST — 14-class multi-label chest X-ray classification (MedMNIST benchmark).  
Train: 78,468 · Val: 11,219 · Test: 22,433 · Native resolution: 28×28 → resized to 224×224.

Downloaded automatically on first run:

```python
from medmnist import ChestMNIST
ds = ChestMNIST(split="test", download=True)
```

---

## Running the Full System

### 1 — MLflow experiment tracking

```bash
mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db --port 5000
# Open http://localhost:5000
```

### 2 — FastAPI serving

```bash
uvicorn src.api:app --port 8000
# Open http://localhost:8000/docs
```

Endpoints:

- `GET /health` — model load status
- `POST /predict/{model_name}` — inference with tuned thresholds (`mobilenet` or `resnet`)
- `POST /predict/ab` — randomised A/B routing between both models

> The API is also deployed on HuggingFace Spaces (containerised via Docker):  
> **https://yasmine0421-chestmnist-api.hf.space/docs**

### 3 — Monitoring dashboard

```bash
streamlit run dashboard/app.py
# Open http://localhost:8501
```

Simulates 4 hospitals with different acquisition profiles, streams images to the API, and runs **Page-Hinkley Test** (PHT) + **ADWIN** per hospital to detect drift in real time.

**On first DRIFT detection per hospital:**

1. Thresholds are **auto-recalibrated** (70th-percentile of recent predictions)
2. A new model version is **registered in the MLflow registry** under alias `staging`
3. An **email alert** is sent to the engineering team
4. The operator reviews the new thresholds in the dashboard
5. Operator clicks **Promote to Production** → alias moves to `production`

Operator actions (retraining requests, recalibration, detector resets, promotions) are all logged to the `mlops-monitoring` MLflow experiment.

## MLOps Practices Implemented

| Practice                         | Implementation                                                                                                                  |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Experiment tracking              | MLflow — params, per-epoch metrics, per-class AUC, artifacts                                                                    |
| Model versioning & registry      | MLflow Model Registry with `staging`/`production` aliases, promotion gates (AUC ≥ 0.77, latency ≤ 200ms)                        |
| Threshold versioning             | Auto-recalibrated thresholds registered as new model versions on drift                                                          |
| Reproducibility                  | Seed 42, pinned requirements.txt, official MedMNIST splits                                                                      |
| Serving                          | FastAPI with input validation, tuned thresholds, latency logging                                                                |
| A/B testing                      | `/predict/ab` endpoint — random routing between MobileNetV2 and ResNet18                                                        |
| Training-serving skew prevention | Identical preprocessing pipeline in training and API                                                                            |
| Drift detection                  | PHT (O(1)) + ADWIN (O(log n)) on live confidence stream, per-hospital                                                           |
| Auto-recalibration               | Threshold adaptation on drift without retraining — safe to automate                                                             |
| Human-in-the-loop                | Staging → production promotion requires operator approval                                                                       |
| Drift alerting                   | Email alert on first drift event per hospital via Gmail SMTP                                                                    |
| Drift logging                    | MLflow runs with `hospital_id`, `drift_type`, `detector`, `pht_value` tags                                                      |
| Inference logging                | JSONL log at `logs/inference_log.jsonl`                                                                                         |
| Novel contribution               | Drift Fingerprint — 5-axis radar chart per hospital (signal volatility, PHT slope, ADWIN gap, confidence trend, latency stress) |
| Containerisation                 | Dockerfile for API serving, deployed on HuggingFace Spaces                                                                      |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Training (Google Colab + GPU)                       │
│  W2: linear probe → W3: full fine-tune               │
│  MLflow tracking → experiments/mlflow.db             │
└────────────────────┬────────────────────────────────┘
                     │ .pth checkpoints + mlflow.db
┌────────────────────▼────────────────────────────────┐
│  Serving (HuggingFace Spaces — Docker)               │
│  FastAPI /predict/mobilenet                          │
│  MobileNetV2 — AUC 0.79, 58.6ms/img                 │
└────────────────────┬────────────────────────────────┘
                     │ HTTP inference calls
┌────────────────────▼────────────────────────────────┐
│  Monitoring (local — Streamlit)                      │
│  4 hospitals · PHT + ADWIN · Drift Fingerprint       │
│  Auto-recalibration → MLflow registry staging        │
│  Human promote → production · Email alerts           │
└─────────────────────────────────────────────────────┘
```

---

## References

- Yang et al., MedMNIST v2, Scientific Data 2023
- Sandler et al., MobileNetV2, CVPR 2018
- He et al., ResNet, CVPR 2016
- Sculley et al., Hidden Technical Debt in ML Systems, NeurIPS 2015
- Zaharia et al., MLflow, IEEE Data Engineering Bulletin 2018
- Bifet & Gavalda, ADWIN, SDM 2007
- Page, E.S., Continuous Inspection Schemes (PHT), Biometrika 1954
- Gama et al., DDM, 2004
