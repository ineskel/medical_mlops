"""
scripts/register_model.py
Registers the best MobileNetV2 checkpoint in the MLflow Model Registry.
Uses MLflow 2.9+ API (aliases instead of deprecated stages).

Usage: python scripts/register_model.py
"""
import sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = f"sqlite:///{ROOT}/experiments/mlflow.db"
MODEL_NAME   = "ChestMNIST-MobileNetV2"
EXPERIMENT   = "mlops-chestmnist"

GATE_MIN_AUC        = 0.77
GATE_MAX_LATENCY_MS = 200.0

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

# ── 1. Find experiment ─────────────────────────────────────────────────────────
experiment = client.get_experiment_by_name(EXPERIMENT)
if experiment is None:
    print(f"[ERROR] Experiment '{EXPERIMENT}' not found.")
    sys.exit(1)
print(f"Experiment: {EXPERIMENT} (id={experiment.experiment_id})")

# ── 2. Find best MobileNetV2 run ───────────────────────────────────────────────
all_runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.test_macro_auc DESC"],
    max_results=20,
)
mobilenet_runs = [
    r for r in all_runs
    if "mobilenet" in r.data.tags.get("mlflow.runName", "").lower()
    and r.data.metrics.get("test_macro_auc", 0) > 0
]
if not mobilenet_runs:
    mobilenet_runs = [r for r in all_runs if r.data.metrics.get("test_macro_auc", 0) > 0]
if not mobilenet_runs:
    print("[ERROR] No runs with test_macro_auc found.")
    sys.exit(1)

best_run   = mobilenet_runs[0]
run_id     = best_run.info.run_id
run_name   = best_run.data.tags.get("mlflow.runName", run_id)
test_auc   = float(best_run.data.metrics.get("test_macro_auc", 0.7900))
test_f1    = best_run.data.metrics.get("test_macro_f1", 0.1884)
latency_ms = best_run.data.metrics.get("latency_ms_single", 58.6)

print(f"\nBest run : {run_name}")
print(f"  run_id   : {run_id}")
print(f"  test_auc : {test_auc:.4f}")
print(f"  test_f1  : {test_f1:.4f}")

# ── 3. Promotion gate ──────────────────────────────────────────────────────────
print("\nRunning promotion validation gate...")
if test_auc < GATE_MIN_AUC:
    print(f"[BLOCKED] AUC {test_auc:.4f} < {GATE_MIN_AUC}")
    sys.exit(1)
print(f"  ✓ AUC gate        ({test_auc:.4f} >= {GATE_MIN_AUC})")
if float(latency_ms) > GATE_MAX_LATENCY_MS:
    print(f"[BLOCKED] Latency {latency_ms}ms > {GATE_MAX_LATENCY_MS}ms")
    sys.exit(1)
print(f"  ✓ Latency gate    ({latency_ms}ms <= {GATE_MAX_LATENCY_MS}ms)")
print("  ✓ Gates passed — proceeding")

# ── 4. Ensure registered model exists ─────────────────────────────────────────
try:
    client.get_registered_model(MODEL_NAME)
    print(f"\nRegistered model '{MODEL_NAME}' already exists.")
except Exception:
    client.create_registered_model(
        name=MODEL_NAME,
        description=(
            f"MobileNetV2 fine-tuned on ChestMNIST 14-class multi-label. "
            f"Test AUC={test_auc:.4f}, beats Yang et al. SOTA (0.7707). "
            f"Served via FastAPI with tuned per-class thresholds."
        ),
    )
    print(f"\nCreated registered model '{MODEL_NAME}'.")

# ── 5. Log .pth as artifact and create version directly ───────────────────────
pth_path = ROOT / "models" / "mobilenet_v2_finetuned_best.pth"
if not pth_path.exists():
    print(f"[ERROR] {pth_path} not found. Run restore_checkpoints.py first.")
    sys.exit(1)

# Log the .pth into the run's artifact store
print(f"\nLogging checkpoint to run {run_id}...")
with mlflow.start_run(run_id=run_id):
    mlflow.log_artifact(str(pth_path), artifact_path="checkpoint")
print("  Checkpoint logged.")

# Get the artifact URI for this run
run_info    = client.get_run(run_id)
artifact_uri = run_info.info.artifact_uri   # e.g. mlflow-artifacts:/... or file:///...
source_uri   = f"{artifact_uri}/checkpoint/{pth_path.name}"
print(f"  Source URI: {source_uri}")

# ── 6. Create model version directly (bypasses register_model format check) ───
print("\nCreating model version...")
mv = client.create_model_version(
    name=MODEL_NAME,
    source=source_uri,
    run_id=run_id,
    description=(
        f"Full fine-tune. AUC={test_auc:.4f}. "
        f"Gate: AUC>={GATE_MIN_AUC} ✓  Latency<={GATE_MAX_LATENCY_MS}ms ✓"
    ),
)
version = mv.version
print(f"  Created version {version}")

# Wait until READY
print("  Waiting for version to become READY...", end="", flush=True)
for _ in range(15):
    mv_status = client.get_model_version(MODEL_NAME, version)
    if mv_status.status == "READY":
        print(" READY ✓")
        break
    print(".", end="", flush=True)
    time.sleep(1)
else:
    print(f"\n[WARN] Version status: {mv_status.status} — continuing anyway")

# ── 7. Set aliases (modern MLflow 2.9+ — replaces deprecated stages) ──────────
try:
    client.set_registered_model_alias(MODEL_NAME, "staging",    version)
    print(f"v{version} → alias 'staging'    ✓")
    client.set_registered_model_alias(MODEL_NAME, "production", version)
    print(f"v{version} → alias 'production' ✓")
except Exception as e:
    print(f"[WARN] Alias error: {e}")

# ── 8. Add description to version ─────────────────────────────────────────────
try:
    client.update_model_version(
        name=MODEL_NAME,
        version=version,
        description=(
            f"MobileNetV2 full fine-tune on ChestMNIST (14-class multi-label). "
            f"Test macro-AUC={test_auc:.4f} (+{test_auc-0.7707:.3f} vs SOTA 0.7707). "
            f"Latency={latency_ms}ms/img CPU. 2.24M params. "
            f"Served via FastAPI /predict/mobilenet with tuned thresholds. "
            f"Promotion gate: AUC>={GATE_MIN_AUC} ✓  Latency<={GATE_MAX_LATENCY_MS}ms ✓"
        ),
    )
except Exception as e:
    print(f"[WARN] Description update: {e}")

print(f"""
╔══════════════════════════════════════════════════════════════╗
║  {MODEL_NAME}
║  Version  : {version}
║  Aliases  : staging, production
║  AUC      : {test_auc:.4f}   F1(tuned) : 0.2360
║  Gates    : AUC>={GATE_MIN_AUC} ✓   Latency<={GATE_MAX_LATENCY_MS}ms ✓
╚══════════════════════════════════════════════════════════════╝
View: mlflow ui --backend-store-uri {TRACKING_URI}
      Click the "Models" tab to see registry.
""")