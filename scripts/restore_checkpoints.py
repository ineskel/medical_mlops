"""
Restores model checkpoints from MLflow artifact store.
Run this after cloning the repo instead of committing .pth files to Git.
Usage: python scripts/restore_checkpoints.py
"""
import mlflow, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

mlflow.set_tracking_uri(f"sqlite:///{ROOT}/experiments/mlflow.db")
client = mlflow.tracking.MlflowClient()

TARGETS = {
    "mobilenet_v2_finetuned": "mobilenet_v2_finetuned_best.pth",
    "resnet18_finetuned":     "resnet18_finetuned_best.pth",
}

os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)

runs = client.search_runs(
    experiment_ids=["2"],
    order_by=["metrics.test_macro_auc DESC"]
)

restored = []
for run in runs:
    artifacts = client.list_artifacts(run.info.run_id)
    for artifact in artifacts:
        if artifact.path.endswith(".pth"):
            dest = os.path.join(ROOT, "models", artifact.path)
            if os.path.exists(dest):
                print(f"[SKIP] {artifact.path} already exists")
                continue
            mlflow.artifacts.download_artifacts(
                run_id=run.info.run_id,
                artifact_path=artifact.path,
                dst_path=os.path.join(ROOT, "models")
            )
            print(f"[OK] Restored {artifact.path} from run {run.info.run_name}")
            restored.append(artifact.path)

if not restored:
    print("Nothing to restore — all checkpoints already present.")
print("Done ✓")