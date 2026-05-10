import mlflow
import os

# SQLite backend — works on Windows, no deprecation warning
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 
                          "..", "experiments", "mlflow.db"))

mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
mlflow.set_experiment("mlops-chestmnist")

with mlflow.start_run(run_name="setup_test"):
    mlflow.log_param("setup", "ok")
    mlflow.log_metric("dummy_auc", 0.0)
    print(f"MLflow DB: {DB_PATH}")
    print("MLflow setup works ✓")