# Topic 10 – MLOps Pipeline for Medical Model Deployment

End-to-end pipeline to train, version, deploy, and monitor a medical imaging model
on **ChestMNIST** (chest X-ray multi-label classification, 14 pathologies).  
Tools: **MLflow** · **FastAPI** · **PyTorch** · **MedMNIST**

---

## Project Structure

```
project/
├── data/                            ← ChestMNIST auto-downloaded here
├── notebooks/
│   ├── W2_data_exploration.ipynb
│   └── W2_baseline_experiments.ipynb
├── src/
│   ├── data_loader.py
│   ├── train.py
│   ├── evaluate.py
│   └── serve.py
├── experiments/
│   └── mlruns/                      ← MLflow runs saved here
├── models/                          ← saved checkpoints
├── reports/
├── figures/
├── config.py
├── requirements.txt
└── README.md
```

---

## Environment Setup

### 1. Create and activate the virtual environment

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset

ChestMNIST is part of the [MedMNIST](https://medmnist.com/) benchmark.  
**No manual download needed** — it is fetched automatically on first run:

```python
from src.data_loader import get_dataloaders
loaders = get_dataloaders(download=True, root="data")
```

Data is saved to `data/` and reused on subsequent runs.

---

## Using the DataLoader

```python
from src.data_loader import get_dataloaders

loaders = get_dataloaders(
    image_size  = 224,    # resize to 224×224 for pretrained backbones
    batch_size  = 32,
    num_workers = 4,      # set to 0 on Windows
    root        = "data",
)

for images, labels in loaders["train"]:
    # images : (B, 3, 224, 224) — grayscale converted to 3-channel
    # labels : (B, 14)          — multi-label binary
    ...

print(loaders["num_classes"])   # 14
print(loaders["label_names"])   # ['Atelectasis', 'Cardiomegaly', ...]
```

Sanity check from the command line:

```bash
python src/data_loader.py
# [train] batch shape : torch.Size([8, 3, 224, 224])
# [train] labels shape: torch.Size([8, 14])
```

---

## MLflow

Start the tracking server (runs saved in `experiments/mlruns/`):

```bash
mlflow ui --backend-store-uri experiments/mlruns --port 5000
```

Open http://localhost:5000.  
In any script, point MLflow to this directory:

```python
import mlflow
mlflow.set_tracking_uri("experiments/mlruns")
mlflow.set_experiment("chestmnist-baseline")
```

---

## FastAPI

Start the model serving API:

```bash
uvicorn src.serve:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs available at http://localhost:8000/docs.
