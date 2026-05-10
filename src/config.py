# src/config.py

DATASET_NAME  = "chestmnist"
IMAGE_SIZE    = 28
NUM_CLASSES   = 14
BATCH_SIZE    = 64

LEARNING_RATE = 1e-3
NUM_EPOCHS    = 15
RANDOM_SEED   = 42

MLFLOW_EXPERIMENT_NAME = "mlops-chestmnist"
MODELS = ["mobilenet_v2", "resnet18"]