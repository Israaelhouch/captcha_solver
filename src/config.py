from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "model"
RESULTS_DIR = MODEL_DIR / "results"
WEIGHTS_DIR = MODEL_DIR / "weights"

# YOLO dataset config file
DATA_YAML = DATA_DIR / "data.yaml"

# Default YOLO model configuration
MODEL_NAME = "yolov8n.pt"   # choose 'yolov8n' for small dataset or 'yolov8s' for larger

# Training parameters
EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 16
SEED = 42

# Output paths
TRAIN_RESULTS_DIR = RESULTS_DIR / "training"
EVAL_RESULTS_DIR = RESULTS_DIR / "evaluation"

