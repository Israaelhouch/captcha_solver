import sys
from ultralytics import YOLO
from pathlib import Path
from src.utils.logger import setup_logger

logger = setup_logger()

from src.config import (
    DATA_YAML,
    MODEL_NAME,
    EPOCHS,
    IMAGE_SIZE,
    BATCH_SIZE,
    WEIGHTS_DIR,
    TRAIN_RESULTS_DIR
)

def train_model():
    """
    Train YOLOv8 model on the CAPTCHA digit dataset.
    """
    try:
        logger.info("Starting YOLOv8 training...")

        model = YOLO(MODEL_NAME)

        results = model.train(
            data=str(DATA_YAML),
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            project=str(TRAIN_RESULTS_DIR),
            name="captcha_yolov8",
            exist_ok=True
        )

        best_weight_src = Path(results.save_dir) / "weights" / "best.pt"
        last_weight_src = Path(results.save_dir) / "weights" / "last.pt"

        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        best_weight_dst = WEIGHTS_DIR / "best.pt"
        last_weight_dst = WEIGHTS_DIR / "last.pt"

        best_weight_dst.write_bytes(best_weight_src.read_bytes())
        last_weight_dst.write_bytes(last_weight_src.read_bytes())

        logger.info(f"Training completed! Best model saved to {best_weight_dst}")
        logger.info(f"Last model saved to {last_weight_dst}")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
