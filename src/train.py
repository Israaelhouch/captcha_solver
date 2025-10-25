import sys
import torch
import json
from ultralytics import YOLO
from pathlib import Path
from src.utils.logger import setup_logger
from src.config import (
    DATA_YAML,
    MODEL_NAME,
    EPOCHS,
    IMAGE_SIZE,
    BATCH_SIZE,
    WEIGHTS_DIR,
    TRAIN_RESULTS_DIR,
)

logger = setup_logger()

def train_model(
    data_yaml: Path = None,
    model_name: str = None,
    epochs: int = None,
    imgsz: int = None,
    batch: int = None,
):
    """
    Train YOLOv8 model on the CAPTCHA digit dataset with optional overrides.
    """

    from datetime import datetime

    try:
        data_yaml = data_yaml or DATA_YAML
        model_name = model_name or MODEL_NAME
        epochs = epochs or EPOCHS
        imgsz = imgsz or IMAGE_SIZE
        batch = batch or BATCH_SIZE

        logger.info(f"Starting YOLOv8 training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        logger.info(f"Data config: {data_yaml}")
        logger.info(f"Model: {model_name}")
        logger.info(f"Epochs: {epochs}, Image Size: {imgsz}, Batch: {batch}")

        # Load model
        resume_weight = WEIGHTS_DIR / "last.pt"
        if resume_weight.exists():
            logger.info(f"Resuming from previous checkpoint: {resume_weight}")
            model = YOLO(str(resume_weight))
        else:
            model = YOLO(model_name)

        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
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

        logger.info(f"Training completed!")
        logger.info(f"Best model saved to: {best_weight_dst}")
        logger.info(f"Last checkpoint saved to: {last_weight_dst}")

        # Save metadata
        summary = {
            "epochs": epochs,
            "batch": batch,
            "image_size": imgsz,
            "best_model": str(best_weight_dst),
            "train_dir": str(TRAIN_RESULTS_DIR),
        }
        (TRAIN_RESULTS_DIR / "training_summary.json").write_text(json.dumps(summary, indent=4))

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
