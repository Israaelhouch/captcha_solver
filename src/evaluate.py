from ultralytics import YOLO
from pathlib import Path
from src.utils.logger import setup_logger
from src.config import WEIGHTS_DIR, RESULTS_DIR, DATA_YAML

logger = setup_logger()

EVAL_RESULTS_DIR = RESULTS_DIR / "evaluation"
EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(model_path: str = None, data_yaml: str = None):
    """
    Evaluate a trained YOLO model on the test dataset.

    Args:
        model_path (str or Path): Path to the trained model weights.
        data_yaml (str or Path): Path to the data.yaml file containing test/val info.
    """
    try:
        model_path = Path(model_path or WEIGHTS_DIR / "best.pt")
        data_yaml = Path(data_yaml or DATA_YAML)

        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found at {model_path}")
        if not data_yaml.exists():
            raise FileNotFoundError(f"Data YAML not found at {data_yaml}")

        logger.info(f"Loading YOLO model from: {model_path}")
        model = YOLO(str(model_path))

        logger.info("Evaluating model performance on test/val set...")
        # Use YAML directly for evaluation
        metrics = model.val(data=str(data_yaml))

        logger.info("Evaluation metrics:")
        for key, value in metrics.results_dict.items():
            logger.info(f"{key}: {value}")

        # Copy generated plots to evaluation folder
        for plot_name in ["confusion_matrix.png", "PR_curve.png", "P_curve.png", "R_curve.png"]:
            plot_path = Path(metrics.save_dir) / plot_name
            if plot_path.exists():
                dest_path = EVAL_RESULTS_DIR / plot_name
                dest_path.write_bytes(plot_path.read_bytes())
                logger.info(f"Saved plot: {dest_path}")

        logger.info("Evaluation completed!")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise
