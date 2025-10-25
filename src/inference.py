from ultralytics import YOLO
import cv2
from pathlib import Path
from src.utils.logger import setup_logger
from src.config import WEIGHTS_DIR, RESULTS_DIR

logger = setup_logger()

PREDICTIONS_DIR = RESULTS_DIR / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


def sort_detections_by_x(detections):
    """
    Sort YOLO detections by x-coordinate (left to right).
    Args:
        detections (list): List of tuples (center_x, cls, conf, bbox)
    Returns:
        sorted list of detections
    """
    return sorted(detections, key=lambda det: det[0])


def solve_captcha(image_path: str, conf_threshold: float = 0.25):
    """
    Run YOLO inference on a single CAPTCHA image.
    
    Args:
        image_path (str): Path to CAPTCHA image.
        conf_threshold (float): Minimum confidence threshold for detections.
        
    Returns:
        digits (list): Detected digits in left-to-right order.
        output_path (Path): Path to saved image with bounding boxes.
    """
    try:
        model_path = WEIGHTS_DIR / "best.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found at {model_path}")

        logger.info(f"Loading YOLO model from: {model_path}")
        model = YOLO(str(model_path))

        # Run detection
        results = model.predict(source=image_path, conf=conf_threshold, save=False, verbose=False)
        img = cv2.imread(str(image_path))

        detected_digits = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                center_x = (x1 + x2) / 2
                detected_digits.append((center_x, cls, conf, (x1, y1, x2, y2)))

        if not detected_digits:
            logger.warning(f"No digits detected in {image_path}")

        # Sort digits left-to-right
        detected_digits = sort_detections_by_x(detected_digits)
        digits = [str(cls) for (_, cls, _, _) in detected_digits]

        # Draw bounding boxes
        for (_, cls, conf, (x1, y1, x2, y2)) in detected_digits:
            label = f"{cls} ({conf:.2f})"
            color = (0, 255, 0)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(img, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        output_path = PREDICTIONS_DIR / f"{Path(image_path).stem}_pred.jpg"
        cv2.imwrite(str(output_path), img)

        return digits, output_path

    except Exception as e:
        logger.error(f"Inference failed for {image_path}: {e}", exc_info=True)
        raise

