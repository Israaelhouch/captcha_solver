import argparse
from pathlib import Path
from src.utils.logger import setup_logger
from src.train import train_model
from src.evaluate import evaluate_model
from src.inference import solve_captcha
from src.config import WEIGHTS_DIR

logger = setup_logger()

def main():
    parser = argparse.ArgumentParser(
        description="CAPTCHA Solver CLI - Train, Evaluate, or Solve CAPTCHA images"
    )
    parser.add_argument(
        "action",
        choices=["train", "evaluate", "solve"],
        help="Action to perform: train / evaluate / solve"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to CAPTCHA image (required if action='solve')"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detection (default: 0.25)"
    )
    args = parser.parse_args()

    required_model = WEIGHTS_DIR / "best.pt"

    try:
        if args.action == "train":
            logger.info("Starting training process...")
            train_model()

        elif args.action in ["evaluate", "solve"]:
            if not required_model.exists():
                logger.error(f"No trained model found at {required_model}. Please run 'train' first!")
                return

            if args.action == "evaluate":
                logger.info("Starting evaluation process...")
                evaluate_model()
            else:  # solve
                if not args.image:
                    logger.error("Please provide an image path with --image when using 'solve'.")
                    return
                image_path = Path(args.image)
                logger.info(f"Solving CAPTCHA for image: {image_path}")
                digits, out_path = solve_captcha(str(image_path), conf_threshold=args.conf)
                logger.info(f"CAPTCHA solved: {digits}")
                logger.info(f"Output saved to: {out_path}")

        else:
            logger.error("Unknown action. Use train / evaluate / solve.")

    except Exception as e:
        logger.error(f"Run failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
