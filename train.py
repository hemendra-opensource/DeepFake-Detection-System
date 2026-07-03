"""
train.py
=========
CLI entrypoint for training DeepFake detection models.

Usage::

    # Train all models with progressive strategy
    py -3.11 train.py --model xceptionnet --phases all

    # Train only Phase 1
    py -3.11 train.py --model efficientnet_b0 --phases phase_1

    # Run preprocessing first
    py -3.11 train.py --preprocess --datasets celeb_df

    # Evaluate all saved models
    py -3.11 train.py --evaluate
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.logger import get_logger, setup_logging

setup_logging(log_dir="logs")
logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="DeepFake Detection — Training & Evaluation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode flags
    mode_group = parser.add_argument_group("Mode")
    mode_group.add_argument(
        "--preprocess",
        action="store_true",
        help="Run the dataset preprocessing pipeline.",
    )
    mode_group.add_argument(
        "--train",
        action="store_true",
        default=True,
        help="Run model training (default).",
    )
    mode_group.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate all trained models on the test split.",
    )

    # Training options
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--model",
        type=str,
        default="xceptionnet",
        choices=["xceptionnet", "efficientnet_b0", "resnet50", "all"],
        help="Model to train (default: xceptionnet).",
    )
    train_group.add_argument(
        "--phases",
        type=str,
        default="all",
        help='Training phases to run: "all", "phase_1", "phase_2", "phase_3" '
             'or comma-separated (e.g. "phase_1,phase_2").',
    )
    train_group.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config YAML (default: configs/config.yaml).",
    )
    train_group.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip training phases where checkpoints already exist.",
    )

    # Preprocessing options
    pre_group = parser.add_argument_group("Preprocessing")
    pre_group.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated dataset keys to preprocess (e.g. 'celeb_df,ff_plus_plus').",
    )
    pre_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate without writing processed files.",
    )

    return parser.parse_args()


def run_preprocessing(args: argparse.Namespace) -> None:
    """Execute the preprocessing pipeline."""
    from preprocessing.pipeline import PreprocessingPipeline

    datasets = (
        [d.strip() for d in args.datasets.split(",")]
        if args.datasets
        else None
    )

    logger.info("Starting preprocessing pipeline…")
    pipeline = PreprocessingPipeline(
        config_path=args.config,
        dry_run=args.dry_run,
    )
    metadata = pipeline.run(datasets=datasets)

    if not metadata.empty:
        logger.info("Preprocessing complete. %d total samples.", len(metadata))
    else:
        logger.warning("No samples were processed. Check dataset paths in config.")


def run_training(args: argparse.Namespace) -> None:
    """Execute progressive training for the selected model(s)."""
    from training.progressive_trainer import ProgressiveTrainer

    models_to_train = (
        ["xceptionnet", "efficientnet_b0", "resnet50"]
        if args.model == "all"
        else [args.model]
    )

    phases = None
    if args.phases != "all":
        phases = [p.strip() for p in args.phases.split(",")]

    for model_name in models_to_train:
        logger.info("=" * 60)
        logger.info("Training model: %s", model_name)
        trainer = ProgressiveTrainer(model_name=model_name, config_path=args.config)
        final_path = trainer.run(
            phases=phases,
            skip_if_exists=args.skip_existing,
        )
        logger.info("Training complete. Final model: %s", final_path)


def run_evaluation(args: argparse.Namespace) -> None:
    """Evaluate all trained models on the test split."""
    from evaluation.evaluator import Evaluator

    logger.info("Starting model evaluation…")
    evaluator = Evaluator(config_path=args.config)
    results = evaluator.evaluate_all()

    if results:
        from evaluation.metrics import compare_models
        comparison = compare_models(results)
        print("\n" + "=" * 60)
        print("MODEL COMPARISON RESULTS")
        print("=" * 60)
        print(comparison.to_string(index=False))
        print("=" * 60)
    else:
        logger.warning("No models were evaluated. Train models first.")


def main() -> None:
    """Main CLI entrypoint."""
    args = parse_args()

    logger.info("DeepFake Detection CLI started.")
    logger.info("Config: %s", args.config)

    try:
        if args.preprocess:
            run_preprocessing(args)

        if args.evaluate:
            run_evaluation(args)
        elif args.train:
            run_training(args)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
