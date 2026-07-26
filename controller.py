"""Beginner-friendly entry point for the fiberglass segmentation project."""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parent
VENV_PYTHON = ROOT / ".ven" / "bin" / "python"


def project_python() -> str:
    return str(VENV_PYTHON if VENV_PYTHON.exists() else sys.executable)


def run(*arguments: str) -> int:
    command = [project_python(), *arguments]
    print("\nRunning:", " ".join(command))
    return subprocess.run(command, cwd=ROOT).returncode


def setup() -> int:
    if not VENV_PYTHON.exists():
        result = subprocess.run([sys.executable, "-m", "venv", ".ven"], cwd=ROOT)
        if result.returncode:
            return result.returncode
    return run("-m", "pip", "install", "-r", "requirements.txt")


def validate(with_overlays: bool = False) -> int:
    arguments = ["-m", "training.validate_data"]
    if with_overlays:
        arguments.append("--with-overlays")
    return run(*arguments)


def train(with_overlays: bool = False) -> int:
    arguments = ["-m", "training.train"]
    if with_overlays:
        arguments.append("--with-overlays")
    return run(*arguments)


def predict(input_path: str | None) -> int:
    return run(
        "-m",
        "predicting.predict_configurable",
        input_path or "predicting/images",
    )


def statistics() -> int:
    return run(
        "-m",
        "predicting.get_statistics",
        "predicting/outputs",
        "--output",
        "predicting/outputs/statistics/statistics.csv",
        "--report-dir",
        "predicting/outputs/statistics",
    )


def analysis() -> int:
    return run(
        "-m",
        "predicting.analyze_statistics",
        "predicting/outputs/statistics/statistics.csv",
    )


def all_steps(with_overlays: bool = False) -> int:
    if validate(with_overlays):
        return 1
    if train(with_overlays):
        return 1
    if predict(None):
        return 1
    if statistics():
        return 1
    if analysis():
        return 1
    return 0


def tests() -> int:
    return run("-m", "unittest", "discover", "-s", "tests")


GUIDE = """
Fiberglass segmentation controller

Start here:
  python controller.py setup       Install the project once.
  python controller.py all         Run validation, training, prediction, and reports.

Commands:
  setup                            Prepare the Python environment.
  validate [--with-overlays]       Check painted training masks.
  train [--with-overlays]          Train a fresh model.
  predict [IMAGE_OR_FOLDER]        Segment images; default: predicting/images.
  statistics                       Make CSV and image reports from predictions.
  analysis                         Make grouped charts and a Markdown report.
  all [--with-overlays]            Run the complete workflow in order.
  test                             Run the code checks.

Folders:
  training/images                  Training pictures.
  training/masks                   Matching painted masks.
  predicting/images                Pictures to segment.
  predicting/outputs               Prediction masks, overlays, and reports.
  checkpoints/final.pt             The trained model used for prediction.

Advanced direct commands:
  python -m predicting.predict_configurable predicting/images
  python -m predicting.predict IMAGE --checkpoint CHECKPOINT --output-dir OUTPUT
  python -m training.train --epochs 60 --resume checkpoints/latest.pt

Use predict_configurable for the easy fixed defaults. Use predict when you
want to choose the checkpoint, output folder, tile size, or overlap yourself.
Use `python -m MODULE --help` to see every advanced option.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("command", nargs="?")
    parser.add_argument("input_path", nargs="?")
    parser.add_argument("--with-overlays", action="store_true")
    parser.add_argument("--help", "-h", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.command or args.help:
        print(GUIDE)
        return 0

    commands = {
        "setup": setup,
        "statistics": statistics,
        "analysis": analysis,
        "test": tests,
    }
    if args.command == "validate":
        return validate(args.with_overlays)
    if args.command == "train":
        return train(args.with_overlays)
    if args.command == "predict":
        return predict(args.input_path)
    if args.command == "all":
        return all_steps(args.with_overlays)
    if args.command in commands:
        return commands[args.command]()

    print(f"Unknown command: {args.command}\n\n{GUIDE}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
