"""Compatibility entry point for the training_dataset_v1_gzy experiment."""

from pathlib import Path
import runpy


runpy.run_path(
    str(Path(__file__).resolve().parent / "experiments/training_dataset_v1_gzy/prepare.py"),
    run_name="__main__",
)
