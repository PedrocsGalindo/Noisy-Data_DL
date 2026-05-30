"""Simple Kaggle launcher for running one configured method."""

import argparse
import subprocess
from pathlib import Path


CONFIG_BY_METHOD = {
    "ce": "configs/kaggle_small_ce.yaml",
    "sce": "configs/kaggle_small_sce.yaml",
    "qmix_like": "configs/kaggle_small_qmix_like.yaml",
}


def main():
    parser = argparse.ArgumentParser(description="Launch a Kaggle retina experiment.")
    parser.add_argument(
        "--method",
        choices=sorted(CONFIG_BY_METHOD),
        default="ce",
        help="Method config to run.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    command = ["python", "-m", "src.train", "--config", CONFIG_BY_METHOD[args.method]]
    subprocess.run(command, cwd=project_root, check=True)


if __name__ == "__main__":
    main()
