# -*- coding: utf-8 -*-
"""
train.py

Training script for the CWG-RGB benchmark experiments.

This script supports experiments with:
1. Different input resolutions: 640 and 960
2. Dataset settings: without background negative samples and with background negative samples
3. Model variants: YOLOv8n, YOLOv8s, YOLO11n and YOLO11s
4. Training-adapted setting: AdamW optimizer
5. P2-head model configurations

Example usage:

    python scripts/train.py --exp 640_no_bg_yolov8n
    python scripts/train.py --exp 960_bg_opt_yolov8s
    python scripts/train.py --exp 960_bg_opt_yolov8s_p2
    python scripts/train.py --exp all

Before running this script, make sure the following files exist:

    configs/data_no_bg.yaml
    configs/data_with_bg.yaml

For P2-head experiments, the model configuration names are resolved by
Ultralytics from its built-in model YAML registry. The corresponding P2 YAML
files are not stored in this repository.
"""

from pathlib import Path
import argparse
import sys

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError(
        "Ultralytics is not installed. Please install it first:\n"
        "pip install ultralytics"
    )


# ============================================================
# 1. Project paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"

DATA_NO_BG = CONFIG_DIR / "data_no_bg.yaml"
DATA_WITH_BG = CONFIG_DIR / "data_with_bg.yaml"

DEFAULT_PROJECT = ROOT / "runs" / "train"


# ============================================================
# 2. Basic training settings
# ============================================================

DEFAULT_EPOCHS = 300
DEFAULT_BATCH = 8
DEFAULT_WORKERS = 4

ADAMW_LR0 = 0.001
ADAMW_WEIGHT_DECAY = 0.0005


# ============================================================
# 3. Experiment definitions
# ============================================================
# data:
#   "no_bg" means configs/data_no_bg.yaml
#   "with_bg" means configs/data_with_bg.yaml
#
# optimizer:
#   "auto" means default Ultralytics optimizer setting
#   "AdamW" means training-adapted setting used in the paper
#
# model:
#   official pretrained weights, such as yolov8n.pt
#
# model_cfg:
#   model YAML name or path, mainly used for P2-head experiments.
#   P2-head experiments use Ultralytics built-in model YAML names.
#   If model_cfg is used, pretrained weights will be loaded from "pretrained".

EXPERIMENTS = {
    # --------------------------------------------------------
    # 640 / without background negative samples
    # --------------------------------------------------------
    "640_no_bg_yolov8n": {
        "data": "no_bg",
        "model": "yolov8n.pt",
        "imgsz": 640,
        "optimizer": "auto",
    },
    "640_no_bg_yolov8s": {
        "data": "no_bg",
        "model": "yolov8s.pt",
        "imgsz": 640,
        "optimizer": "auto",
    },
    "640_no_bg_yolo11n": {
        "data": "no_bg",
        "model": "yolo11n.pt",
        "imgsz": 640,
        "optimizer": "auto",
    },
    "640_no_bg_yolo11s": {
        "data": "no_bg",
        "model": "yolo11s.pt",
        "imgsz": 640,
        "optimizer": "auto",
    },

    # --------------------------------------------------------
    # 960 / without background negative samples
    # --------------------------------------------------------
    "960_no_bg_yolov8n": {
        "data": "no_bg",
        "model": "yolov8n.pt",
        "imgsz": 960,
        "optimizer": "auto",
    },
    "960_no_bg_yolov8s": {
        "data": "no_bg",
        "model": "yolov8s.pt",
        "imgsz": 960,
        "optimizer": "auto",
    },
    "960_no_bg_yolo11n": {
        "data": "no_bg",
        "model": "yolo11n.pt",
        "imgsz": 960,
        "optimizer": "auto",
    },
    "960_no_bg_yolo11s": {
        "data": "no_bg",
        "model": "yolo11s.pt",
        "imgsz": 960,
        "optimizer": "auto",
    },

    # --------------------------------------------------------
    # 640 / with background negative samples
    # --------------------------------------------------------
    "640_bg_yolov8n": {
        "data": "with_bg",
        "model": "yolov8n.pt",
        "imgsz": 640,
        "optimizer": "auto",
    },

    # --------------------------------------------------------
    # 960 / with background negative samples
    # --------------------------------------------------------
    "960_bg_yolov8n": {
        "data": "with_bg",
        "model": "yolov8n.pt",
        "imgsz": 960,
        "optimizer": "auto",
    },
    "960_bg_yolov8s": {
        "data": "with_bg",
        "model": "yolov8s.pt",
        "imgsz": 960,
        "optimizer": "auto",
    },
    "960_bg_yolo11n": {
        "data": "with_bg",
        "model": "yolo11n.pt",
        "imgsz": 960,
        "optimizer": "auto",
    },
    "960_bg_yolo11s": {
        "data": "with_bg",
        "model": "yolo11s.pt",
        "imgsz": 960,
        "optimizer": "auto",
    },

    # --------------------------------------------------------
    # 960 / with background negative samples / AdamW optimization
    # --------------------------------------------------------
    "960_bg_opt_yolov8n": {
        "data": "with_bg",
        "model": "yolov8n.pt",
        "imgsz": 960,
        "optimizer": "AdamW",
        "lr0": ADAMW_LR0,
        "weight_decay": ADAMW_WEIGHT_DECAY,
    },
    "960_bg_opt_yolov8s": {
        "data": "with_bg",
        "model": "yolov8s.pt",
        "imgsz": 960,
        "optimizer": "AdamW",
        "lr0": ADAMW_LR0,
        "weight_decay": ADAMW_WEIGHT_DECAY,
    },
    "960_bg_opt_yolo11n": {
        "data": "with_bg",
        "model": "yolo11n.pt",
        "imgsz": 960,
        "optimizer": "AdamW",
        "lr0": ADAMW_LR0,
        "weight_decay": ADAMW_WEIGHT_DECAY,
    },
    "960_bg_opt_yolo11s": {
        "data": "with_bg",
        "model": "yolo11s.pt",
        "imgsz": 960,
        "optimizer": "AdamW",
        "lr0": ADAMW_LR0,
        "weight_decay": ADAMW_WEIGHT_DECAY,
    },

    # --------------------------------------------------------
    # P2-head experiments
    # These experiments use Ultralytics built-in P2 model yaml names.
    # --------------------------------------------------------
    "960_bg_opt_yolov8n_p2": {
        "data": "with_bg",
        "model_cfg": "yolov8n-p2.yaml",
        "pretrained": "yolov8n.pt",
        "imgsz": 960,
        "optimizer": "AdamW",
        "lr0": ADAMW_LR0,
        "weight_decay": ADAMW_WEIGHT_DECAY,
    },
    "960_bg_opt_yolov8s_p2": {
        "data": "with_bg",
        "model_cfg": "yolov8s-p2.yaml",
        "pretrained": "yolov8s.pt",
        "imgsz": 960,
        "optimizer": "AdamW",
        "lr0": ADAMW_LR0,
        "weight_decay": ADAMW_WEIGHT_DECAY,
    },
    "960_bg_opt_yolo11n_p2": {
        "data": "with_bg",
        "model_cfg": "yolo11n-p2.yaml",
        "pretrained": "yolo11n.pt",
        "imgsz": 960,
        "optimizer": "AdamW",
        "lr0": ADAMW_LR0,
        "weight_decay": ADAMW_WEIGHT_DECAY,
    },
    "960_bg_opt_yolo11s_p2": {
        "data": "with_bg",
        "model_cfg": "yolo11s-p2.yaml",
        "pretrained": "yolo11s.pt",
        "imgsz": 960,
        "optimizer": "AdamW",
        "lr0": ADAMW_LR0,
        "weight_decay": ADAMW_WEIGHT_DECAY,
    },
}


# ============================================================
# 4. Helper functions
# ============================================================

def get_data_yaml(data_key: str) -> Path:
    """Return dataset yaml path according to data setting."""
    if data_key == "no_bg":
        return DATA_NO_BG
    if data_key == "with_bg":
        return DATA_WITH_BG
    raise ValueError(f"Unknown data setting: {data_key}")


def check_file_exists(file_path: Path, description: str) -> None:
    """Check whether a required file exists."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n{file_path}\n"
            f"Please check the repository structure or file name."
        )


def build_model(exp_cfg: dict) -> YOLO:
    """
    Build YOLO model.

    For official models:
        YOLO('yolov8s.pt')

    For P2-head models:
        YOLO('yolov8s-p2.yaml').load('yolov8s.pt')
    """
    if "model_cfg" in exp_cfg:
        model_cfg = str(exp_cfg["model_cfg"])

        model = YOLO(model_cfg)

        pretrained = exp_cfg.get("pretrained", None)
        if pretrained is not None:
            model.load(pretrained)

        return model

    model_weight = exp_cfg["model"]
    return YOLO(model_weight)


def print_experiment_list() -> None:
    """Print all available experiment names."""
    print("\nAvailable experiments:\n")
    for name in EXPERIMENTS.keys():
        print(f"  {name}")
    print("\nExample:")
    print("  python scripts/train.py --exp 960_bg_opt_yolov8s")
    print("  python scripts/train.py --exp all\n")


def run_one_experiment(exp_name: str, args: argparse.Namespace) -> None:
    """Run one training experiment."""
    if exp_name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment name: {exp_name}")

    exp_cfg = EXPERIMENTS[exp_name]

    data_yaml = get_data_yaml(exp_cfg["data"])
    check_file_exists(data_yaml, "Dataset configuration file")

    epochs = args.epochs if args.epochs is not None else DEFAULT_EPOCHS
    batch = args.batch if args.batch is not None else DEFAULT_BATCH
    workers = args.workers if args.workers is not None else DEFAULT_WORKERS

    project_dir = Path(args.project) if args.project is not None else DEFAULT_PROJECT

    train_args = {
        "data": str(data_yaml),
        "imgsz": exp_cfg["imgsz"],
        "epochs": epochs,
        "batch": batch,
        "workers": workers,
        "project": str(project_dir),
        "name": exp_name,
        "device": args.device,
        "seed": args.seed,
        "exist_ok": args.exist_ok,
    }

    # Use cache only when specified.
    if args.cache:
        train_args["cache"] = True

    # Optimizer setting
    optimizer = exp_cfg.get("optimizer", "auto")
    train_args["optimizer"] = optimizer

    if optimizer == "AdamW":
        train_args["lr0"] = exp_cfg.get("lr0", ADAMW_LR0)
        train_args["weight_decay"] = exp_cfg.get("weight_decay", ADAMW_WEIGHT_DECAY)

    print("\n" + "=" * 80)
    print(f"Start experiment: {exp_name}")
    print("=" * 80)
    print(f"Dataset yaml : {data_yaml}")
    print(f"Image size   : {exp_cfg['imgsz']}")
    print(f"Epochs       : {epochs}")
    print(f"Batch size   : {batch}")
    print(f"Workers      : {workers}")
    print(f"Optimizer    : {optimizer}")
    print(f"Device       : {args.device}")
    print(f"Project dir  : {project_dir}")

    if "model_cfg" in exp_cfg:
        print(f"Model cfg    : {exp_cfg['model_cfg']}")
        print(f"Pretrained   : {exp_cfg.get('pretrained')}")
    else:
        print(f"Model weight : {exp_cfg['model']}")

    if optimizer == "AdamW":
        print(f"lr0          : {train_args['lr0']}")
        print(f"weight_decay : {train_args['weight_decay']}")

    print("=" * 80 + "\n")

    if args.dry_run:
        print("Dry run mode: training is not executed.")
        return

    model = build_model(exp_cfg)
    model.train(**train_args)

    print("\n" + "=" * 80)
    print(f"Finished experiment: {exp_name}")
    print("=" * 80 + "\n")


# ============================================================
# 5. Main function
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO models for the CWG-RGB benchmark experiments."
    )

    parser.add_argument(
        "--exp",
        type=str,
        default="list",
        help=(
            "Experiment name. Use 'list' to show all experiments, "
            "or 'all' to run all experiments."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Training device. Use '0' for GPU 0 or 'cpu' for CPU."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"Number of training epochs. Default: {DEFAULT_EPOCHS}"
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help=f"Batch size. Default: {DEFAULT_BATCH}"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Number of dataloader workers. Default: {DEFAULT_WORKERS}"
    )

    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Output project directory. Default: runs/train"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed. Default: 0"
    )

    parser.add_argument(
        "--cache",
        action="store_true",
        help="Use dataset caching during training."
    )

    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow existing output directory."
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print experiment settings without running training."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.exp == "list":
        print_experiment_list()
        return

    if args.exp == "all":
        exp_names = list(EXPERIMENTS.keys())
    else:
        exp_names = [args.exp]

    for exp_name in exp_names:
        if exp_name not in EXPERIMENTS:
            print(f"\nUnknown experiment: {exp_name}")
            print_experiment_list()
            sys.exit(1)

        run_one_experiment(exp_name, args)


if __name__ == "__main__":
    main()
