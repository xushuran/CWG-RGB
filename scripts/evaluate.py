# -*- coding: utf-8 -*-
"""
evaluate.py

Evaluation script for the CWG-RGB benchmark experiments.

This script supports:
1. Evaluation of trained YOLO models
2. Different input sizes: 640 and 960
3. Dataset settings: without background negative samples and with background negative samples
4. Model variants: YOLOv8n, YOLOv8s, YOLO11n and YOLO11s
5. Training-adapted models using AdamW
6. P2-head models
7. NMS parameter calibration

Example usage:

    python scripts/evaluate.py --exp 960_bg_opt_yolov8s

    python scripts/evaluate.py --exp 960_bg_opt_yolov8s --preset nms_calibrated

    python scripts/evaluate.py --weights runs/train/960_bg_opt_yolov8s/weights/best.pt --data configs/data_with_bg.yaml --imgsz 960

    python scripts/evaluate.py --exp 960_bg_opt_yolov8s --nms-grid

    python scripts/evaluate.py --eval-all

The main calibrated NMS setting used in the benchmark is:

    confidence threshold = 0.35
    IoU threshold = 0.45
    max detections = 300
"""

from pathlib import Path
import argparse
import csv
import sys
from datetime import datetime

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

DEFAULT_TRAIN_RUNS = ROOT / "runs" / "train"
DEFAULT_VAL_RUNS = ROOT / "runs" / "val"
DEFAULT_RESULTS_DIR = ROOT / "results"


# ============================================================
# 2. Evaluation presets
# ============================================================

EVAL_PRESETS = {
    # Standard validation setting.
    # This follows the general YOLO validation setting.
    "standard_640": {
        "imgsz": 640,
        "conf": 0.001,
        "iou": 0.70,
        "max_det": 300,
    },
    "standard_960": {
        "imgsz": 960,
        "conf": 0.001,
        "iou": 0.70,
        "max_det": 300,
    },

    # Main NMS-calibrated setting used for the final benchmark.
    "nms_calibrated": {
        "imgsz": 960,
        "conf": 0.35,
        "iou": 0.45,
        "max_det": 300,
    },
}


# ============================================================
# 3. Experiment definitions
# ============================================================
# These names are consistent with scripts/train.py.
#
# data:
#   "no_bg" means configs/data_no_bg.yaml
#   "with_bg" means configs/data_with_bg.yaml
#
# weights:
#   By default, the script searches:
#   runs/train/<experiment_name>/weights/best.pt

EXPERIMENTS = {
    # --------------------------------------------------------
    # 640 / without background negative samples
    # --------------------------------------------------------
    "640_no_bg_yolov8n": {
        "data": "no_bg",
        "imgsz": 640,
    },
    "640_no_bg_yolov8s": {
        "data": "no_bg",
        "imgsz": 640,
    },
    "640_no_bg_yolo11n": {
        "data": "no_bg",
        "imgsz": 640,
    },
    "640_no_bg_yolo11s": {
        "data": "no_bg",
        "imgsz": 640,
    },

    # --------------------------------------------------------
    # 960 / without background negative samples
    # --------------------------------------------------------
    "960_no_bg_yolov8n": {
        "data": "no_bg",
        "imgsz": 960,
    },
    "960_no_bg_yolov8s": {
        "data": "no_bg",
        "imgsz": 960,
    },
    "960_no_bg_yolo11n": {
        "data": "no_bg",
        "imgsz": 960,
    },
    "960_no_bg_yolo11s": {
        "data": "no_bg",
        "imgsz": 960,
    },

    # --------------------------------------------------------
    # 640 / with background negative samples
    # --------------------------------------------------------
    "640_bg_yolov8n": {
        "data": "with_bg",
        "imgsz": 640,
    },

    # --------------------------------------------------------
    # 960 / with background negative samples
    # --------------------------------------------------------
    "960_bg_yolov8n": {
        "data": "with_bg",
        "imgsz": 960,
    },
    "960_bg_yolov8s": {
        "data": "with_bg",
        "imgsz": 960,
    },
    "960_bg_yolo11n": {
        "data": "with_bg",
        "imgsz": 960,
    },
    "960_bg_yolo11s": {
        "data": "with_bg",
        "imgsz": 960,
    },

    # --------------------------------------------------------
    # 960 / with background negative samples / AdamW optimization
    # --------------------------------------------------------
    "960_bg_opt_yolov8n": {
        "data": "with_bg",
        "imgsz": 960,
    },
    "960_bg_opt_yolov8s": {
        "data": "with_bg",
        "imgsz": 960,
    },
    "960_bg_opt_yolo11n": {
        "data": "with_bg",
        "imgsz": 960,
    },
    "960_bg_opt_yolo11s": {
        "data": "with_bg",
        "imgsz": 960,
    },

    # --------------------------------------------------------
    # P2-head experiments
    # --------------------------------------------------------
    "960_bg_opt_yolov8n_p2": {
        "data": "with_bg",
        "imgsz": 960,
    },
    "960_bg_opt_yolov8s_p2": {
        "data": "with_bg",
        "imgsz": 960,
    },
    "960_bg_opt_yolo11n_p2": {
        "data": "with_bg",
        "imgsz": 960,
    },
    "960_bg_opt_yolo11s_p2": {
        "data": "with_bg",
        "imgsz": 960,
    },
}


# ============================================================
# 4. NMS grid for parameter calibration
# ============================================================

NMS_GRID = {
    "conf": [0.25, 0.30, 0.35, 0.40],
    "iou": [0.40, 0.45, 0.50, 0.55],
    "max_det": [300],
}


# ============================================================
# 5. Helper functions
# ============================================================

def get_data_yaml(data_key: str) -> Path:
    """Return dataset yaml path according to data setting."""
    if data_key == "no_bg":
        return DATA_NO_BG
    if data_key == "with_bg":
        return DATA_WITH_BG
    raise ValueError(f"Unknown data setting: {data_key}")


def resolve_path(path_str: str | None, default_path: Path | None = None) -> Path:
    """Resolve path. Relative paths are treated as relative to project root."""
    if path_str is None:
        if default_path is None:
            raise ValueError("Path is not provided.")
        return default_path

    path = Path(path_str)

    if path.is_absolute():
        return path

    return ROOT / path


def check_file_exists(path: Path, description: str) -> None:
    """Check whether a required file exists."""
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n{path}\n"
            f"Please check the path."
        )


def get_default_weights(exp_name: str) -> Path:
    """Return default best.pt path for a given experiment."""
    return DEFAULT_TRAIN_RUNS / exp_name / "weights" / "best.pt"


def print_experiment_list() -> None:
    """Print available experiments."""
    print("\nAvailable experiments:\n")
    for name in EXPERIMENTS.keys():
        print(f"  {name}")

    print("\nExamples:")
    print("  python scripts/evaluate.py --exp 960_bg_opt_yolov8s")
    print("  python scripts/evaluate.py --exp 960_bg_opt_yolov8s --preset nms_calibrated")
    print("  python scripts/evaluate.py --exp 960_bg_opt_yolov8s --nms-grid")
    print("  python scripts/evaluate.py --eval-all\n")


def print_preset_list() -> None:
    """Print available evaluation presets."""
    print("\nAvailable evaluation presets:\n")
    for name, cfg in EVAL_PRESETS.items():
        print(
            f"  {name}: "
            f"imgsz={cfg['imgsz']}, "
            f"conf={cfg['conf']}, "
            f"iou={cfg['iou']}, "
            f"max_det={cfg['max_det']}"
        )
    print()


def safe_getattr(obj, attr: str, default=None):
    """Safely get nested attribute."""
    try:
        return getattr(obj, attr)
    except Exception:
        return default


def extract_metrics(metrics) -> dict:
    """
    Extract detection metrics from Ultralytics validation results.

    Common attributes:
        metrics.box.mp     Precision
        metrics.box.mr     Recall
        metrics.box.map50  mAP@0.5
        metrics.box.map    mAP@0.5:0.95
        metrics.box.maps   class-wise mAP@0.5:0.95
    """
    box = safe_getattr(metrics, "box", None)

    if box is None:
        return {
            "precision": None,
            "recall": None,
            "map50": None,
            "map50_95": None,
        }

    result = {
        "precision": safe_getattr(box, "mp", None),
        "recall": safe_getattr(box, "mr", None),
        "map50": safe_getattr(box, "map50", None),
        "map50_95": safe_getattr(box, "map", None),
    }

    return result


def format_metric(value):
    """Format metric value for CSV output."""
    if value is None:
        return ""

    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value)


def append_csv(csv_path: Path, row: dict, fieldnames: list[str]) -> None:
    """Append one row to a CSV file."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def write_classwise_csv(csv_path: Path, exp_name: str, metrics, class_names: list[str]) -> None:
    """
    Save class-wise mAP@0.5:0.95 if available.

    Note:
        Ultralytics usually provides class-wise mAP@0.5:0.95 through metrics.box.maps.
        Class-wise AP@0.5 may not be directly available in all versions.
    """
    box = safe_getattr(metrics, "box", None)

    if box is None:
        return

    maps = safe_getattr(box, "maps", None)

    if maps is None:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["experiment", "class_id", "class_name", "map50_95"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for class_id, value in enumerate(maps):
            if class_id < len(class_names):
                class_name = class_names[class_id]
            else:
                class_name = f"class_{class_id}"

            writer.writerow(
                {
                    "experiment": exp_name,
                    "class_id": class_id,
                    "class_name": class_name,
                    "map50_95": format_metric(value),
                }
            )


# ============================================================
# 6. Evaluation functions
# ============================================================

def evaluate_one(
    weights: Path,
    data_yaml: Path,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    device: str,
    project: Path,
    name: str,
    save_json: bool = False,
    plots: bool = True,
    exist_ok: bool = True,
):
    """Run one YOLO validation."""
    check_file_exists(weights, "Model weights")
    check_file_exists(data_yaml, "Dataset configuration file")

    model = YOLO(str(weights))

    metrics = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=device,
        project=str(project),
        name=name,
        save_json=save_json,
        plots=plots,
        exist_ok=exist_ok,
    )

    return metrics


def build_eval_setting_from_exp(exp_name: str, preset_name: str) -> dict:
    """Build evaluation setting using experiment name and preset."""
    if exp_name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment name: {exp_name}")

    if preset_name not in EVAL_PRESETS:
        raise ValueError(f"Unknown evaluation preset: {preset_name}")

    exp_cfg = EXPERIMENTS[exp_name]
    preset_cfg = EVAL_PRESETS[preset_name]

    data_yaml = get_data_yaml(exp_cfg["data"])

    # For standard presets, use image size from experiment.
    # For nms_calibrated, use the preset setting unless overwritten by command line.
    if preset_name.startswith("standard"):
        imgsz = exp_cfg["imgsz"]
    else:
        imgsz = preset_cfg["imgsz"]

    return {
        "data_yaml": data_yaml,
        "weights": get_default_weights(exp_name),
        "imgsz": imgsz,
        "conf": preset_cfg["conf"],
        "iou": preset_cfg["iou"],
        "max_det": preset_cfg["max_det"],
    }


def run_single_evaluation(args: argparse.Namespace) -> None:
    """Evaluate one model and save metrics to CSV."""

    if args.exp is not None:
        if args.exp not in EXPERIMENTS:
            print(f"\nUnknown experiment: {args.exp}")
            print_experiment_list()
            sys.exit(1)

        setting = build_eval_setting_from_exp(args.exp, args.preset)
        exp_name = args.exp

    else:
        if args.weights is None or args.data is None:
            raise ValueError(
                "Please provide either --exp or both --weights and --data.\n"
                "Example:\n"
                "python scripts/evaluate.py --exp 960_bg_opt_yolov8s\n"
                "or:\n"
                "python scripts/evaluate.py --weights path/to/best.pt --data configs/data_with_bg.yaml"
            )

        exp_name = args.name
        setting = {
            "weights": resolve_path(args.weights),
            "data_yaml": resolve_path(args.data),
            "imgsz": args.imgsz if args.imgsz is not None else 960,
            "conf": args.conf if args.conf is not None else 0.001,
            "iou": args.iou if args.iou is not None else 0.70,
            "max_det": args.max_det if args.max_det is not None else 300,
        }

    # Command-line parameters override preset settings.
    weights = resolve_path(args.weights) if args.weights is not None else setting["weights"]
    data_yaml = resolve_path(args.data) if args.data is not None else setting["data_yaml"]
    imgsz = args.imgsz if args.imgsz is not None else setting["imgsz"]
    conf = args.conf if args.conf is not None else setting["conf"]
    iou = args.iou if args.iou is not None else setting["iou"]
    max_det = args.max_det if args.max_det is not None else setting["max_det"]

    project = resolve_path(args.project, DEFAULT_VAL_RUNS)

    run_name = args.name
    if run_name == "auto":
        run_name = f"{exp_name}_{args.preset}_conf{conf}_iou{iou}_maxdet{max_det}"

    print("\n" + "=" * 80)
    print("Start evaluation")
    print("=" * 80)
    print(f"Experiment    : {exp_name}")
    print(f"Weights       : {weights}")
    print(f"Data yaml     : {data_yaml}")
    print(f"Preset        : {args.preset}")
    print(f"Image size    : {imgsz}")
    print(f"Confidence    : {conf}")
    print(f"IoU threshold : {iou}")
    print(f"Max detections: {max_det}")
    print(f"Device        : {args.device}")
    print(f"Project dir   : {project}")
    print(f"Run name      : {run_name}")
    print("=" * 80 + "\n")

    if args.dry_run:
        print("Dry run mode: evaluation is not executed.")
        return

    metrics = evaluate_one(
        weights=weights,
        data_yaml=data_yaml,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=args.device,
        project=project,
        name=run_name,
        save_json=args.save_json,
        plots=not args.no_plots,
        exist_ok=args.exist_ok,
    )

    metric_values = extract_metrics(metrics)

    results_dir = resolve_path(args.results_dir, DEFAULT_RESULTS_DIR)
    summary_csv = results_dir / "evaluation_summary.csv"

    row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": exp_name,
        "weights": str(weights),
        "data_yaml": str(data_yaml),
        "preset": args.preset,
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "max_det": max_det,
        "precision": format_metric(metric_values["precision"]),
        "recall": format_metric(metric_values["recall"]),
        "map50": format_metric(metric_values["map50"]),
        "map50_95": format_metric(metric_values["map50_95"]),
    }

    fieldnames = [
        "time",
        "experiment",
        "weights",
        "data_yaml",
        "preset",
        "imgsz",
        "conf",
        "iou",
        "max_det",
        "precision",
        "recall",
        "map50",
        "map50_95",
    ]

    append_csv(summary_csv, row, fieldnames)

    class_names = ["Bags-Films", "Wood", "Foam", "Bottles-Cans", "Dead Fish"]
    classwise_csv = results_dir / f"{exp_name}_classwise_map50_95.csv"
    write_classwise_csv(classwise_csv, exp_name, metrics, class_names)

    print("\n" + "=" * 80)
    print("Evaluation finished.")
    print(f"Precision     : {row['precision']}")
    print(f"Recall        : {row['recall']}")
    print(f"mAP@0.5       : {row['map50']}")
    print(f"mAP@0.5:0.95  : {row['map50_95']}")
    print(f"Summary saved : {summary_csv}")
    print("=" * 80 + "\n")


def run_eval_all(args: argparse.Namespace) -> None:
    """Evaluate all experiments defined in EXPERIMENTS."""
    for exp_name in EXPERIMENTS.keys():
        weights = get_default_weights(exp_name)

        if not weights.exists():
            print(f"[Skip] Weights not found for {exp_name}: {weights}")
            continue

        args.exp = exp_name
        run_single_evaluation(args)


def run_nms_grid(args: argparse.Namespace) -> None:
    """
    Run NMS grid search for a trained model.

    Usually used for:
        960_bg_opt_yolov8s
    """
    if args.exp is None and args.weights is None:
        raise ValueError(
            "NMS grid requires --exp or --weights.\n"
            "Example:\n"
            "python scripts/evaluate.py --exp 960_bg_opt_yolov8s --nms-grid"
        )

    if args.exp is not None:
        if args.exp not in EXPERIMENTS:
            print(f"\nUnknown experiment: {args.exp}")
            print_experiment_list()
            sys.exit(1)

        exp_name = args.exp
        exp_cfg = EXPERIMENTS[exp_name]
        weights = get_default_weights(exp_name)
        data_yaml = get_data_yaml(exp_cfg["data"])
        imgsz = args.imgsz if args.imgsz is not None else exp_cfg["imgsz"]

    else:
        exp_name = args.name
        weights = resolve_path(args.weights)
        data_yaml = resolve_path(args.data)
        imgsz = args.imgsz if args.imgsz is not None else 960

    project = resolve_path(args.project, DEFAULT_VAL_RUNS)
    results_dir = resolve_path(args.results_dir, DEFAULT_RESULTS_DIR)

    nms_csv = results_dir / "nms_calibration_results.csv"

    fieldnames = [
        "time",
        "experiment",
        "weights",
        "data_yaml",
        "imgsz",
        "conf",
        "iou",
        "max_det",
        "precision",
        "recall",
        "map50",
        "map50_95",
    ]

    print("\n" + "=" * 80)
    print("Start NMS grid evaluation")
    print("=" * 80)
    print(f"Experiment : {exp_name}")
    print(f"Weights    : {weights}")
    print(f"Data yaml  : {data_yaml}")
    print(f"Image size : {imgsz}")
    print(f"Output CSV : {nms_csv}")
    print("=" * 80 + "\n")

    check_file_exists(weights, "Model weights")
    check_file_exists(data_yaml, "Dataset configuration file")

    if args.dry_run:
        print("Dry run mode: NMS grid evaluation is not executed.")
        return

    for conf in NMS_GRID["conf"]:
        for iou in NMS_GRID["iou"]:
            for max_det in NMS_GRID["max_det"]:
                run_name = f"{exp_name}_nms_conf{conf}_iou{iou}_maxdet{max_det}"

                print("\n" + "-" * 80)
                print(f"NMS setting: conf={conf}, iou={iou}, max_det={max_det}")
                print("-" * 80 + "\n")

                metrics = evaluate_one(
                    weights=weights,
                    data_yaml=data_yaml,
                    imgsz=imgsz,
                    conf=conf,
                    iou=iou,
                    max_det=max_det,
                    device=args.device,
                    project=project,
                    name=run_name,
                    save_json=args.save_json,
                    plots=not args.no_plots,
                    exist_ok=args.exist_ok,
                )

                metric_values = extract_metrics(metrics)

                row = {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "experiment": exp_name,
                    "weights": str(weights),
                    "data_yaml": str(data_yaml),
                    "imgsz": imgsz,
                    "conf": conf,
                    "iou": iou,
                    "max_det": max_det,
                    "precision": format_metric(metric_values["precision"]),
                    "recall": format_metric(metric_values["recall"]),
                    "map50": format_metric(metric_values["map50"]),
                    "map50_95": format_metric(metric_values["map50_95"]),
                }

                append_csv(nms_csv, row, fieldnames)

    print("\n" + "=" * 80)
    print("NMS grid evaluation finished.")
    print(f"Results saved: {nms_csv}")
    print("=" * 80 + "\n")


# ============================================================
# 7. Argument parser
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO models for the CWG-RGB benchmark."
    )

    parser.add_argument(
        "--exp",
        type=str,
        default=None,
        help=(
            "Experiment name. Use --list to show all experiments. "
            "If provided, weights will be loaded from runs/train/<exp>/weights/best.pt."
        ),
    )

    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to model weights, for example runs/train/960_bg_opt_yolov8s/weights/best.pt."
    )

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to dataset yaml. If --exp is used, this can be omitted."
    )

    parser.add_argument(
        "--preset",
        type=str,
        default="standard_960",
        help=(
            "Evaluation preset. Options: standard_640, standard_960, nms_calibrated. "
            "Use --list-presets to show details."
        ),
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Input image size. If not set, use experiment or preset setting."
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Confidence threshold. If not set, use preset setting."
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=None,
        help="IoU threshold for NMS. If not set, use preset setting."
    )

    parser.add_argument(
        "--max-det",
        dest="max_det",
        type=int,
        default=None,
        help="Maximum number of detections per image. If not set, use preset setting."
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Evaluation device. Use '0' for GPU 0 or 'cpu' for CPU."
    )

    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Output validation run directory. Default: runs/val"
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory for CSV result files. Default: results"
    )

    parser.add_argument(
        "--name",
        type=str,
        default="auto",
        help="Validation run name. Default: auto"
    )

    parser.add_argument(
        "--eval-all",
        action="store_true",
        help="Evaluate all experiments with existing weights."
    )

    parser.add_argument(
        "--nms-grid",
        action="store_true",
        help="Run NMS parameter calibration grid."
    )

    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save validation results in COCO JSON format if supported."
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable saving validation plots."
    )

    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow existing validation output directory."
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiments."
    )

    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available evaluation presets."
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print evaluation settings without running validation."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        print_experiment_list()
        return

    if args.list_presets:
        print_preset_list()
        return

    if args.preset not in EVAL_PRESETS:
        print(f"\nUnknown preset: {args.preset}")
        print_preset_list()
        sys.exit(1)

    if args.eval_all:
        run_eval_all(args)
        return

    if args.nms_grid:
        run_nms_grid(args)
        return

    run_single_evaluation(args)


if __name__ == "__main__":
    main()