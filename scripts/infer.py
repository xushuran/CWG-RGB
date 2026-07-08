# -*- coding: utf-8 -*-
"""
infer.py

Inference script for the CWG-RGB benchmark experiments.

This script is used to generate qualitative detection results for:
1. YOLOv8n / YOLOv8s / YOLO11n / YOLO11s
2. 640 and 960 input resolutions
3. Models trained with or without background negative samples
4. Training-adapted models using AdamW
5. P2-head models
6. Calibrated NMS inference setting

Example usage:

    python scripts/infer.py --weights runs/train/960_bg_opt_yolov8s/weights/best.pt

    python scripts/infer.py \
        --weights runs/train/960_bg_opt_yolov8s/weights/best.pt \
        --source images/val \
        --imgsz 960 \
        --conf 0.35 \
        --iou 0.45 \
        --max-det 300

    python scripts/infer.py \
        --weights runs/train/960_bg_opt_yolov8s_p2/weights/best.pt \
        --source images/val \
        --imgsz 960

The main calibrated NMS setting used in the benchmark is:

    confidence threshold = 0.35
    IoU threshold = 0.45
    max detections = 300
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

DEFAULT_SOURCE = ROOT / "images" / "val"
DEFAULT_PROJECT = ROOT / "runs" / "predict"


# ============================================================
# 2. Inference presets
# ============================================================
# These presets correspond to the main experimental settings.
# The actual model is determined by the --weights argument.

INFER_PRESETS = {
    "640_default": {
        "imgsz": 640,
        "conf": 0.25,
        "iou": 0.70,
        "max_det": 300,
    },
    "960_default": {
        "imgsz": 960,
        "conf": 0.25,
        "iou": 0.70,
        "max_det": 300,
    },
    "960_nms_calibrated": {
        "imgsz": 960,
        "conf": 0.35,
        "iou": 0.45,
        "max_det": 300,
    },
}


# ============================================================
# 3. Helper functions
# ============================================================

def resolve_path(path_str: str | None, default_path: Path | None = None) -> Path:
    """
    Resolve input path.

    If the path is relative, it will be treated as relative to the project root.
    """
    if path_str is None:
        if default_path is None:
            raise ValueError("Path is not provided.")
        return default_path

    path = Path(path_str)

    if path.is_absolute():
        return path

    return ROOT / path


def check_path_exists(path: Path, description: str) -> None:
    """Check whether a path exists."""
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n{path}\n"
            f"Please check the file path."
        )


def print_preset_list() -> None:
    """Print available inference presets."""
    print("\nAvailable inference presets:\n")
    for name, cfg in INFER_PRESETS.items():
        print(
            f"  {name}: "
            f"imgsz={cfg['imgsz']}, "
            f"conf={cfg['conf']}, "
            f"iou={cfg['iou']}, "
            f"max_det={cfg['max_det']}"
        )

    print("\nExample:")
    print(
        "  python scripts/infer.py "
        "--weights runs/train/960_bg_opt_yolov8s/weights/best.pt "
        "--preset 960_nms_calibrated"
    )
    print()


def parse_classes(classes_str: str | None):
    """
    Parse class IDs from command line.

    Example:
        --classes 0,2,3

    Returns:
        None or list[int]
    """
    if classes_str is None or classes_str.strip() == "":
        return None

    try:
        return [int(x.strip()) for x in classes_str.split(",")]
    except ValueError:
        raise ValueError(
            "Invalid --classes format. Use comma-separated class IDs, for example: 0,2,3"
        )


# ============================================================
# 4. Main inference function
# ============================================================

def run_inference(args: argparse.Namespace) -> None:
    """Run YOLO inference."""

    if args.preset == "list":
        print_preset_list()
        return

    if args.preset not in INFER_PRESETS:
        print(f"\nUnknown preset: {args.preset}")
        print_preset_list()
        sys.exit(1)

    preset_cfg = INFER_PRESETS[args.preset]

    weights_path = resolve_path(args.weights)
    source_path = resolve_path(args.source, DEFAULT_SOURCE)
    project_dir = resolve_path(args.project, DEFAULT_PROJECT)

    check_path_exists(weights_path, "Model weights")
    check_path_exists(source_path, "Inference source")

    imgsz = args.imgsz if args.imgsz is not None else preset_cfg["imgsz"]
    conf = args.conf if args.conf is not None else preset_cfg["conf"]
    iou = args.iou if args.iou is not None else preset_cfg["iou"]
    max_det = args.max_det if args.max_det is not None else preset_cfg["max_det"]

    classes = parse_classes(args.classes)

    predict_args = {
        "source": str(source_path),
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "max_det": max_det,
        "device": args.device,
        "project": str(project_dir),
        "name": args.name,
        "exist_ok": args.exist_ok,
        "save": True,
        "save_txt": args.save_txt,
        "save_conf": args.save_conf,
        "save_crop": args.save_crop,
        "line_width": args.line_width,
        "show_labels": not args.hide_labels,
        "show_conf": not args.hide_conf,
        "agnostic_nms": args.agnostic_nms,
    }

    if classes is not None:
        predict_args["classes"] = classes

    print("\n" + "=" * 80)
    print("Start inference")
    print("=" * 80)
    print(f"Weights       : {weights_path}")
    print(f"Source        : {source_path}")
    print(f"Preset        : {args.preset}")
    print(f"Image size    : {imgsz}")
    print(f"Confidence    : {conf}")
    print(f"IoU threshold : {iou}")
    print(f"Max detections: {max_det}")
    print(f"Device        : {args.device}")
    print(f"Project dir   : {project_dir}")
    print(f"Run name      : {args.name}")
    print(f"Save TXT      : {args.save_txt}")
    print(f"Save conf     : {args.save_conf}")
    print(f"Save crop     : {args.save_crop}")
    print(f"Classes       : {classes if classes is not None else 'all'}")
    print("=" * 80 + "\n")

    if args.dry_run:
        print("Dry run mode: inference is not executed.")
        return

    model = YOLO(str(weights_path))
    model.predict(**predict_args)

    print("\n" + "=" * 80)
    print("Inference finished.")
    print(f"Results saved to: {project_dir / args.name}")
    print("=" * 80 + "\n")


# ============================================================
# 5. Argument parser
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO inference for the CWG-RGB benchmark."
    )

    parser.add_argument(
        "--weights",
        type=str,
        required=False,
        help=(
            "Path to trained model weights, for example: "
            "runs/train/960_bg_opt_yolov8s/weights/best.pt"
        ),
    )

    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help=(
            "Inference source. It can be an image, a folder or a video. "
            "Default: images/val"
        ),
    )

    parser.add_argument(
        "--preset",
        type=str,
        default="960_nms_calibrated",
        help=(
            "Inference preset. Use 'list' to show available presets. "
            "Default: 960_nms_calibrated"
        ),
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Input image size. If not set, use the selected preset."
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Confidence threshold. If not set, use the selected preset."
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=None,
        help="IoU threshold for NMS. If not set, use the selected preset."
    )

    parser.add_argument(
        "--max-det",
        dest="max_det",
        type=int,
        default=None,
        help="Maximum number of detections per image. If not set, use the selected preset."
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Inference device. Use '0' for GPU 0 or 'cpu' for CPU."
    )

    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Output project directory. Default: runs/predict"
    )

    parser.add_argument(
        "--name",
        type=str,
        default="cwg_inference",
        help="Output run name."
    )

    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save detection results as YOLO-format TXT files."
    )

    parser.add_argument(
        "--save-conf",
        action="store_true",
        help="Save confidence scores in TXT files. Only valid when --save-txt is used."
    )

    parser.add_argument(
        "--save-crop",
        action="store_true",
        help="Save cropped detection results."
    )

    parser.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="Line width of bounding boxes in saved images."
    )

    parser.add_argument(
        "--hide-labels",
        action="store_true",
        help="Hide class labels in saved images."
    )

    parser.add_argument(
        "--hide-conf",
        action="store_true",
        help="Hide confidence scores in saved images."
    )

    parser.add_argument(
        "--agnostic-nms",
        action="store_true",
        help="Use class-agnostic NMS."
    )

    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Only predict selected classes, for example: --classes 0,2,3"
    )

    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow existing output directory."
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print inference settings without running prediction."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.preset == "list":
        print_preset_list()
        return

    if args.weights is None:
        raise ValueError(
            "Please provide model weights using --weights.\n"
            "Example:\n"
            "python scripts/infer.py "
            "--weights runs/train/960_bg_opt_yolov8s/weights/best.pt"
        )

    run_inference(args)


if __name__ == "__main__":
    main()