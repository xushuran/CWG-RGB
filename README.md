# CWG-RGB Dataset

## Overview

CWG-RGB is a low-cost RGB image dataset for floating garbage detection in complex urban waterways. The dataset is designed for urban waterway inspection scenarios with small floating objects, complex water-surface backgrounds, and false-positive-prone non-target regions.

The dataset contains 1,760 RGB images, including 1,176 target-containing images and 584 background negative images. A total of 3,557 floating-garbage instances are annotated in YOLO format. Background negative images are provided with empty YOLO label files and are not counted as target instances.

## Dataset Statistics

| Item | Number |
|---|---:|
| Total images | 1,760 |
| Target-containing images | 1,176 |
| Background negative images | 584 |
| Annotated target instances | 3,557 |
| Number of classes | 5 |

## Class Definitions

The dataset contains five floating-garbage categories. The class IDs are defined as follows:

| Class ID | Class name |
|---:|---|
| 0 | Bags-Films |
| 1 | Wood |
| 2 | Foam |
| 3 | Bottles-Cans |
| 4 | Dead Fish |

Each YOLO annotation line follows this format:

```text
class_id x_center y_center width height
```

All bounding-box coordinates are normalized by the image width and height.

## Data Split

The target-containing images are divided into training and validation subsets. Background negative images are divided into training, validation, and reserved subsets.

| Subset | Number of images |
|---|---:|
| Target training images | 919 |
| Target validation images | 257 |
| Background negative images for training | 100 |
| Background negative images for validation | 30 |
| Reserved background negative images | 454 |

The main benchmark setting with background negative samples uses:

```text
train_with_bg.txt: 919 target training images + 100 background negative images
val_with_bg.txt: 257 target validation images + 30 background negative images
```

## Repository Structure

```text
CWG-RGB/
├── images/
│   ├── train/
│   ├── val/
│   └── background/
│       ├── train/
│       ├── val/
│       └── reserved/
│
├── labels/
│   ├── train/
│   ├── val/
│   └── background/
│       ├── train/
│       ├── val/
│       └── reserved/
│
├── splits/
│   ├── train_pos.txt
│   ├── val_pos.txt
│   ├── bg_train.txt
│   ├── bg_val.txt
│   ├── bg_reserved.txt
│   ├── train_with_bg.txt
│   └── val_with_bg.txt
│
├── configs/
│   ├── dataset.yaml
│   ├── data_no_bg.yaml
│   ├── data_with_bg.yaml
│   └── training_config.yaml
│
├── metadata/
│   ├── image_metadata.csv
│   ├── class_definitions.csv
│   └── dataset_summary.csv
│
├── scripts/
│   ├── train.py
│   ├── infer.py
│   └── evaluate.py
│
├── results/
│   └── main_results.csv
│
├── README.md
├── LICENSE.txt
├── VERSION.txt
└── requirements.txt
```

## Images and Annotations

All images are stored as RGB image files in the `images/` directory. Target-containing images are stored in `images/train/` and `images/val/`. Background negative images are stored under `images/background/`.

Each target-containing image has a corresponding YOLO-format TXT annotation file in the matching `labels/` directory. Background negative images have corresponding empty TXT label files under `labels/background/`. These empty label files indicate that the images contain no annotated garbage targets and are intentionally included as background negative samples rather than missing annotations.

Background negative images are not treated as an additional detection category and are not counted as target instances.

## Background Negative Samples

Background negative samples are included to represent target-free but false-positive-prone water-surface regions. These images contain no annotated garbage targets but include visually confusing non-target backgrounds, such as strong reflections, aquatic vegetation, leaves, natural floating objects, building reflections, ripples, turbid water, bridge shadows, and complex bank-side regions.

These samples are used as empty-label images in YOLO training and validation. They are intended to support false-positive analysis and background discrimination in complex urban waterway scenes.

## Metadata

The `metadata/` directory provides three CSV files for dataset reuse: `image_metadata.csv`, `class_definitions.csv`, and `dataset_summary.csv`.

`image_metadata.csv` records image-level information, including the image identifier, relative image path, split, scene type, viewpoint, target presence, target count, background-negative status, and privacy-screening status for each image. Background negative images are marked with `has_target = false` and `target_count = 0`.

`class_definitions.csv` records the class IDs, class names, category definitions, and annotation notes.

`dataset_summary.csv` provides dataset-level statistics, split information, configuration-file names, license information, and the dataset DOI.

## Configuration Files

Four configuration files are provided in the `configs/` directory:

```text
configs/dataset.yaml
configs/data_no_bg.yaml
configs/data_with_bg.yaml
configs/training_config.yaml
```

`configs/dataset.yaml` is the original YOLO dataset configuration used during model development.

`configs/data_no_bg.yaml` is provided for reproducible experiments without background negative samples. It uses only target-containing images for training and validation.

`configs/data_with_bg.yaml` is provided for reproducible experiments with background negative samples. It uses target-containing images and selected background negative images for training and validation.

`configs/training_config.yaml` records the benchmark training settings, model variants, evaluation settings, NMS calibration parameters, and reproduction commands.

Example of `configs/data_no_bg.yaml`:

```yaml
path: ../

train: splits/train_pos.txt
val: splits/val_pos.txt

names:
  0: Bags-Films
  1: Wood
  2: Foam
  3: Bottles-Cans
  4: Dead Fish
```

Example of `configs/data_with_bg.yaml`:

```yaml
path: ../

train: splits/train_with_bg.txt
val: splits/val_with_bg.txt

names:
  0: Bags-Films
  1: Wood
  2: Foam
  3: Bottles-Cans
  4: Dead Fish
```

The original development experiments used `dataset.yaml`, while `data_no_bg.yaml` and `data_with_bg.yaml` are provided to make the no-background and with-background experimental settings reproducible without manually moving background negative images.

## Benchmark Experiments

The benchmark scripts support the reference benchmark settings and auxiliary experiments described in the associated paper, including comparisons of input resolution, background negative samples, model variants, training-adapted settings, P2-head configurations, and NMS parameter calibration.

The evaluated model variants include:

```text
YOLOv8n
YOLOv8s
YOLO11n
YOLO11s
```

The supported benchmark and auxiliary experimental settings include:

```text
640 / without background + YOLOv8n
640 / without background + YOLOv8s
640 / without background + YOLO11n
640 / without background + YOLO11s
960 / without background + YOLOv8n
960 / without background + YOLOv8s
960 / without background + YOLO11n
960 / without background + YOLO11s

640 / with background + YOLOv8n
960 / with background + YOLOv8n
960 / with background + YOLOv8s
960 / with background + YOLO11n
960 / with background + YOLO11s

960 / with background + AdamW + YOLOv8n
960 / with background + AdamW + YOLOv8s
960 / with background + AdamW + YOLO11n
960 / with background + AdamW + YOLO11s

P2-head experiments
NMS-calibrated evaluation
```

For P2-head experiments, the model YAML names such as `yolov8s-p2.yaml` are resolved by Ultralytics from its built-in model configuration registry. These P2 configuration files are not stored as local files in the `configs/` directory.

## Main Benchmark Setting

All models were initialized with official Ultralytics pretrained weights. Unless otherwise specified, models were trained for 300 epochs with a batch size of 8 and 4 workers.

The main baseline setting is:

```text
Model: YOLOv8s
Dataset configuration: configs/data_with_bg.yaml
Input size: 960 × 960
Epochs: 300
Batch size: 8
Workers: 4
Optimizer: AdamW
Initial learning rate: 0.001
Weight decay: 0.0005
```

The calibrated NMS inference setting is:

```text
Confidence threshold: 0.35
IoU threshold: 0.45
Maximum detections: 300
```

## Software and Hardware Environment

Experiments were conducted on a Precision 7865 Tower workstation equipped with an AMD Ryzen Threadripper PRO 5975WX 32-Core CPU, 256 GB RAM, and a single NVIDIA RTX A5500 GPU with 24 GB memory.

The software environment was:

```text
Operating system: Windows 10
Python: 3.10.11
PyTorch: 2.11.0+cu126
CUDA: 12.6
Ultralytics YOLO: 8.4.37
```

## Reproduction

List all training experiments:

```bash
python scripts/train.py --exp list
```

Train the main baseline model:

```bash
python scripts/train.py --exp 960_bg_opt_yolov8s
```

Train the P2-head model:

```bash
python scripts/train.py --exp 960_bg_opt_yolov8s_p2
```

Evaluate the main baseline model:

```bash
python scripts/evaluate.py --exp 960_bg_opt_yolov8s
```

Evaluate the main baseline model with calibrated NMS parameters:

```bash
python scripts/evaluate.py --exp 960_bg_opt_yolov8s --preset nms_calibrated
```

Run NMS parameter calibration:

```bash
python scripts/evaluate.py --exp 960_bg_opt_yolov8s --nms-grid
```

Generate qualitative inference results:

```bash
python scripts/infer.py --weights runs/train/960_bg_opt_yolov8s/weights/best.pt --source images/val --preset 960_nms_calibrated
```

The scripts are provided to reproduce the reference benchmark settings and auxiliary experiments described in the associated paper.

## Data Availability

The CWG-RGB dataset and accompanying files have been deposited in Zenodo with a permanent DOI. The repository includes RGB images, YOLO-format annotations, empty-label background negative samples, split files, metadata CSV files, configuration files, benchmark scripts, and documentation.

Repository: Zenodo

Dataset DOI: 10.5281/zenodo.21261194

DOI URL: https://doi.org/10.5281/zenodo.21261194

Review URL: [Review URL to be provided for peer review]

## Code Availability

The training configurations, inference scripts, evaluation scripts, NMS settings, data checking utilities, and reproduction instructions are available at:

GitHub URL: https://github.com/xushuran/CWG-RGB

The benchmark scripts are also included in the Zenodo dataset archive associated with DOI 10.5281/zenodo.21261194.

## License

This dataset is released under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

Please refer to `LICENSE.txt` for detailed license information.

## Version

Current version: v1.0.0

Please refer to `VERSION.txt` for detailed version information.

## Privacy and Ethical Use

Images containing privacy-sensitive or research-irrelevant background information were screened before release. Users should not attempt to identify individuals, locations, or other non-research-related information that may appear in the images. The dataset should be used only for lawful and ethical research or application development.

## Citation

If you use this dataset or benchmark, please cite the associated paper and dataset repository DOI.

Dataset DOI: 10.5281/zenodo.21261194

DOI URL: https://doi.org/10.5281/zenodo.21261194

GitHub repository: https://github.com/xushuran/CWG-RGB

```text
[Citation information to be updated after publication]
```
