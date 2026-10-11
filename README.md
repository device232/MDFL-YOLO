# MDFL-YOLO

Code for **MDFL-YOLO: morphology-guided feature enhancement, detail fusion and localization for cigarette inner-liner defect detection**.

MDFL-YOLO adds direction-aware feature enhancement (DPE), orientation-consistent detail fusion (OCDF), and a training-only slender-axis localization loss to YOLO11n. Predictions retain the P3/P4/P5 scales and strides 8/16/32.

## Installation

Python 3.10 or 3.11 is recommended. The training stack uses PyTorch 2.5.1, Torchvision 0.20.1, and Ultralytics 8.3.9.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# For the CUDA 12.1 environment used in the paper:
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e '.[test]'

# Run correctness checks:
pytest -q
```

For CPU execution, replace `cu121` with `cpu` in the PyTorch installation command and use `--device cpu` in the commands below. Install the package before loading its checkpoints so the custom model classes are available.

## Dataset preparation

Prepare a YOLO detection dataset with matching `images/` and `labels/` paths. Each annotation row contains `class_id center_x center_y width height`, with coordinates normalized to the image dimensions. Negative images have empty labels. Split lists contain image paths; use the same fixed lists for every seed and model variant.

```text
dataset/
  images/
  labels/
  splits/
    train.txt
    val.txt
    test.txt
```

Copy and edit one of the dataset configuration examples:

- `configs/inner_liner.example.yaml`: four inner-liner classes.
- `configs/gc10_det.example.yaml`: ten GC10-DET classes.
- `configs/neu_det.example.yaml`: six NEU-DET classes.

The example class IDs must agree with your annotation files. The private inner-liner images, fixed split lists, and trained weights are not included in this repository.

## Train

Provide a local COCO-pretrained YOLO11n checkpoint. Compatible backbone, neck, and head tensors are loaded through an explicit baseline-to-MDFL layer map. New module tensors and incompatible classification tensors keep their initialization.

```bash
mdfl-yolo train \
  --data configs/inner_liner.example.yaml \
  --pretrained /path/to/yolo11n.pt \
  --variant mdfl --seed 2025 --device 0
```

Repeat with seeds `3407` and `42`. The default recipe uses 640-pixel inputs, batch 16, eight workers, 300 epochs, Nesterov SGD, and validation-based selection. `best.pt` is saved under `runs/train/<variant>-seed<seed>/weights/`. Training uses FP32 by default; `--amp` enables mixed precision explicitly. Use `--from-scratch` in place of `--pretrained` for a run without COCO initialization.

All eight ablation conditions are available through `--variant`:

| Variant | DPE | OCDF | Slender-axis loss |
| --- | --- | --- | --- |
| `baseline` | | | |
| `dpe` | ✓ | | |
| `ocdf` | | ✓ | |
| `sa` | | | ✓ |
| `dpe-ocdf` | ✓ | ✓ | |
| `dpe-sa` | ✓ | | ✓ |
| `ocdf-sa` | | ✓ | ✓ |
| `mdfl` | ✓ | ✓ | ✓ |

## Evaluate and predict

Select checkpoints using validation, then evaluate the fixed test split:

```bash
mdfl-yolo val \
  --weights runs/train/mdfl-seed2025/weights/best.pt \
  --data configs/inner_liner.example.yaml --split test --device 0

mdfl-yolo predict \
  --weights runs/train/mdfl-seed2025/weights/best.pt \
  --source /path/to/images --device 0
```

Evaluation uses confidence 0.001, NMS IoU 0.7, and at most 300 detections per image. Prediction examples use confidence 0.25 by default.

## Inspect the architecture

```bash
mdfl-yolo build --variant mdfl --nc 4 --output weights/mdfl-yolo-untrained.pt
```

This command saves an **untrained** architecture checkpoint for inspection. Training produces the usable detection weights.

## Forward throughput

```bash
mdfl-yolo benchmark \
  --weights runs/train/mdfl-seed2025/weights/best.pt \
  --batch 16 --device cuda:0
```

The benchmark uses FP32 GPU-resident 640 × 640 zero tensors, disables TF32, performs 50 warmup calls, and times 200 forward calls with GPU synchronization. It includes internal decoding and excludes input preprocessing, transfers, and final NMS. Reported ms/image is amortized batch time. The paper used an RTX 4090 D.

## Code layout

```text
mdfl_yolo/
  modules.py       DPE and OCDF
  loss.py          slender-axis objective and criterion integration
  model.py         network topology and pretrained tensor mapping
  trainer.py       training integration and experiment settings
  cli.py           build, train, val, predict, benchmark
  yolo11n.yaml     baseline blueprint
configs/           dataset examples
tests/             module, geometry, model, and checkpoint checks
docs/method.md     architecture and loss details
docs/upload.md     GitHub upload instructions
```

## Dependencies

This project builds on PyTorch, Torchvision, and Ultralytics. Their respective distribution and licensing terms apply to those dependencies.
