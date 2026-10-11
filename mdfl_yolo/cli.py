"""Command-line entry points for model inspection, training, evaluation and inference."""

import argparse
import json
import time
from pathlib import Path

import torch

from .model import BLUEPRINT, MDFLDetectionModel, VARIANTS, save_untrained


def _model_from_checkpoint(path):
    from ultralytics import YOLO
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    detector = YOLO(str(path), task="detect")
    if not isinstance(detector.model, MDFLDetectionModel):
        raise ValueError("Expected a checkpoint produced by mdfl-yolo")
    return detector


def _train(args):
    from .trainer import MDFLTrainer, PAPER_TRAIN_ARGS
    if args.pretrained and not Path(args.pretrained).is_file():
        raise FileNotFoundError(args.pretrained)
    options = dict(PAPER_TRAIN_ARGS)
    options.update(
        model=str(BLUEPRINT), data=args.data, seed=args.seed, device=args.device,
        epochs=args.epochs, batch=args.batch, workers=args.workers,
        project=args.project, name=args.name or f"{args.variant}-seed{args.seed}",
        amp=args.amp, val=True, plots=True,
    )
    # Custom attributes must survive DDP subprocess creation, which the stock
    # trainer does not guarantee. Restrict this entry point to one device.
    if "," in str(args.device):
        raise ValueError("Use one GPU per run; launch different seeds as separate runs")
    trainer = MDFLTrainer(
        overrides=options, variant=args.variant, pretrained_path=args.pretrained,
    )
    trainer.train()


def _evaluate(args):
    model = _model_from_checkpoint(args.weights)
    metrics = model.val(
        data=args.data, split=args.split, imgsz=640, batch=args.batch,
        workers=args.workers, device=args.device, conf=0.001,
        iou=0.7, max_det=300, augment=False, plots=True,
        project=args.project, name=args.name,
    )
    print(json.dumps(metrics.results_dict, indent=2))


def _predict(args):
    model = _model_from_checkpoint(args.weights)
    for result in model.predict(
        source=args.source, imgsz=640, conf=args.conf, iou=0.7,
        max_det=300, device=args.device, save=True, save_txt=True,
        project=args.project, name=args.name, stream=True,
    ):
        print(f"{result.path}: {len(result.boxes)} detections")


def _benchmark(args):
    if not torch.cuda.is_available():
        raise RuntimeError("The paper's benchmark protocol requires a CUDA GPU")
    model = _model_from_checkpoint(args.weights).model.to(args.device).float().eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    images = torch.zeros(args.batch, 3, 640, 640, device=args.device, dtype=torch.float32)
    with torch.inference_mode():
        for _ in range(50):
            model(images)
        torch.cuda.synchronize(images.device)
        start = time.perf_counter()
        for _ in range(200):
            model(images)
        torch.cuda.synchronize(images.device)
        duration = time.perf_counter() - start
    output = {
        "gpu": torch.cuda.get_device_name(images.device), "precision": "FP32",
        "batch": args.batch, "warmup_calls": 50, "timed_calls": 200,
        "fps": 200 * args.batch / duration,
        "amortized_ms_per_image": 1000 * duration / (200 * args.batch),
        "parameters": sum(p.numel() for p in model.parameters()),
        "includes": "network forward and internal decoding",
        "excludes": "image loading, preprocessing, transfer, final NMS",
    }
    print(json.dumps(output, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description="MDFL-YOLO")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Save an untrained architecture checkpoint")
    build.add_argument("--variant", choices=VARIANTS, default="mdfl")
    build.add_argument("--nc", type=int, default=4)
    build.add_argument("--output", default="weights/mdfl-yolo-untrained.pt")

    train = sub.add_parser("train", help="Train a model using the paper's recipe")
    train.add_argument("--data", required=True)
    train.add_argument("--variant", choices=VARIANTS, default="mdfl")
    initialization = train.add_mutually_exclusive_group(required=True)
    initialization.add_argument("--pretrained", help="Local YOLO11n COCO .pt checkpoint")
    initialization.add_argument("--from-scratch", action="store_true")
    train.add_argument("--seed", type=int, choices=(2025, 3407, 42), default=2025)
    train.add_argument("--epochs", type=int, default=300)
    train.add_argument("--batch", type=int, default=16)
    train.add_argument("--workers", type=int, default=8)
    train.add_argument("--device", default="0")
    train.add_argument("--project", default="runs/train")
    train.add_argument("--name")
    # Training AMP was not specified in the PDF: expose it as an explicit option.
    train.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)

    evaluate = sub.add_parser("val", help="Evaluate a checkpoint on a fixed split")
    evaluate.add_argument("--weights", required=True)
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--split", choices=("val", "test"), default="test")
    evaluate.add_argument("--batch", type=int, default=16)
    evaluate.add_argument("--workers", type=int, default=8)
    evaluate.add_argument("--device", default="0")
    evaluate.add_argument("--project", default="runs/val")
    evaluate.add_argument("--name", default="mdfl-yolo")

    predict = sub.add_parser("predict", help="Save predictions and YOLO-format boxes")
    predict.add_argument("--weights", required=True)
    predict.add_argument("--source", required=True)
    predict.add_argument("--conf", type=float, default=0.25)
    predict.add_argument("--device", default="0")
    predict.add_argument("--project", default="runs/predict")
    predict.add_argument("--name", default="mdfl-yolo")

    benchmark = sub.add_parser("benchmark", help="FP32 network-forward throughput")
    benchmark.add_argument("--weights", required=True)
    benchmark.add_argument("--batch", type=int, default=16)
    benchmark.add_argument("--device", default="cuda:0")

    args = parser.parse_args(argv)
    if args.command == "build":
        if args.nc < 1:
            parser.error("--nc must be positive")
        torch.set_num_threads(min(torch.get_num_threads(), 4))
        model = MDFLDetectionModel(nc=args.nc, variant=args.variant)
        save_untrained(model, args.output)
        print(json.dumps({
            "checkpoint": str(Path(args.output).resolve()), "variant": args.variant,
            "parameters": sum(p.numel() for p in model.parameters()),
            "detect_inputs": model.model[-1].f, "strides": model.stride.tolist(),
        }, indent=2))
    elif args.command == "train":
        _train(args)
    elif args.command == "val":
        _evaluate(args)
    elif args.command == "predict":
        _predict(args)
    elif args.command == "benchmark":
        _benchmark(args)


if __name__ == "__main__":
    main()
