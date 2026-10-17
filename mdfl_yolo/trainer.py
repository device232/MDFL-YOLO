"""Ultralytics integration and the training recipe in manuscript Section 4.1."""

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import attempt_load_one_weight

from .model import MDFLDetectionModel

PAPER_TRAIN_ARGS = {
    "epochs": 300, "imgsz": 640, "batch": 16, "workers": 8,
    "optimizer": "SGD", "lr0": 0.01, "lrf": 0.01, "momentum": 0.937,
    "weight_decay": 0.0005, "warmup_epochs": 3.0, "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1, "nbs": 64, "cos_lr": False,
    "box": 7.5, "cls": 0.5, "dfl": 1.5,
    "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
    "translate": 0.1, "scale": 0.5, "fliplr": 0.5, "mosaic": 1.0,
    "close_mosaic": 10, "degrees": 0.0, "shear": 0.0, "perspective": 0.0,
    "flipud": 0.0, "mixup": 0.0, "copy_paste": 0.0, "multi_scale": False,
    "rect": False, "seed": 2025, "pretrained": False,
    # Stop after 100 epochs without validation-fitness improvement.
    "patience": 100,
}


class MDFLTrainer(DetectionTrainer):
    def __init__(self, *args, variant="mdfl", pretrained_path=None, **kwargs):
        self.mdfl_variant = variant
        self.pretrained_path = pretrained_path
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        variant = cfg.get("mdfl_variant", self.mdfl_variant) if isinstance(cfg, dict) else self.mdfl_variant
        model = MDFLDetectionModel(nc=self.data["nc"], variant=variant, verbose=verbose)
        if weights is not None:
            model.load(weights, verbose=verbose)
        elif self.pretrained_path:
            pretrained, _ = attempt_load_one_weight(self.pretrained_path, device="cpu", fuse=False)
            model.load(pretrained, verbose=verbose)
        return model
