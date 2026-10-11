"""MDFL-YOLO's 27-layer graph with explicit YOLO11n weight mapping."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from torch import nn
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT

from .loss import MDFLDetectionLoss
from .modules import DPE, OCDF, SemanticIdentity

VARIANTS = {
    "baseline": (False, False, False),
    "dpe": (True, False, False),
    "ocdf": (False, True, False),
    "sa": (False, False, True),
    "dpe-ocdf": (True, True, False),
    "dpe-sa": (True, False, True),
    "ocdf-sa": (False, True, True),
    "mdfl": (True, True, True),
}
BLUEPRINT = Path(__file__).with_name("yolo11n.yaml")
# Tensor ownership: original baseline layer -> actual parameter-bearing new layer.
LAYER_MAP = {
    0: 0, 1: 1, 2: 2, 3: 4, 4: 5, 5: 7, 6: 8, 7: 9,
    8: 10, 9: 11, 10: 12, 11: 13, 12: 14, 13: 15, 14: 16,
    15: 17, 16: 18, 17: 20, 18: 21, 19: 22, 20: 23, 21: 24,
    22: 25, 23: 26,
}


def baseline_config():
    return yaml.safe_load(BLUEPRINT.read_text(encoding="utf-8"))


def _decorate(module, index, source, name=None):
    module.i = index
    module.f = source
    module.type = name or type(module).__name__
    module.np = sum(p.numel() for p in module.parameters())
    return module


class MDFLDetectionModel(DetectionModel):
    def __init__(self, nc=4, variant="mdfl", verbose=False):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}; choose from {tuple(VARIANTS)}")
        super().__init__(cfg=baseline_config(), ch=3, nc=nc, verbose=verbose)
        self.variant = variant
        dpe, ocdf, self.use_slender_axis = VARIANTS[variant]
        old_layers = list(self.model)
        layers = []
        # References point to enhanced outputs; weights map to original ownership.
        output_map = dict(LAYER_MAP)
        output_map.update({2: 3, 4: 6, 16: 19})
        for old_index, layer in enumerate(old_layers):
            source = layer.f
            if isinstance(source, int):
                source = -1 if source == -1 else output_map[source]
            else:
                source = [-1 if s == -1 else output_map[s] for s in source]
            layers.append(_decorate(layer, len(layers), source, layer.type))
            if old_index == 2:
                layers.append(_decorate(DPE(64) if dpe else nn.Identity(), 3, -1))
            elif old_index == 4:
                layers.append(_decorate(DPE(128) if dpe else nn.Identity(), 6, -1))
            elif old_index == 16:
                layers.append(_decorate(OCDF() if ocdf else SemanticIdentity(), 19, [3, 18]))
        self.model = nn.Sequential(*layers)
        self.save = sorted({
            source for layer in layers
            for source in ([layer.f] if isinstance(layer.f, int) else layer.f)
            if source != -1
        })
        self.yaml["mdfl_variant"] = variant
        self.yaml["mdfl_layer_map"] = dict(LAYER_MAP)
        args = deepcopy(DEFAULT_CFG_DICT)
        args.update(task="detect", model="mdfl-yolo", box=7.5, cls=0.5, dfl=1.5)
        self.args = SimpleNamespace(**args)
        if self.model[-1].f != [19, 22, 25]:
            raise RuntimeError("Unexpected detection graph")

    def init_criterion(self):
        return MDFLDetectionLoss(self) if self.use_slender_axis else super().init_criterion()

    def load(self, weights, verbose=True):
        """Load MDFL checkpoints directly, or remap standard YOLO11n tensors."""
        source = weights.get("model") if isinstance(weights, dict) else weights
        if not isinstance(source, nn.Module):
            raise TypeError("weights must be a model or checkpoint containing a model")
        if isinstance(source, MDFLDetectionModel):
            if source.variant != self.variant:
                raise ValueError("Checkpoint variant differs from requested variant")
            return super().load(source, verbose=verbose)
        state = source.float().state_dict()
        destination = self.state_dict()
        transferred = {}
        skipped = []
        for key, value in state.items():
            parts = key.split(".")
            if len(parts) < 3 or parts[0] != "model" or not parts[1].isdigit():
                skipped.append(key)
                continue
            old_index = int(parts[1])
            if old_index not in LAYER_MAP:
                skipped.append(key)
                continue
            parts[1] = str(LAYER_MAP[old_index])
            mapped = ".".join(parts)
            if mapped in destination and value.shape == destination[mapped].shape:
                transferred[mapped] = value
            else:
                skipped.append(key)
        if not transferred:
            raise ValueError("No compatible YOLO11n tensors found in checkpoint")
        self.load_state_dict(transferred, strict=False)
        self.pretrained_report = {"loaded_tensors": len(transferred), "skipped_source_keys": skipped}
        if verbose:
            print(f"Remapped {len(transferred)} tensors; skipped {len(skipped)} incompatible tensors")
        return self


def save_untrained(model, path):
    """Create an architecture checkpoint for inspection; this is not a trained weight release."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": deepcopy(model).cpu().float().eval(),
        "epoch": -1,
        "best_fitness": None,
        "optimizer": None,
        "train_args": {"task": "detect", "imgsz": 640, "model": str(path)},
        "trained": False,
    }, path)
