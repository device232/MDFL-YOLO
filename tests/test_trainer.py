import numpy as np
import yaml
from PIL import Image

from mdfl_yolo.model import BLUEPRINT, MDFLDetectionModel
from mdfl_yolo.trainer import MDFLTrainer, PAPER_TRAIN_ARGS


def test_one_epoch_with_synthetic_images(tmp_path):
    """Exercise the real trainer, optimizer, validator, and checkpoint writer."""
    for split in ("train", "val"):
        image_dir = tmp_path / "images" / split
        label_dir = tmp_path / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for index in range(2):
            pixels = np.random.default_rng(index).integers(0, 256, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(image_dir / f"{index}.png")
            (label_dir / f"{index}.txt").write_text("0 0.5 0.4375 0.6 0.08\n", encoding="utf-8")
    config = tmp_path / "data.yaml"
    config.write_text(yaml.safe_dump({
        "path": str(tmp_path), "train": "images/train", "val": "images/val",
        "names": {0: "line"},
    }), encoding="utf-8")
    options = dict(PAPER_TRAIN_ARGS)
    options.update(
        model=str(BLUEPRINT), data=str(config), imgsz=64,
        epochs=1, batch=2, workers=0, device="cpu", amp=False,
        mosaic=0.0, close_mosaic=0, plots=False,
        project=str(tmp_path / "runs"), name="smoke", verbose=False,
    )
    trainer = MDFLTrainer(overrides=options, variant="mdfl")
    trainer.train()
    assert isinstance(trainer.model, MDFLDetectionModel)
    assert trainer.best.is_file() and trainer.last.is_file()
    from mdfl_yolo.cli import _model_from_checkpoint
    loaded = _model_from_checkpoint(trainer.best)
    assert loaded.model.variant == "mdfl"
    assert loaded.model.model[-1].nc == 1
