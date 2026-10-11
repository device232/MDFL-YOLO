import pytest
import torch
from ultralytics.nn.tasks import DetectionModel

from mdfl_yolo.loss import MDFLDetectionLoss
from mdfl_yolo.model import LAYER_MAP, MDFLDetectionModel, VARIANTS, baseline_config, save_untrained


@pytest.mark.parametrize("variant", VARIANTS)
def test_all_eight_graphs(variant):
    model = MDFLDetectionModel(nc=4, variant=variant).eval()
    assert len(model.model) == 27
    assert model.model[-1].f == [19, 22, 25]
    assert model.stride.tolist() == [8., 16., 32.]
    with torch.no_grad():
        decoded, raw = model(torch.zeros(1, 3, 64, 64))
    assert decoded.shape == (1, 8, 84)
    assert [v.shape[-2:] for v in raw] == [(8, 8), (4, 4), (2, 2)]


def test_coco_layer_remapping():
    source = DetectionModel(cfg=baseline_config(), nc=80, verbose=False)
    target = MDFLDetectionModel(nc=4)
    new_module_before = {key: value.clone() for key, value in target.model[3].state_dict().items()}
    target.load(source, verbose=False)
    target_state = target.state_dict()
    for key, value in source.state_dict().items():
        parts = key.split(".")
        parts[1] = str(LAYER_MAP[int(parts[1])])
        mapped = ".".join(parts)
        if mapped in target_state and target_state[mapped].shape == value.shape:
            torch.testing.assert_close(target_state[mapped], value)
    assert target.pretrained_report["loaded_tensors"] > 400
    for key, value in target.model[3].state_dict().items():
        torch.testing.assert_close(value, new_module_before[key])


def test_identity_ablation_matches_baseline_forward():
    source = DetectionModel(cfg=baseline_config(), nc=4, verbose=False).eval()
    target = MDFLDetectionModel(nc=4, variant="baseline").eval()
    target.load(source, verbose=False)
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        expected = source(image)[0]
        actual = target(image)[0]
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("with_target", [True, False])
def test_integrated_training_loss(with_target):
    model = MDFLDetectionModel().train()
    batch = {
        "img": torch.rand(2, 3, 128, 128),
        "batch_idx": torch.tensor([0]) if with_target else torch.empty(0, dtype=torch.long),
        "cls": torch.tensor([[0.]]) if with_target else torch.empty(0, 1),
        "bboxes": torch.tensor([[0.5, 0.5, 0.6, 0.2]]) if with_target else torch.empty(0, 4),
    }
    total, components = model(batch)
    assert isinstance(model.criterion, MDFLDetectionLoss)
    assert torch.isfinite(total) and torch.isfinite(components).all()
    total.backward()
    assert model.model[3].alpha.grad is not None
    if not with_target:
        assert components[0] == 0 and components[2] == 0
    else:
        assert components[0] > 0 and components[2] > 0


def test_saving_loading_and_loss_only_parameter_count(tmp_path):
    from mdfl_yolo.cli import _model_from_checkpoint
    model = MDFLDetectionModel().eval()
    without_loss = MDFLDetectionModel(variant="dpe-ocdf")
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in without_loss.parameters())
    path = tmp_path / "untrained.pt"
    save_untrained(model, path)
    loaded = _model_from_checkpoint(path).model.eval()
    image = torch.zeros(1, 3, 64, 64)
    with torch.no_grad():
        torch.testing.assert_close(model(image)[0], loaded(image)[0])
