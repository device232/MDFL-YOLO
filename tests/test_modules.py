import torch

from mdfl_yolo.modules import DPE, OCDF


def test_dpe_residual_gate_and_gradients():
    model = DPE(64)
    x = torch.randn(2, 64, 20, 20, requires_grad=True)
    weights = model.gate(x)
    assert weights.shape == (2, 3, 1, 1)
    torch.testing.assert_close(weights.sum(1), torch.ones_like(weights[:, 0]))
    output = model(x)
    assert output.shape == x.shape
    output.square().mean().backward()
    assert model.alpha.grad is not None and model.alpha.grad.abs() > 0
    assert model.horizontal[0].weight.grad.abs().sum() > 0
    with torch.no_grad():
        model.alpha.zero_()
    torch.testing.assert_close(model(x), x)


def test_ocdf_semantic_identity_and_gradients():
    model = OCDF()
    p2 = torch.randn(2, 64, 40, 40, requires_grad=True)
    p3 = torch.randn(2, 64, 20, 20, requires_grad=True)
    output = model([p2, p3])
    assert output.shape == p3.shape
    output.square().mean().backward()
    assert model.beta.grad is not None and model.beta.grad.abs() > 0
    assert p2.grad.abs().sum() > 0
    assert p3.grad.abs().sum() > 0
    with torch.no_grad():
        model.beta.zero_()
    torch.testing.assert_close(model([p2, p3]), p3)
