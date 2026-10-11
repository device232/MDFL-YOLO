import math

import torch

from mdfl_yolo.loss import SlenderAxisLoss


def test_zero_loss_and_empty_foreground():
    boxes = torch.tensor([[[0., 0., 10., 1.]]], requires_grad=True)
    scores = torch.ones(1, 1, 1)
    positive = torch.ones(1, 1, dtype=torch.bool)
    loss = SlenderAxisLoss()
    torch.testing.assert_close(loss(boxes, boxes, scores, positive), torch.tensor(0.))
    empty = loss(boxes, boxes, scores, ~positive)
    empty.backward()
    assert boxes.grad is not None and torch.isfinite(boxes.grad).all()


def test_analytic_short_axis_penalty_and_score_normalizer():
    target = torch.tensor([[[0., 0., 10., 1.]]])
    prediction = target + torch.tensor([0., 0.25, 0., 0.25])
    scores = torch.tensor([[[0.4, 0.2]]], requires_grad=True)
    actual = SlenderAxisLoss()(prediction, target, scores, torch.tensor([[True]]))
    aspect = math.log(10.)
    u = 1. / (1. + math.exp(-aspect / 0.5))
    g = 1. / (1. + math.exp(-2. * (aspect - math.log(5.))))
    expected = 0.6 * g * u * (0.25 ** 2 / (2. * 0.5))
    torch.testing.assert_close(actual, torch.tensor(expected))
    assert not actual.requires_grad  # detached target scores are not optimized.


def test_axis_swap_symmetry_and_short_axis_emphasis():
    loss = SlenderAxisLoss()
    target = torch.tensor([[[0., 0., 20., 1.]]])
    shift_x = target + torch.tensor([0.2, 0., 0.2, 0.])
    shift_y = target + torch.tensor([0., 0.2, 0., 0.2])
    scores = torch.ones(1, 1, 1)
    mask = torch.tensor([[True]])
    assert loss(shift_y, target, scores, mask) > loss(shift_x, target, scores, mask)
    torch.testing.assert_close(
        loss(shift_y, target, scores, mask),
        loss(shift_y[..., [1, 0, 3, 2]], target[..., [1, 0, 3, 2]], scores, mask),
    )


def test_finite_gradients_for_degenerate_boxes():
    prediction = torch.zeros(1, 1, 4, requires_grad=True)
    target = torch.zeros_like(prediction)
    value = SlenderAxisLoss()(prediction, target, torch.ones(1, 1, 1), torch.tensor([[True]]))
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(prediction.grad).all()
