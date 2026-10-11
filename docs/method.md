# Method and experiment settings

## DPE

DPE follows backbone P2 and P3 at layers 3 and 6. At 640-pixel input, their shapes are `64 × 160 × 160` and `128 × 80 × 80`.

Each instance reduces channels by four with Conv–BN–SiLU, then evaluates dense `1 × 5`, `5 × 1`, and `3 × 3` Conv–BN–SiLU branches. A gate reads the original input, applies global average pooling, biased `1 × 1` convolutions with widths `C → max(C/8, 8) → 3`, an intermediate SiLU, and a branch Softmax. The three scalar weights apply image-wide.

The weighted mixture is expanded with Conv–BN without activation and added to the input with an unconstrained learned scalar initialized to 0.1. Spatial convolutions use groups one. BN uses epsilon 0.001 and momentum 0.03.

## OCDF

Layer 19 receives enhanced P2 from layer 3 and the semantic neck P3 from layer 18. The two P2 branches use depthwise `1 × 5` and `5 × 1` convolutions with stride `(2, 2)`, BN–SiLU, then dense `1 × 1` Conv–BN–SiLU projections. A `64 → 8 → 2` image-level gate reads P3 and weights these oriented detail responses.

Concatenated semantic and mixed detail features feed a bias-free `128 → 64` Conv–BN–Sigmoid gate. The result is `S + beta * G * D`, with an unconstrained learned `beta` initialized to 0.02. P3 is the identity semantic path.

The Detect layer consumes `[19, 22, 25]`, preserving strides `[8, 16, 32]`. Ablations replace disabled feature modules with parameter-free identities while retaining stable layer references.

## Pretrained layer ownership

| Baseline YOLO11n layer | MDFL parameter-bearing layer |
| --- | --- |
| 0, 1, 2 | 0, 1, 2 |
| 3, 4 | 4, 5 |
| 5, 6, 7, 8, 9, 10 | 7, 8, 9, 10, 11, 12 |
| 11, 12, 13, 14, 15, 16 | 13, 14, 15, 16, 17, 18 |
| 17, 18, 19, 20, 21, 22, 23 | 20, 21, 22, 23, 24, 25, 26 |

New DPE/OCDF parameters retain their constructor initialization. Weight mapping requires matching tensor names and shapes. Existing detect biases receive the framework's stride-dependent initialization before compatible tensors are loaded.

## Slender-axis objective

Predicted and assigned target boxes use decoded xyxy coordinates in feature-grid units. Widths and heights are clamped at `1e-6`.

For target log aspect `a = log(w_t/h_t)`, define:

```text
u = sigmoid(a / 0.5)
g = sigmoid(2 * (abs(a) - log(5)))
e_center,j = SmoothL1((c_j - c_t,j) / s_t,j, beta=0.5)
e_scale,j  = SmoothL1(log(s_j / s_t,j), beta=0.5)
ell = u * (e_center,y + e_scale,y)
    + (1-u) * (e_center,x + e_scale,x)
q_i = sum_k detached_target_score[i,k]
Z = max(sum_i,k detached_target_score[i,k], 1)
L_SA = sum_foreground(q_i * g_i * ell_i) / Z
L = 7.5 * (L_CIoU + 0.1 * L_SA) + 0.5 * L_BCE + 1.5 * L_DFL
```

The normalizer excludes elongation weights. The inherited task-aligned assigner uses top-10 candidates, classification exponent 0.5, and overlap exponent 6.0. BCE, DFL, and assignment remain in Ultralytics. The criterion applies its existing final mini-batch multiplier; logged components omit that multiplier. With no foreground, box and DFL loss are zero and classification stays active. The auxiliary term introduces no inference layers.

## Training protocol

`trainer.py` contains the 300-epoch training recipe: SGD LR 0.01, momentum 0.937, weight decay 0.0005, final LR factor 0.01, three warmup epochs, nominal batch 64, and the augmentation probabilities described in Section 4.1. The inherited SGD optimizer uses Nesterov. Mosaic closes for the final ten epochs. Early stopping is disabled to retain the scheduled budget.

Throughput measurement uses FP32. Training AMP is controlled by `--amp` and defaults off. Supply dataset class mappings and the fixed split lists with the data.
