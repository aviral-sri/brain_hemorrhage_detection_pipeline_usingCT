# Slice-Based 2D CNN for CT Hemorrhage Detection

A lightweight 2D CNN trained on individual CT slices to detect intracranial hemorrhage (ICH).

---

## Datasets

- **Training:** [CT-ICH — PhysioNet](https://physionet.org/content/ct-ich/1.3.1/ct_scans/) — 75 NIfTI volumes with segmentation masks
- **Evaluation:** [CQ500 — Kaggle](https://www.kaggle.com/datasets/crawford/qureai-headct) — DICOM scans with patient-level ICH labels

---

## How it works

Each CT volume is split into 2D axial slices. A slice is labeled positive if its segmentation mask contains any hemorrhage pixels. At inference time, a scan is called positive if any single slice exceeds a 0.5 probability threshold.

---

## Model Architecture

```
Input: (1 × 256 × 256)  — single grayscale CT slice, brain window [-40, 80] HU

Conv2d(1 → 16, 3×3)  + ReLU + MaxPool2d(2)   →  (16 × 128 × 128)
Conv2d(16 → 32, 3×3) + ReLU + MaxPool2d(2)   →  (32 × 64 × 64)
Conv2d(32 → 64, 3×3) + ReLU + MaxPool2d(2)   →  (64 × 32 × 32)

Flatten  →  65536
FC(65536 → 128) + ReLU
FC(128 → 2)     →  [normal, hemorrhage]
```

Trained with CrossEntropyLoss, Adam (lr=1e-3), 10 epochs, batch size 64.

---

## 2D Slice CNN vs 3D CNN

| | 3D CNN | 2D Slice CNN (this model) |
|---|---|---|
| Input | Full volume `(128, 256, 256)` | Per-slice `(1, 256, 256)` |
| Labels | Patient-level (noisy) | Mask-derived per slice (clean) |
| Data leakage | Yes — slices from same patient in train and val | No |
| Internal accuracy | Near-perfect (inflated by leakage) | 0.96 |
| CQ500 recall (ICH) | Poor generalisation | 0.78 |
| Training speed | Slow | Fast |

The 3D CNN appeared to perform well internally but suffered from data leakage — slices from the same patient ended up in both training and validation sets, inflating metrics. The slice CNN uses clean mask-derived labels and a proper split, which led to more honest and better cross-dataset generalisation despite being the simpler model.
