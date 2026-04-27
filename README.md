# Brain Tumor Detection & Segmentation using Generative Models

---

## Overview

This project tackles **unsupervised anomaly detection** and **supervised tumor segmentation** on brain MRI scans. The pipeline spans three interconnected notebooks:

1. **VAE (Variational Autoencoder)** — learns the distribution of healthy brain tissue, then flags tumors as high-reconstruction-error regions.
2. **GAN (Wasserstein GAN)** — trains exclusively on normal MRI slices; anomalies are detected by subtracting generator reconstructions from real inputs.
3. **UNet** — a segmentation network trained in three modes: image-only baseline, VAE-heatmap guided, and GAN-heatmap guided.

---

## Dataset

**LGG MRI Segmentation** — Lower-Grade Glioma brain MRI slices with pixel-level tumor masks.

| Source | Link |
|--------|------|
| Kaggle | [mateuszbuda/lgg-mri-segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) |
| Format | `.tif` image + `_mask.tif` binary mask pairs |

Images are classified into two groups based on mask content:

- **Normal** — masks contain only black pixels (no tumor)
- **Abnormal** — masks contain white pixels (tumor present)

---

## Project Structure

```
.
├── VAE.ipynb       # Anomaly detection via Variational Autoencoder
├── GAN.ipynb       # Anomaly detection via Wasserstein GAN
└── UNet.ipynb      # Tumor segmentation (baseline + heatmap-guided)
```

---

## Pipeline

```
MRI Dataset
    │
    ├── Normal slices ──────────────────────────────────────────────────┐
    │                                                                    │
    │   ┌─── VAE Training (150 epochs) ────────────────────────────┐   │
    │   │  Learns P(normal MRI)                                     │   │
    │   │  Reconstruction error → anomaly heatmap (.npy)            │   │
    │   └───────────────────────────────────────────────────────────┘   │
    │                                                                    │
    │   ┌─── WGAN Training (100 epochs) ───────────────────────────┐   │
    │   │  Generator + Critic on normal slices                      │   │
    │   │  |input − G(z*)| → anomaly heatmap (.npy)                │   │
    │   └───────────────────────────────────────────────────────────┘   │
    │                                                                    │
    └── Abnormal slices ─────────────────────────────────────────────┐  │
                                                                      ▼  ▼
                                              ┌─── UNet Segmentation ────┐
                                              │  Baseline  (1-ch input)  │
                                              │  VAE-guided (2-ch input) │
                                              │  GAN-guided (2-ch input) │
                                              └──────────────────────────┘
```

---

## Models

### Variational Autoencoder (VAE)

- **Architecture**: 4-layer convolutional encoder → `(μ, log σ²)` bottleneck → 4-layer transposed-conv decoder
- **Latent dimension**: 128
- **Loss**: β-VAE loss — MSE reconstruction + KL divergence (KL ramped over first 30% of training)
- **Anomaly detection**: pixel-wise `|input − reconstruction|` → blur → threshold → morphological cleaning → largest connected component

### Wasserstein GAN (WGAN)

- **Generator**: 5-layer transposed-conv with BatchNorm + ReLU + Tanh output
- **Critic**: 5-layer conv with LayerNorm + LeakyReLU (no BatchNorm, no Sigmoid)
- **Training**: critic trains 5× per generator step; weight clipping to `[−0.01, 0.01]`; RMSprop optimizer at `lr = 5e-5`
- **Anomaly detection**: subtract generator reconstruction from original → pixel-level difference map as heatmap

### UNet Segmentation

- **Architecture**: Standard encoder-decoder with skip connections; double-conv blocks, MaxPool downsampling, ConvTranspose2d upsampling
- **Feature channels**: `[64, 128, 256, 512]`
- **Input modes**:
  - `baseline` — 1-channel (MRI only)
  - `vae` — 2-channel (MRI + VAE heatmap)
  - `gan` — 2-channel (MRI + GAN heatmap)
- **Loss**: 50/50 BCE + Dice loss
- **Optimizer**: Adam with `ReduceLROnPlateau` scheduler

---

## Requirements

```bash
pip install torch torchvision opencv-python matplotlib numpy pandas scikit-learn tqdm scipy seaborn pillow
```

All notebooks are designed to run on **Kaggle** with GPU acceleration. The data path is pre-configured to `/kaggle/input/datasets/mateuszbuda/lgg-mri-segmentation/`.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Dice Score** | `2·|P∩G| / (|P|+|G|)` — overlap between predicted and ground-truth mask |
| **IoU (Jaccard)** | `|P∩G| / |P∪G|` — stricter overlap measure |
| **W-distance** | `C(real) − C(fake)` — WGAN training health indicator |
| **MSE** | Per-image reconstruction error used as anomaly score in VAE |

---

## Key Design Decisions

- **WGAN over vanilla DCGAN** — eliminates mode collapse and training instability; weight clipping enforces the Lipschitz constraint.
- **β-VAE (β=4)** — stronger KL regularisation pushes the latent space closer to a Gaussian prior, making the model more sensitive to out-of-distribution inputs.
- **Heatmap injection into UNet** — pre-computed anomaly maps from VAE and GAN are stacked as a second input channel, providing a spatial prior for the segmentation network.
- **Morphological post-processing on ROI** — blur → percentile threshold → dilation → largest connected component significantly reduces false positives.
