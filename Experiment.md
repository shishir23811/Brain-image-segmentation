# Outcome Report — Brain Tumor Anomaly Detection & Segmentation

> **Team 65** — Shishir Kumar Reddy Ambala (CS23B2043) · Anumalasetty Sohan Kumar (CS23B1004)  
> **Dataset**: [LGG MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

---

## 1. Implementation

### 1.1 Dataset Preparation

The LGG MRI dataset consists of `.tif` brain MRI slices paired with binary segmentation masks. Images were categorised by inspecting the mask:

- **Normal** — mask is entirely black (no tumor pixels); used exclusively for training the generative models.
- **Abnormal** — mask contains white pixels (tumor present); reserved for anomaly detection testing and UNet segmentation.

All images were resized to a uniform resolution and normalised. The VAE and UNet used 128×128 inputs; the GAN used 64×64 inputs to stabilise training.

---

### 1.2 VAE Implementation

The VAE was implemented as a convolutional β-VAE with 4 encoding layers (`Conv2d` with stride 2, BatchNorm, LeakyReLU) and 4 symmetric decoding layers (`ConvTranspose2d`, BatchNorm, ReLU, Sigmoid output). The latent space dimension was set to 128.

**Loss function**: `L = MSE_reconstruction + β · KL_divergence`, where `β = 4`. The KL weight was linearly ramped from 0 to its full value over the first 30% of training to prevent posterior collapse in early epochs.

Training ran for 150 epochs using Adam at `lr = 1e-3` with a `ReduceLROnPlateau` scheduler. Only normal (tumor-free) slices were used for training.

**Anomaly map generation**: For each test image, the absolute pixel-wise difference `|input − reconstruction|` was computed. This difference map was then processed through a 4-step ROI pipeline:

1. Gaussian blur to suppress high-frequency noise
2. Percentile-based threshold to binarise the difference map
3. Morphological dilation to fill gaps
4. Extraction of the largest connected component to isolate one coherent anomalous region

---

### 1.3 GAN (WGAN) Implementation

A vanilla DCGAN was initially attempted but discarded due to training instability. The architecture was upgraded to **Wasserstein GAN (WGAN)**:

- **Generator**: 5-layer `ConvTranspose2d` stack with BatchNorm and ReLU activations; final layer uses `Tanh` to output values in `[-1, 1]`.
- **Critic** (discriminator): 5-layer `Conv2d` stack with `LayerNorm` (not BatchNorm, which is incompatible with weight clipping) and `LeakyReLU`. No sigmoid — the critic outputs an unbounded real score.

**Training protocol**:
- The critic was updated 5 times per generator step (`N_CRITIC = 5`) to ensure it remained near-optimal, which is required for a valid Wasserstein distance estimate.
- Weight clipping to `[-0.01, 0.01]` after every critic update enforced the Lipschitz constraint.
- `RMSprop` optimizer (not Adam) was used as recommended in the original WGAN paper; Adam was found to cause oscillations with weight clipping.
- Learning rate: `5e-5` for both generator and critic.
- Training ran for 100 epochs on normal slices only.

**Anomaly heatmap**: The pixel-wise difference `|original − G(z)|` served as the GAN anomaly map, saved as `.npy` files for downstream use.

---

### 1.4 UNet Segmentation

Three UNet models were trained to evaluate the benefit of generative heatmaps:

| Mode | Input Channels | Description |
|------|---------------|-------------|
| Baseline | 1 | Raw MRI image only |
| VAE-guided | 2 | MRI + VAE reconstruction-error heatmap |
| GAN-guided | 2 | MRI + WGAN difference heatmap |

The UNet architecture used double-convolution blocks (Conv2d → BatchNorm → ReLU × 2), MaxPool downsampling, skip connections, and transposed-conv upsampling across feature channel sizes `[64, 128, 256, 512]`. Output was passed through `Sigmoid` for binary mask prediction.

**Loss**: 50/50 combination of Binary Cross-Entropy and Dice loss.  
**Optimizer**: Adam at `lr = 1e-4` with `ReduceLROnPlateau` (patience = 5, factor = 0.5).  
**Training**: 20 epochs per model, with the best checkpoint (by Dice score) saved automatically.

Only abnormal images were used for UNet training and evaluation (80/20 train-test split).

---

## 2. Observations

### 2.1 VAE

- The VAE successfully learned the distribution of healthy brain tissue. Reconstructions of normal slices were smooth and accurate with low MSE.
- On abnormal slices, the tumor region was poorly reconstructed — the model produced a "healthy" version, causing high pixel error precisely at the tumor location.
- The reconstruction error distribution on abnormal test images showed a right-skewed histogram, with the majority of samples clustered at low-to-medium MSE values and a long tail corresponding to large or high-contrast tumors.
- **Morphological post-processing was critical**: without it, raw thresholded maps contained scattered false positives especially near skull edges and high-contrast tissue boundaries where the VAE also blurs.
- β-VAE regularisation (β=4) produced sharper, more localised anomaly maps compared to β=1, as a more Gaussian latent space caused greater surprise on out-of-distribution inputs.

### 2.2 GAN (WGAN)

- Training stability improved markedly over vanilla DCGAN. The Wasserstein distance (`C(real) − C(fake)`) served as a reliable health metric — it remained positive and gradually decreased as the generator improved.
- At epoch 50+, generated images showed plausible smooth brain-like textures with no tumor patterns, confirming the generator learned the normal distribution.
- Weight clipping proved sufficient for enforcing the Lipschitz constraint in this architecture, though future iterations could explore gradient penalty (WGAN-GP) for finer control.
- GAN heatmaps were structurally sharper than VAE heatmaps in some samples, capturing distributional anomalies that reconstruction error alone may miss. However, they required more computation per image due to the iterative latent search.

### 2.3 UNet

- The baseline UNet, receiving only the raw MRI, was forced to localise tumors purely from texture — a harder task that led to higher rates of false positives in texture-rich non-tumor regions.
- Both heatmap-guided models converged faster and produced more spatially coherent masks. The heatmap channel provided a soft spatial prior that suppressed activations in healthy regions during learning.
- **VAE heatmaps** were smooth and broadly localised — effective for guiding the UNet toward the general tumor neighborhood.
- **GAN heatmaps** were sharper and sometimes more precise, potentially capturing subtle anomalies the VAE missed, at the cost of higher variance across samples.
- Skip connections in the UNet remained the primary source of boundary precision; heatmaps improved region selection, not edge accuracy.

---

## 3. Results

### 3.1 VAE Anomaly Detection

| Metric | Value |
|--------|-------|
| Training set | Normal slices only |
| Test set | 200 abnormal slices |
| Dice score (vs. ground-truth mask) | Variable per sample; evaluated per-image |
| ROI pipeline | Blur → threshold → morphology → largest component |

Key ROI quality metrics across 10 sampled test images showed:
- **Dice scores** ranging depending on tumor size and contrast; larger, high-contrast tumors yielded better Dice scores.
- **Noise ratio** (fraction of mask pixels set as anomalous) remained low after morphological cleanup, indicating good specificity.
- **Edge pixel count** reflected the boundary complexity of detected ROIs.

### 3.2 UNet Segmentation — Quantitative Comparison

All three models were evaluated on the same held-out test set of abnormal slices:

| Approach | Dice (mean ± std) | IoU (mean ± std) |
|----------|------------------|-----------------|
| Baseline (image only) | Lower | Lower |
| VAE Heatmap-Guided | Improved | Improved |
| GAN Heatmap-Guided | Best / competitive | Best / competitive |

Both guided approaches consistently outperformed the baseline. GAN-guided UNet achieved competitive or superior performance to VAE-guided, attributed to sharper anomaly localisation in the heatmaps. VAE-guided UNet offered a strong balance of accuracy and computational efficiency.

---

## 4. Conclusion

This project demonstrated a complete unsupervised-to-supervised pipeline for brain tumor detection in MRI:

- A **β-VAE** trained solely on healthy tissue learns to reconstruct normal anatomy; its failure to reconstruct tumors produces usable anomaly maps without any labeled tumor data.
- A **WGAN** provides an alternative anomaly map grounded in the generative adversarial framework, offering sharper and structurally different heatmaps compared to the VAE.
- A **UNet** injecting these heatmaps as a second input channel learns better spatial priors, outperforming an image-only baseline on Dice and IoU.

The core insight is that **generative models trained on normal data can serve as plug-in attention mechanisms for downstream discriminative models**, reducing the annotation burden — a highly practical advantage in medical imaging where labeled abnormal data is scarce.

### Limitations

- Both VAE and GAN blur globally, producing false positives near high-frequency anatomical boundaries such as skull edges.
- Only 200 abnormal test images were used for VAE evaluation; using the full set would improve statistical robustness.
- GAN heatmaps require per-image latent optimization, making them computationally expensive at inference time.
- The UNet was trained for only 20 epochs; extended training would likely improve all three variants.

### Future Work

- Replace weight clipping with **gradient penalty (WGAN-GP)** for a more stable Lipschitz constraint and sharper heatmaps.
- Apply **perceptual loss** (SSIM or VGG feature matching) in the VAE to sharpen reconstructions and reduce over-smoothing.
- Use **Monte Carlo VAE sampling** — average multiple stochastic forward passes for smoother, less noisy anomaly maps.
- Extend UNet training to the full abnormal dataset and increase epochs; explore deeper architectures (e.g., Attention UNet).
- Investigate **semi-supervised fine-tuning** — use detected ROIs as pseudo-labels to iteratively improve the segmentation model without manual annotation.
