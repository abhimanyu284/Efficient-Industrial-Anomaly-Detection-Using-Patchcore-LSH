# Efficient Industrial Anomaly Detection Using PatchCore-LSH

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.10-EE4C2C?logo=pytorch" />
  <img src="https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia" />
  <img src="https://img.shields.io/badge/AUROC-96.13%25-brightgreen" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
</p>

<p align="center">
  A <b>training-free, memory-efficient</b> industrial anomaly detection framework that integrates <b>Locality Sensitive Hashing (LSH)</b> with <b>PatchCore</b> — reducing nearest-neighbour search complexity from <b>O(N) → O(log N)</b> while achieving <b>96.13% average AUROC</b> on the MVTec AD1 benchmark.
</p>

---

## Overview

Modern manufacturing lines require automated visual inspection at high throughput. Classical PatchCore — while state-of-the-art for unsupervised anomaly detection — suffers from O(N) linear search complexity, making it impractical for real-time or large-scale industrial deployment.

PatchCore-LSH solves this by replacing exhaustive search with an approximate nearest-neighbour lookup powered by a 100-tree LSH forest, built on top of a compact coreset memory bank that retains only 5% of training patches. No training, no labels, no GPU-intensive optimization — just a single forward pass through a frozen backbone.

---

## Why PatchCore Needs LSH
Standard PatchCore Memory Bank
┌─────────────────────────────────────────┐
│  50,000 – 300,000 patch vectors         │
│  Every test patch compared to ALL of    │
│  them → O(N) per query                  │
│  Hazelnut alone: 391 × 784 = 306,544    │
│  vectors → very slow at scale           │
└─────────────────────────────────────────┘
↓ Problem
Bottleneck for real-time
industrial deployment
PatchCore-LSH Solution
┌─────────────────────────────────────────┐
│  Coreset: 300,000 → 15,000 (5%)         │
│  LSH Index: 100 trees, angular hash     │
│  Query hits ~100 candidates only        │
│  → O(log N) per query                   │
│  Theoretical speedup: 3,125×            │
└─────────────────────────────────────────┘

---

## Full Pipeline
┌─────────────────────────────────────────────────────┐
│                  Input Image 224×224×3               │
└──────────────────────────┬──────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────┐
│          WideResNet-50  (Frozen — no fine-tuning)    │
└────────────────┬────────────────────┬───────────────┘
│                    │
▼                    ▼
Layer 2 Features      Layer 3 Features
28×28×512              14×14×1024
│
▼ Upsample (bilinear)
28×28×1024
│                    │
└────────┬───────────┘
▼
Concatenate → 28×28×1536
L2 Normalise per vector
Sparse Random Projection
1536 → 256 dimensions
│
┌─────────────┴─────────────┐
│                           │
[TRAINING]                   [TESTING]
│                           │
▼                           ▼
Greedy Coreset Sampling       Extract 784 patches
300,000 → 15,000 (5%)         per test image
│                           │
▼                           ▼
Build LSH Index              Query LSH Index
100 random projection        k=9 nearest neighbours
trees (angular hash)         per patch
│                           │
└─────────────┬─────────────┘
▼
Mean distance over k=9 neighbours
28×28 anomaly heatmap (patch scores)
95th percentile → image-level score
│
┌───────────┴───────────┐
▼                       ▼
Score > Threshold         Score ≤ Threshold
DEFECT DETECTED              NORMAL — PASS

---

## Coreset Sampling
FULL MEMORY BANK
████████████████████████████████████████  300,000 patches
████████████████████████████████████████  (many redundant,
████████████████████████████████████████   similar embeddings)
│
▼  Greedy k-center selection
│  Each iteration picks the point
│  FURTHEST from already selected ones
▼
CORESET (5%)
█ · · · █ · · · █ · · · █ · · · █ · · ·   15,000 patches
· · · · · · · · · · · · · · · · · · · ·   maximally diverse,
· · █ · · · · · █ · · · · · █ · · · · ·   covers full normal
feature distribution
Memory: 95% smaller    |    Accuracy loss: < 0.5%

---

## LSH Index Structure
Memory Bank (15,000 coreset vectors)
│
▼  100 independent random projection trees
┌──────────┬──────────┬──────────┬─────────────┐
│  Tree 1  │  Tree 2  │  Tree 3  │  Trees 4-100│
│ Bucket A │ Bucket X │ Bucket P │     ...     │
│ Bucket B │ Bucket Y │ Bucket Q │             │
│ Bucket C │ Bucket Z │ Bucket R │             │
└──────────┴──────────┴──────────┴─────────────┘
│
│  Hash function: h(v) = sign(v · r)
│  r ~ Gaussian random vector
│
Query vector (test patch embedding)
│
▼  Apply same hash functions
Matching buckets → ~100 candidates only
│
▼
k=9 nearest neighbours found
Compute mean distance → anomaly score
Complexity: O(N) ──────────► O(log N)
Exhaustive         LSH approximate

---

## Project Structure
Efficient-Industrial-Anomaly-Detection-Using-Patchcore-LSH/
│
├── Code/
│   └── Patchcore_LSH_file.ipynb      ← Main implementation
│
├── Dataset/                          ← MVTec AD1 (download separately)
│
├── Research Paper/
│   └── Research_Paper_Patchcore.pdf
│
└── Results/
├── Graphs/
│   ├── AUROC + Inference Speed per category.png
│   ├── Accuracy Speed trade off.png
│   ├── Confusion Matrix.png
│   ├── Distribution of AUROC across categories.png
│   ├── High & Low Performing Categories.png
│   ├── MVTEC AD1 Dataset Statistics.png
│   ├── Method comparison.png
│   ├── Multi metric performance.png
│   ├── Patchcore-LSH performance on MVTEC AD 1.png
│   ├── Performance Heatmap.png
│   ├── Performance Radar Chart.png
│   └── ROC Curves.png
│
└── Tables/
├── Dataset Overview.png
├── Final Results on MVTEC AD1.png
├── Individual Category Performance.png
└── Statistics.png

---

## Setup & Installation

```bash
git clone https://github.com/abhimanyu284/Efficient-Industrial-Anomaly-Detection-Using-Patchcore-LSH.git
cd Efficient-Industrial-Anomaly-Detection-Using-Patchcore-LSH
```

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy scikit-learn annoy tqdm pillow matplotlib seaborn
```

Download the MVTec AD dataset from the [official MVTec website](https://www.mvtec.com/company/research/datasets/mvtec-ad) and place it inside the `Dataset/` folder.

Open and run `Code/Patchcore_LSH_file.ipynb` in Jupyter.

---

## Experimental Configuration

| Parameter | Value |
|---|---|
| Python | 3.12 |
| PyTorch | 2.10 |
| CUDA | 12.8 |
| Image Size | 224×224 |
| Batch Size | 32 |
| Feature Dimension | 256 |
| LSH Trees | 100 |
| Coreset Size | 5% |
| Neighbours k | 9 |
| Seed | 42 |

---

## Results

### Performance Across All 13 MVTec AD1 Categories

| Category | AUROC (%) | F1 (%) | Precision (%) | Recall (%) | FPS |
|---|---|---|---|---|---|
| **Tile** | **100.00** | 100.00 | 100.00 | 100.00 | 7.6 |
| Bottle | 99.92 | 99.21 | 98.44 | 100.00 | 10.0 |
| Leather | 99.49 | 98.36 | 98.90 | 97.83 | 6.9 |
| Hazelnut | 99.14 | 97.18 | 95.83 | 98.57 | 6.6 |
| Wood | 98.86 | 98.36 | 96.77 | 100.00 | 7.7 |
| Cable | 97.41 | 93.99 | 94.51 | 93.48 | 8.5 |
| Toothbrush | 97.22 | 95.24 | 90.91 | 100.00 | 10.1 |
| Zipper | 97.03 | 95.87 | 94.31 | 97.48 | 8.3 |
| Metal Nut | 96.29 | 95.34 | 92.00 | 98.92 | 8.6 |
| Transistor | 93.50 | 87.18 | 89.47 | 85.00 | 8.6 |
| Carpet | 90.77 | 91.10 | 85.29 | 97.75 | 6.7 |
| Capsule | 90.31 | 94.69 | 91.45 | 98.17 | 8.6 |
| Pill | 89.74 | 94.37 | 93.71 | 95.04 | 7.4 |
| **Average** | **96.13** | **95.44** | **93.86** | **96.81** | **8.1** |
AUROC by Category
Average: 96.13%
100% ┤ ██                                     ─ ─ ─ ─ ─ ─ ─ ─
99% ┤ ██ ██
98% ┤ ██ ██ ██
97% ┤ ██ ██ ██ ██ ██ ██
96% ┤ ██ ██ ██ ██ ██ ██ ██
95% ┤ ██ ██ ██ ██ ██ ██ ██
94% ┤ ██ ██ ██ ██ ██ ██ ██ ██
93% ┤ ██ ██ ██ ██ ██ ██ ██ ██ ██
92% ┤ ██ ██ ██ ██ ██ ██ ██ ██ ██
91% ┤ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██
90% ┤ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██
89% ┤ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██
└──────────────────────────────────────────
Ti Bo Le Ha Wo Ca To Zi MN Tr Ca Ca Pi
le tt at ze od bl ot pp et an rp ps ll
le he ln    le h  er  Nu si et ul e
r  ut       er  t  st  e  e

8 out of 13 categories (61.5%) achieved AUROC above 95%. 12 out of 13 (92.3%) achieved above 90%.

### Confusion Matrix (Average Across All Categories)
              Predicted Normal   Predicted Defect
            ┌──────────────────┬──────────────────┐
Actual Normal │      96.2%       │      3.8%        │
│  (True Negative) │ (False Positive) │
├──────────────────┼──────────────────┤
Actual Defect │      3.2%        │      96.8%       │
│ (False Negative) │  (True Positive) │
└──────────────────┴──────────────────┘
Specificity: 96.2%  |  Recall: 96.8%  |  False Positive Rate: 3.8%

---

## Comparison with Other Methods

| Method | AUROC (%) | F1 (%) | Precision (%) | Recall (%) | FPS |
|---|---|---|---|---|---|
| Autoencoder (AED) | 60.20 | 71.49 | 83.20 | 58.21 | 25.30 |
| Diffusion (SD + LoRA) | 63.49 | 89.42 | 80.87 | 100.00 | 1.17 |
| **PatchCore-LSH (Ours)** | **96.13** | **95.44** | **93.86** | **96.81** | **8.10** |
AUROC Comparison
─────────────────────────────────────────
PatchCore-LSH  ████████████████████  96.1%
Diffusion      █████████             63.5%
Autoencoder    ████████              60.2%
─────────────────────────────────────────
+35.9% over Autoencoder
+32.6% over Diffusion
6.9× faster than Diffusion

---

## Future Work

**Adaptive Patch Sizing** — Use smaller patches for fine-grained textures and larger patches for structural categories to better tune the accuracy-speed tradeoff.

**Higher Input Resolution** — Experiment with 448×448 input for small object classes like Pill and Capsule where 224×224 loses critical spatial detail.

**Ensemble LSH Indices** — Average predictions from 3–5 independent LSH indices for an estimated 1–2% AUROC improvement.

**Edge Deployment** — Apply knowledge distillation to compress WideResNet50 into MobileNet for Raspberry Pi and smartphone deployment.

**Automatic Threshold Selection** — Replace label-dependent threshold calibration with an unsupervised threshold estimation method.

**Cross-Category Learning** — Share memory banks across similar categories to reduce total storage requirements.

---

## Authors

**Krishka Kate** and **Abhimanyu Nema**

---

## References

- Bergmann et al., "MVTec AD," CVPR 2019 / IJCV 2021
- Roth et al., "Towards total recall in industrial anomaly detection," CVPR 2022
- Roth et al., "PatchCore," arXiv 2021
- Gionis et al., "Similarity search in high dimensions via hashing," VLDB 1999
- Andoni & Indyk, "Near-optimal hashing algorithms for ANN," CACM 2008
- Zagoruyko & Komodakis, "Wide residual networks," BMVC 2016

---

<p align="center">If you find this work useful, please ⭐ star the repository!</p>
