# Efficient Industrial Anomaly Detection Using PatchCore-LSH

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.10-EE4C2C?style=flat-square&logo=pytorch" />
  <img src="https://img.shields.io/badge/CUDA-12.8-76B900?style=flat-square&logo=nvidia" />
  <img src="https://img.shields.io/badge/Dataset-MVTec%20AD1-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/AUROC-96.13%25-brightgreen?style=flat-square" />
</p>

<p align="center">
  <b>A training-free, memory-efficient industrial anomaly detection framework combining PatchCore with Locality Sensitive Hashing (LSH)</b><br/>
  <i>Validated on MVTec AD1 · Average AUROC: 96.13% · Inference Speed: 8.1 FPS</i>
</p>

---

## 📌 Table of Contents

- [Overview](#overview)
- [Key Contributions](#key-contributions)
- [Folder Structure](#folder-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Graphs & Visualizations](#graphs--visualizations)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [Authors](#authors)
- [References](#references)

---

## Overview

Automated quality control in manufacturing demands fast, accurate, and scalable defect detection systems. This project presents **PatchCore-LSH**, a novel framework that integrates **Locality Sensitive Hashing (LSH)** with the **PatchCore** architecture to address the scalability bottleneck inherent in exhaustive nearest-neighbor search.

The framework:
- Extracts **patch-level features** from a frozen **WideResNet50** backbone (no fine-tuning required)
- Builds a **memory bank** using greedy coreset sampling (retaining only 5% of patches)
- Indexes the memory bank using **LSH with 100 random projection trees**
- Performs **approximate nearest-neighbor search** in O(log N) instead of O(N)

This results in a **training-free**, **memory-efficient**, and **scalable** system achieving an average **AUROC of 96.13%** across 13 MVTec AD1 categories.

---

## Key Contributions

| Contribution | Detail |
|---|---|
| **LSH Integration** | Reduces query complexity from O(N) to O(log N) while maintaining detection accuracy |
| **Coreset Sampling** | 95% memory reduction by retaining only 5% of training patches via greedy k-center selection |
| **Multi-Scale Features** | Combines Layer 2 (28×28, 512ch) and Layer 3 (14×14, 1024ch) from WideResNet50 |
| **Dimensionality Reduction** | Sparse random projection from 1536 → 256 dims using Johnson-Lindenstrauss |
| **Inference Speed** | Average 8.1 FPS (~123ms/image), suitable for industrial lines at 5–10 units/sec |
| **Zero Training** | Single forward pass through frozen backbone — no GPU-intensive optimization needed |

---

## Folder Structure
Efficient-Industrial-Anomaly-Detection-Using-Patchcore-LSH/
│
├── Code/
│   └── Patchcore_LSH_file.ipynb        # Main implementation notebook
│
├── Dataset/
│   └── (MVTec AD1 — download separately, see Dataset section below)
│
├── Research Paper/
│   └── Research_Paper_Patchcore.pdf    # Full paper
│
└── Results/
├── Graphs/
│   ├── AUROC + Inference Speed per category.png
│   ├── Accuracy Speed trade off.png
│   ├── Confusion Matrix.png
│   ├── Distributiuon of AUROC across categories.png
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

## Dataset

We use the **MVTec Anomaly Detection Dataset (MVTec AD1)**, a widely adopted benchmark for unsupervised industrial anomaly detection containing **15 categories** of manufactured goods and textures.

> 📥 **Download MVTec AD1:** [https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad)

After downloading, place the dataset inside the `Dataset/` folder. The expected structure is:
Dataset/
├── bottle/
│   ├── train/good/
│   └── test/good/, test/broken_large/, ...
├── cable/
├── capsule/
├── carpet/
├── hazelnut/
├── leather/
├── metal_nut/
├── pill/
├── tile/
├── toothbrush/
├── transistor/
├── wood/
└── zipper/

### Dataset Statistics (13 categories evaluated)

| Category | Train (Good) | Test Good | Test Defect | Total Test |
|---|---|---|---|---|
| Bottle | 209 | 20 | 63 | 83 |
| Cable | 224 | 58 | 92 | 150 |
| Capsule | 219 | 23 | 109 | 132 |
| Carpet | 280 | 28 | 89 | 117 |
| Hazelnut | 391 | 40 | 70 | 110 |
| Leather | 245 | 32 | 92 | 124 |
| Metal Nut | 220 | 22 | 93 | 115 |
| Pill | 267 | 26 | 141 | 167 |
| Tile | 230 | 33 | 84 | 117 |
| Toothbrush | 60 | 12 | 30 | 42 |
| Transistor | 213 | 60 | 40 | 100 |
| Wood | 247 | 19 | 60 | 79 |
| Zipper | 240 | 32 | 119 | 151 |

> Training sets contain **only normal (defect-free) images**. Test sets contain both normal and defective samples with various defect types — scratches, dents, cuts, holes, structural anomalies — varying per category.

---

## Methodology

### Full Pipeline
Input Image (224×224×3)
│
▼
[ WideResNet50 — Frozen Backbone, pretrained on ImageNet ]
│
┌────┴────┐
▼         ▼
Layer 2    Layer 3
(28×28     (14×14
512ch)    1024ch)
│         │
│   Upsample (14×14 → 28×28 via bilinear interpolation)
│         │
└────┬────┘
▼
Concatenate along channel dim → 28×28×1536
│
L2 Normalize (unit hypersphere)
│
Sparse Random Projection → 28×28×256
│
┌────┴──────────────────────┐
│                           │
[TRAINING PHASE]          [TESTING PHASE]
│                           │
Coreset Sampling           LSH Query
(greedy k-center,          (k=9 approximate
retain 5%)                 neighbours)
│                           │
LSH Index Build            Mean Distance
(100 random                over 9 neighbours
projection trees)             │
28×28 Patch Heatmap
│
95th Percentile → Image Score
│
Score > Threshold?
/              
YES               NO
DEFECT           NORMAL ✓

### 1. Pre-Trained Backbone — WideResNet50

- Pre-trained on **ImageNet**, kept **completely frozen** throughout — no weight updates ever occur
- Wider residual blocks provide richer, more diverse feature representations than standard ResNet50
- Each of 4 main stages comprises residual blocks with **ReLU** and **Batch Normalization**
- Input images resized to **224×224** via bilinear interpolation
- Normalized with ImageNet statistics: `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`

### 2. Multi-Scale Feature Extraction

| Layer | Spatial Size | Channels | Receptive Field | Best For |
|---|---|---|---|---|
| Layer 2 | 28×28 | 512 | ~32×32 px | Small surface defects (scratches, dents) |
| Layer 3 | 14×14 | 1024 | ~64×64 px | Large structural anomalies, global context |

- Layer 3 upsampled to 28×28 via bilinear interpolation (`align_corners=False`)
- Both layers **concatenated** along channel dimension → **1536-dim** per spatial position
- All vectors **L2-normalized** → unit hypersphere (angular/cosine similarity metric)
- Total: **784 patch vectors per image** (28×28 grid)

### 3. Coreset Sampling — Memory Optimization

Reduces up to 300,000 patch vectors (e.g., Hazelnut: 391 × 784 = 306,544) down to **5%** using **greedy k-center sampling**:

| Stage | Action |
|---|---|
| **Stage 1 — Random Subsampling** | Sample 50,000 vectors from full bank → reduces complexity O(N²) → O(N·M) |
| **Stage 2 — Initialization** | Randomly select first coreset point as starting seed |
| **Stage 3 — Iterative Selection** | Each iteration: pick vector with **maximum minimum distance** to already-selected points |
| **Stage 4 — Termination** | Stop when coreset reaches 5% of original size |

> Result: **95% memory reduction** · Less than **0.5% accuracy loss** · Greedy algorithm provides **2-approximation guarantee** for the k-center problem

### 4. Dimensionality Reduction

- **Sparse Random Projection**: 1536 → **256 dimensions**
- Based on the **Johnson-Lindenstrauss lemma** — approximately preserves all pairwise distances
- Runtime O(nd) vs O(n²d) for PCA — much faster and entirely **unsupervised** (no training data needed)
- Vectors are re-L2-normalized after projection to maintain unit norm

### 5. LSH Index Construction

- Uses **angular (cosine) distance** metric
- **100 independent random projection trees** — each tree adds recall at cost of build time and memory
- Hash function: `h(v) = sign(v · r)` where `r ~ Gaussian(0, I)`
- Vectors on the same side of a random hyperplane → same hash bucket
- Probability of finding true nearest neighbor **> 95%** with 100 trees
- Query complexity: **O(N) → O(log N)** — theoretical speedup of ~3,125× for 300,000 patch banks

### 6. Anomaly Scoring

- For each test image: extract 784 patch vectors (28×28 grid)
- Each patch queries LSH index for **k=9 nearest neighbors**
- Patch anomaly score = **mean distance** to 9 neighbors (more robust than minimum — resists noise/outliers)
- Image-level anomaly score = **95th percentile** of all 784 patch scores (balances sensitivity vs specificity)
- **28×28 heatmap** localizes defects spatially — operator can identify both *whether* and *where* a defect exists
- Score compared against calibrated threshold → **Defect** or **Normal**

---

## Experimental Setup

| Parameter | Value |
|---|---|
| CPU | 12 vCPUs |
| RAM | 78 GB |
| OS | Linux / Windows / Mac OS |
| Python | 3.12 |
| PyTorch | 2.10 |
| CUDA | 12.8 |
| Image Size | 224×224 |
| Batch Size | 32 |
| Feature Dimension | 256 (post sparse random projection) |
| LSH Trees | 100 |
| Coreset Size | 5% of training patches |
| Nearest Neighbors (k) | 9 |
| Random Seed | 42 |

---

## Results

### Overall Performance — MVTec AD1 (13 Categories)

| Category | AUROC (%) | F1 (%) | Precision (%) | Recall (%) | FPS |
|---|---|---|---|---|---|
| **Tile** | **100.00** | **100.00** | **100.00** | **100.00** | 7.6 |
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
| **Average** | **96.13** | **95.44** | **93.86** | **96.81** | **8.10** |

> ✅ 8/13 categories (61.5%) achieved AUROC above 95%  
> ✅ 12/13 categories (92.3%) achieved AUROC above 90%  
> ✅ Tile achieved perfect 100% AUROC  
> ✅ Standard deviation of AUROC scores: 3.8% — consistent performance across all categories  

### Confusion Matrix Summary (Averaged across all 13 categories)
               Predicted Normal    Predicted Defect
Actual Normal  |      96.2%        |       3.8%       |
Actual Defect  |       3.2%        |      96.8%       |

- **Specificity: 96.2%** — Only 3.8% false positive rate (low false alarms, no unnecessary production halts)
- **Recall: 96.8%** — Catches 96.8% of all actual defects (false negative rate of only 3.2%)

### Comparison with Alternative Methods

| Method | AUROC (%) | F1 (%) | Precision (%) | Recall (%) | FPS |
|---|---|---|---|---|---|
| Autoencoder (AED) | 60.20 | 71.49 | 83.20 | 58.21 | 25.30 |
| Diffusion Model (SD + LoRA) | 63.49 | 89.42 | 80.87 | 100.00 | 1.17 |
| **PatchCore-LSH (Ours)** | **96.13** | **95.44** | **93.86** | **96.81** | **8.10** |

> **+32.64% higher AUROC** than diffusion models while running **6.9× faster**  
> **+35.93% higher AUROC** than autoencoder baseline

### Inference Speed Breakdown

| Category | FPS | Approx. Memory Bank Vectors | Notes |
|---|---|---|---|
| Toothbrush | 10.1 (fastest) | ~47,040 (60 × 784) | Smallest training set |
| Hazelnut | 6.6 (slowest) | ~306,544 (391 × 784) | Largest training set |
| **Average** | **8.1** | — | **~123ms per image** |

> LSH search contributes only **15–25ms** per image. WideResNet50 backbone accounts for **80–100ms** — future speed improvements should target backbone compression, not LSH tuning.

### High Performing Categories (AUROC ≥ 97%)

| Category | AUROC | Why It Performs Well |
|---|---|---|
| Tile | 100.00% | Structured, repetitive texture — crack features are highly distinct from normal patterns |
| Bottle | 99.92% | Clear structural anomalies (scratches, breakage) stand out strongly in feature space |
| Leather | 99.49% | Fine-grained surface captured well by Layer 2, structural context by Layer 3 |
| Hazelnut | 99.14% | Surface cracks on shells produce distinctive patch-level deviations |
| Wood | 98.86% | Statistical regularities in grain captured even in non-repetitive natural texture |

### Low Performing Categories (AUROC < 95%)

| Category | AUROC | Root Cause |
|---|---|---|
| Pill | 89.74% | Small object (~30–50px), subtle colour/shape defects, limited training diversity |
| Capsule | 90.31% | Similar small-object challenges, defects near edges hard to capture at 28×28 resolution |
| Carpet | 90.77% | Random/stochastic texture — normal variation can resemble anomalies |
| Transistor | 93.50% | Multi-component structure — global misalignment not captured at patch level |

---

## Graphs & Visualizations

All graphs are in [`Results/Graphs/`](Results/Graphs/) and all result tables are in [`Results/Tables/`](Results/Tables/).

| File | Description |
|---|---|
| `Patchcore-LSH performance on MVTEC AD 1.png` | Bar chart of AUROC across all 13 categories with 96.13% average line |
| `Performance Heatmap.png` | Colour heatmap of AUROC, F1, Precision, Recall across all categories |
| `Multi metric performance.png` | Grouped bar chart comparing AUROC, F1, Recall per category |
| `High & Low Performing Categories.png` | Separate comparison charts for top (≥97%) and bottom (<95%) performers |
| `Accuracy Speed trade off.png` | Bubble chart — AUROC vs FPS, bubble size proportional to AUROC |
| `AUROC + Inference Speed per category.png` | Per-category FPS with AUROC overlay |
| `Confusion Matrix.png` | Normalized average confusion matrix across all 13 categories |
| `ROC Curves.png` | ROC curves per category |
| `Method comparison.png` | Bar comparison: PatchCore-LSH vs AED vs Diffusion model |
| `Performance Radar Chart.png` | Radar/spider chart across AUROC, F1, Precision, Recall, FPS |
| `Distributiuon of AUROC across categories.png` | Distribution plot showing spread of AUROC scores |
| `MVTEC AD1 Dataset Statistics.png` | Visual overview of training/test splits per category |
| `Final Results on MVTEC AD1.png` | Complete results table rendered as image |
| `Individual Category Performance.png` | Per-category breakdown table |
| `Dataset Overview.png` | Dataset summary statistics table |

---

## Conclusion

This paper presented **PatchCore-LSH**, a novel industrial anomaly detection framework that integrates Locality Sensitive Hashing with the PatchCore architecture to overcome its fundamental O(N) scalability bottleneck.

**Summary of achievements:**

| Metric | Value |
|---|---|
| Average AUROC | 96.13% (exceeds 96% target) |
| Best Category | Tile — 100.00% AUROC |
| Memory Reduction | 95% (5% coreset retention) |
| Search Complexity | O(N) → O(log N) |
| Average Inference Speed | 8.1 FPS (~123ms/image) |
| vs. Diffusion Model | +32.64% AUROC, 6.9× faster |
| vs. Autoencoder | +35.93% AUROC |
| Training Required | None — completely training-free |

The framework is well-suited for real-world industrial deployment where defect samples are rare, labeling is expensive, and real-time processing is required. Its memory efficiency makes it viable even on resource-constrained edge devices.

---

## Future Work

| Direction | Description |
|---|---|
| **Adaptive Patch Sizing** | Smaller patches for fine-grained textures, larger for coarse/structural objects |
| **Higher Input Resolution** | 448×448 input for small object categories (Pill, Capsule) to improve patch coverage |
| **Ensemble LSH Indices** | Average across 3–5 independent LSH indices for ~1–2% AUROC improvement |
| **Edge Device Deployment** | Knowledge distillation: WideResNet50 → MobileNet for Raspberry Pi / Smartphone use |
| **Auto Threshold Selection** | Remove dependency on labeled validation data via unsupervised threshold calibration |
| **Cross-Category Memory Sharing** | Share memory banks across related categories to reduce total storage requirements |

---

## Authors

**Krishka Kate** and **Abhimanyu Nema**  
*NMIMS Indore*

> 🔗 **GitHub:** [https://github.com/abhimanyu284/Efficient-Industrial-Anomaly-Detection-Using-Patchcore-LSH](https://github.com/abhimanyu284/Efficient-Industrial-Anomaly-Detection-Using-Patchcore-LSH)  
> 📄 **Paper:** [`Research Paper/Research_Paper_Patchcore.pdf`](Research%20Paper/Research_Paper_Patchcore.pdf)    
> 📦 **Dataset:** [MVTec AD1 — Official Download](https://www.mvtec.com/company/research/datasets/mvtec-ad)

---

## References

1. P. Bergmann et al., "MVTec AD – A comprehensive real-world dataset for unsupervised anomaly detection," CVPR, 2019.
2. P. Bergmann et al., "The MVTec anomaly detection dataset," Int. J. Comput. Vis., 2021.
3. K. Roth et al., "Towards total recall in industrial anomaly detection," CVPR, 2022.
4. K. Roth et al., "PatchCore: Anomaly detection with local patch features," arXiv:2106.08265, 2021.
5. A. Gionis, P. Indyk, R. Motwani, "Similarity search in high dimensions via hashing," VLDB, 1999.
6. A. Andoni and P. Indyk, "Near-optimal hashing algorithms for approximate nearest neighbor," Commun. ACM, 2008.
7. M. Datar et al., "Locality-sensitive hashing scheme based on p-stable distributions," SCG, 2004.
8. T. Defard et al., "PaDiM: A patch distribution modeling framework," ICPR, 2021.
9. D. Gong et al., "Memorizing normality to detect anomaly," ICCV, 2019.
10. T. Schlegl et al., "f-AnoGAN: Fast unsupervised anomaly detection with GANs," Med. Image Anal., 2019.
11. S. Zagoruyko and N. Komodakis, "Wide residual networks," BMVC, 2016.
12. J. Deng et al., "ImageNet: A large-scale hierarchical image database," CVPR, 2009.
13. Y. Liu et al., "Deep learning for industrial anomaly detection: A survey," IEEE Trans. Ind. Informat., 2022.
14. G. Pang et al., "Deep learning for anomaly detection: A review," ACM Comput. Surv., 2021.
