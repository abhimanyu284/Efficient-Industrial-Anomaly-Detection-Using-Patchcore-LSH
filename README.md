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
