# Efficient Industrial Anomaly Detection Using PatchCore-LSH

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.10-EE4C2C?logo=pytorch" />
  <img src="https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia" />
  <img src="https://img.shields.io/badge/AUROC-96.13%25-brightgreen" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
</p>

> A **training-free, memory-efficient** industrial anomaly detection framework integrating **Locality Sensitive Hashing (LSH)** with **PatchCore** — reducing nearest-neighbour search from **O(N) → O(log N)** while achieving **96.13% average AUROC** on MVTec AD1.

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Key Contributions](#-key-contributions)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Experimental Configuration](#-experimental-configuration)
- [Results](#-results)
- [Comparison with Other Methods](#-comparison-with-other-methods)
- [Future Work](#-future-work)
- [Authors](#-authors)
- [References](#-references)

---

## 🔍 Overview

Modern manufacturing lines require automated visual inspection at high throughput. Classical PatchCore — while state-of-the-art for unsupervised anomaly detection — suffers from **O(N) linear search complexity**, making it impractical for real-time or large-scale industrial deployment.

**PatchCore-LSH** addresses this by:
1. Extracting multi-scale patch-level features from a **frozen WideResNet50** backbone (no fine-tuning needed).
2. Building a compact **coreset memory bank** (5% of patches) via greedy k-center sampling.
3. Indexing the memory bank with a **100-tree LSH forest** for approximate nearest-neighbour queries in O(log N).

The result is a framework that is **training-free**, **95% more memory-efficient**, and ready for **edge deployment**.

---

## 🏆 Key Contributions

| Contribution | Detail |
|---|---|
| **LSH Integration** | Approximate nearest-neighbour search via 100-tree LSH forest; O(N) → O(log N) |
| **Coreset Sampling** | Greedy k-center sampling retains only 5% of patches — 95% memory reduction with <0.5% accuracy loss |
| **Multi-Scale Features** | Layer 2 (28×28×512) + Layer 3 upsampled → concatenated 1536-dim embeddings |
| **Dimensionality Reduction** | Sparse random projection: 1536 → 256 dims (Johnson-Lindenstrauss) |
| **Comprehensive Evaluation** | 13 MVTec AD1 categories; avg AUROC 96.13%, Tile achieving perfect 100% |
| **Practical Speed** | 8.1 FPS average (123ms/image); suitable for production lines at 5–10 units/sec |

---

## ⚙️ How It Works

### Pipeline
