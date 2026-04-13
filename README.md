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
