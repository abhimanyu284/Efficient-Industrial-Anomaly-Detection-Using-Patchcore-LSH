# Efficient Industrial Anomaly Detection Using PatchCore-LSH

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.10-EE4C2C?style=flat-square&logo=pytorch" />
  <img src="https://img.shields.io/badge/CUDA-12.8-76B900?style=flat-square&logo=nvidia" />
  <img src="https://img.shields.io/badge/Dataset-MVTec%20AD-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/AUROC-96.13%25-brightgreen?style=flat-square" />
</p>

<p align="center">
  <b>Training-free, memory-efficient industrial anomaly detection using PatchCore + LSH</b><br/>
  <i>Validated on MVTec AD · AUROC: 96.13% · Speed: 8.1 FPS</i>
</p>

---

## 📌 Overview
PatchCore-LSH is a scalable anomaly detection system that improves PatchCore using Locality Sensitive Hashing (LSH).

- Converts O(N) search → O(log N)
- No training required
- Suitable for real-time industrial inspection

---

## 🔥 Key Contributions
- LSH reduces complexity O(N) → O(log N)
- 95% memory reduction via coreset sampling
- Multi-scale feature extraction (Layer 2 + Layer 3)
- Training-free pipeline
- Achieves 96.13% AUROC

---

## 📂 Folder Structure

Efficient-Industrial-Anomaly-Detection-Using-Patchcore-LSH/
│
├── Code/
│   └── Patchcore_LSH_file.ipynb
│
├── Dataset/
│   └── (MVTec AD - download separately)
│
├── Research_Paper/
│   └── Research_Paper_Patchcore.pdf
│
├── Results/
│   ├── Graphs/
│   │   ├── AUROC + Inference Speed per category.png
│   │   ├── Accuracy Speed trade off.png
│   │   ├── Confusion Matrix.png
│   │   ├── Distribution of AUROC across categories.png
│   │   ├── High & Low Performing Categories.png
│   │   ├── MVTEC AD Dataset Statistics.png
│   │   ├── Method comparison.png
│   │   ├── Multi metric performance.png
│   │   ├── Patchcore-LSH performance.png
│   │   ├── Performance Heatmap.png
│   │   ├── Performance Radar Chart.png
│   │   └── ROC Curves.png
│
│   └── Tables/
│       ├── Dataset Overview.png
│       ├── Final Results.png
│       ├── Individual Category Performance.png
│       └── Statistics.png

---

## 📦 Dataset

Download: https://www.mvtec.com/company/research/datasets/mvtec-ad

Expected structure:

Dataset/
├── bottle/
│   ├── train/good/
│   └── test/good/, test/broken_large/
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

---

## ⚙️ Methodology

Pipeline:

Input Image (224×224×3)
→ WideResNet50 (Frozen Backbone)
→ Layer 2 (28×28×512) + Layer 3 (14×14×1024)
→ Upsample Layer 3
→ Concatenate → 1536 dim
→ L2 Normalize
→ Random Projection → 256 dim
→ TRAIN: Coreset Sampling (5%)
→ TEST: LSH Query (k=9)
→ Patch Heatmap
→ 95th Percentile Score
→ Threshold → Defect / Normal

---

## 🧠 Core Components

1. Backbone  
WideResNet50 pretrained on ImageNet (frozen)

2. Feature Extraction  
Layer 2 → small defects  
Layer 3 → large defects  

3. Coreset Sampling  
Retains only 5% memory using greedy k-center  

4. Dimensionality Reduction  
1536 → 256 using random projection  

5. LSH Index  
100 trees for approximate nearest neighbor search  

6. Scoring  
Patch score = mean distance  
Image score = 95th percentile  

---

## 🧪 Experimental Setup

Image Size: 224×224  
Batch Size: 32  
Feature Dimension: 256  
LSH Trees: 100  
Coreset Size: 5%  
k-NN: 9  
FPS: 8.1  

---

## 📊 Results

AUROC: 96.13%  
F1 Score: 95.44%  
Precision: 93.86%  
Recall: 96.81%  
FPS: 8.10  

---

## 📈 Comparison

Autoencoder → AUROC: 60.20  
Diffusion → AUROC: 63.49  
PatchCore-LSH → AUROC: 96.13  

---

## ✅ Advantages

- No training required  
- Memory efficient  
- Scalable  
- Real-time capable  

---

## 🚀 Future Work

- Higher resolution (448×448)  
- Mobile deployment  
- Ensemble LSH  
- Adaptive patch sizing  

---

## 👨‍💻 Authors

Abhimanyu Nema  
Krishka Kate  

---

## 📚 References

PatchCore (CVPR 2022)  
MVTec AD Dataset  
LSH (Indyk & Motwani)  
WideResNet  
