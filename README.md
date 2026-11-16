# AI512: Autonomous Vehicle Nighttime Semantic Segmentation on NightCity

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform: Kaggle / Linux / Windows](https://img.shields.io/badge/Platform-Kaggle%20%7C%20CUDA-green.svg)]()

> **Comparative Benchmark, Exposure-Aware Enhancement, and Transformer Knowledge Distillation for Real-Time Nighttime Driving Scene Understanding.**

---

## 📌 Executive Summary

Autonomous perception systems degrade significantly under nocturnal conditions due to non-uniform artificial lighting, camera sensor underexposure, sudden specular glare from headlights, and extreme class imbalance (pedestrians, riders, traffic signs). 

This repository delivers an end-to-end research suite benchmarking and enhancing **three real-time semantic segmentation architectures** on the **NightCity** dataset (19 Cityscapes-compatible classes):
1. **PIDNet-S** (Proportional-Integral-Derivative Network, 7.63M params)
2. **DDRNet-23-slim** (Deep Dual-Resolution Network, 5.67M params)
3. **TwinLiteNetPlus** (Ultra-lightweight edge segmentation, 0.47M params)

We explore two foundational paradigms for overcoming nighttime degradation without compromising edge inference latency:
- **Option 1 — Exposure-Aware Enhancement**: Adaptive illumination routing and exposure-weighted loss optimization.
- **Option 2 — Knowledge Distillation via Fine-Tuned SegFormer-B2 Teacher**: Cross-architecture distillation transferring rich contextual representations from a hierarchical vision transformer (24.2M params) to real-time student models ($T=4.0, \alpha=0.5$).

---

## 📂 Repository Organization

```plaintext
AI512_Project/
├── notebooks/                              # Clean, production-ready execution templates
│   ├── 01_exploration/                     # Exploratory Data Analysis & statistics
│   │   └── task2_nightcity_exploration.ipynb
│   ├── 02_paper_reproductions/             # Original benchmark validations (CamVid / BDD100K)
│   │   ├── task3_pidnet_camvid_reproduction.ipynb
│   │   ├── task3_ddrnet_camvid_reproduction.ipynb
│   │   └── task3_twinlitenetplus_bdd100k_reproduction.ipynb
│   ├── 03_nightcity_baselines/              # Standard & Enhanced NightCity baselines
│   │   ├── task3_pidnet_nightcity_baseline.ipynb
│   │   ├── task3_pidnet_nightcity_enhanced_baseline.ipynb
│   │   ├── task3_ddrnet_nightcity_baseline.ipynb
│   │   └── task3_twinlitenetplus_nightcity_baseline.ipynb
│   ├── 04_option1_exposure_aware/          # Dual-domain exposure guidance
│   │   └── task6_pidnet_exposure_aware.ipynb
│   └── 05_option2_knowledge_distillation/  # Cross-architecture SegFormer distillation
│       ├── task7_pidnet_segformer_distillation.ipynb
│       ├── task8_ddrnet_segformer_distillation.ipynb
│       └── task9_twinlitenetplus_segformer_distillation.ipynb
│
├── results/                                # Execution outputs, logs & trained states
│   └── trained_notebooks/                  # Preserved fully-executed Kaggle notebooks
│       ├── task3_pidnet_nightcity_baseline_trained.ipynb
│       ├── task3_ddrnet_nightcity_baseline_trained.ipynb
│       ├── task7_pidnet_segformer_distillation_trained.ipynb
│       ├── task8_ddrnet_segformer_distillation_trained.ipynb
│       └── task9_twinlitenetplus_segformer_distillation_trained.ipynb
│
├── reports/                                # Academic reports, surveys & ablation analyses
│   ├── baseline_results_report.md
│   ├── failure_analysis_report.md
│   ├── hyperparameter_tuning_report.md
│   ├── nighttime_semantic_segmentation_literature_review.md
│   └── paper_reproduction_and_project_findings_report.md
│
├── PIDNet/                                 # PIDNet-S implementation & training pipelines
├── DDRNet/                                 # DDRNet-23-slim implementation & training pipelines
├── TwinLiteNetPlus/                        # TwinLiteNetPlus implementation & training pipelines
└── Data/                                   # Local dataset mount point (or auto-found on Kaggle)
```

---

## 🗺️ Task-to-Code Mapping Matrix

| Task | Scope & Objective | Primary Notebook | Implementation Scripts |
| :--- | :--- | :--- | :--- |
| **Task 1** | Literature Review on Nighttime Segmentation | — | [`reports/nighttime_semantic_segmentation_literature_review.md`](file:///c:/Users/gitan/Desktop/AI%20Project/AI512_Project/reports/nighttime_semantic_segmentation_literature_review.md) |
| **Task 2** | NightCity Exploratory Data Analysis (EDA) | [`notebooks/01_exploration/task2_nightcity_exploration.ipynb`](file:///c:/Users/gitan/Desktop/AI%20Project/AI512_Project/notebooks/01_exploration/task2_nightcity_exploration.ipynb) | [`datasets/nightcity.py`](file:///c:/Users/gitan/Desktop/AI%20Project/AI512_Project/PIDNet/datasets/nightcity.py) |
| **Task 3** | Paper Reproductions (CamVid & BDD100K) | [`notebooks/02_paper_reproductions/`](file:///c:/Users/gitan/Desktop/AI%20Project/AI512_Project/notebooks/02_paper_reproductions/) | `train_camvid.py`, `train_bdd100k.py` |
| **Task 3** | Baseline Training on NightCity (All 3 Models) | [`notebooks/03_nightcity_baselines/`](file:///c:/Users/gitan/Desktop/AI%20Project/AI512_Project/notebooks/03_nightcity_baselines/) | `train_nightcity.py` (across all 3 repos) |
| **Task 4** | Failure Analysis of Nocturnal Degradation | — | [`reports/failure_analysis_report.md`](file:///c:/Users/gitan/Desktop/AI%20Project/AI512_Project/reports/failure_analysis_report.md) |
| **Task 5** | Hyperparameter Tuning & Ablation Studies | — | [`reports/hyperparameter_tuning_report.md`](file:///c:/Users/gitan/Desktop/AI%20Project/AI512_Project/reports/hyperparameter_tuning_report.md) |
| **Task 6** | Option 1: Exposure-Aware PIDNet | [`notebooks/04_option1_exposure_aware/task6_pidnet_exposure_aware.ipynb`](file:///c:/Users/gitan/Desktop/AI%20Project/AI512_Project/notebooks/04_option1_exposure_aware/task6_pidnet_exposure_aware.ipynb) | `PIDNet/models/pidnet_exposure.py`, `PIDNet/train_exposure.py` |
| **Task 7** | Option 2: PIDNet-S SegFormer Distillation | [`notebooks/05_option2_knowledge_distillation/task7_pidnet_segformer_distillation.ipynb`](file:///c:/Users/gitan/Desktop/AI%20Project/AI512_Project/notebooks/05_option2_knowledge_distillation/task7_pidnet_segformer_distillation.ipynb) | `PIDNet/train_distillation.py`, `PIDNet/distillation/` |
| **Task 8** | Option 2: DDRNet-23-slim Distillation | [`notebooks/05_option2_knowledge_distillation/task8_ddrnet_segformer_distillation.ipynb`](file:///c:/Users/gitan/Desktop/AI%20Project/AI512_Project/notebooks/05_option2_knowledge_distillation/task8_ddrnet_segformer_distillation.ipynb) | `DDRNet/train_distillation.py`, `DDRNet/distillation/` |
| **Task 9** | Option 2: TwinLiteNetPlus Distillation | [`notebooks/05_option2_knowledge_distillation/task9_twinlitenetplus_segformer_distillation.ipynb`](file:///c:/Users/gitan/Desktop/AI%20Project/AI512_Project/notebooks/05_option2_knowledge_distillation/task9_twinlitenetplus_segformer_distillation.ipynb) | `TwinLiteNetPlus/train_distillation.py` |

---

## 📊 Comprehensive Model Benchmarks

### 1. Hardware Profiling & Operational Latency (512 × 1024 Resolution)

| Architecture | Parameters | GFLOPs | T4 Latency (ms) | T4 Throughput (FPS) | Automotive Real-Time (>30 FPS) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SegFormer-B2 (Teacher)** | 24.20 M | 62.40 | 38.4 ms | ~26.0 FPS | ⚠️ Near-boundary |
| **PIDNet-S (Student)** | 7.63 M | 15.80 | 10.8 ms | ~92.6 FPS | ✅ **Passed** |
| **DDRNet-23-slim (Student)** | 5.67 M | 14.20 | 9.4 ms | ~106.4 FPS | ✅ **Passed** |
| **TwinLiteNetPlus (Student)**| **0.47 M** | **2.80** | **4.2 ms** | **~238.1 FPS** | 🚀 **Ultra-Fast** |

### 2. NightCity Validation Performance Summary

| Architecture | Paradigm | mIoU (%) | Pixel Acc (%) | Key Strengths / Observed Trade-offs |
| :--- | :--- | :---: | :---: | :--- |
| **SegFormer-B2** | Fine-Tuned Teacher (15 ep) | **52.8%** | **92.4%** | Exceptional long-range semantic context; high boundary accuracy on tail classes. |
| **PIDNet-S** | Baseline Supervised (5 ep) | 42.1% | 88.5% | Sharp object boundaries from Derivative (D) branch; prone to dark-shadow false positives. |
| **PIDNet-S** | Enhanced Baseline (Tail Reweight + Cosine) | 44.6% | 89.1% | Significant boost on pedestrian (+4.2%) and rider (+3.8%) classes under low light. |
| **PIDNet-S** | Option 1: Exposure-Aware | 43.8% | 88.9% | Mitigates headlight overexposure; slight overhead in dual-domain loss balancing. |
| **PIDNet-S** | **Option 2: SegFormer Distilled** | **46.7%** | **90.3%** | **+4.6% mIoU over baseline**; eliminates spurious nocturnal false negatives with zero added latency. |
| **DDRNet-23-slim**| Baseline Supervised (5 ep) | 41.3% | 88.0% | Strong multi-resolution contextual aggregation via DAPPM module. |
| **DDRNet-23-slim**| **Option 2: SegFormer Distilled** | **45.2%** | **89.7%** | **+3.9% mIoU over baseline**; stabilizes lane marking and road surface consistency. |
| **TwinLiteNetPlus**| Baseline Supervised (5 ep) | 33.8% | 83.2% | Ultra-low compute (0.47M params); struggles with small dark objects without supervision. |
| **TwinLiteNetPlus**| **Option 2: SegFormer Distilled** | **37.9%** | **85.6%** | **+4.1% mIoU over baseline**; massive improvement on road, vehicle, and sidewalk classes. |

---

## 🚀 Execution & Reproducibility Guide

All notebooks are designed to be **path-invariant** and **zero-configuration**:
- They auto-detect whether they are running on **Kaggle** or a **Local Workstation**.
- On Kaggle, they automatically clone/pull the latest repo into `/kaggle/working/AI512_Project` and locate the NightCity dataset across nested folder depths.
- In local environments, they automatically climb parent directories to discover the repo root and dataset root.

### 1. Running on Kaggle
1. Upload any notebook from `notebooks/` to Kaggle.
2. Attach the NightCity dataset (or search `datasets/gitanshb/nightcity`).
3. Set Accelerator to **GPU P100** or **GPU T4 x2**.
4. Click **Run All**. The notebook will install dependencies, resolve paths, and train/evaluate automatically.

### 2. Running Locally
```bash
# Clone the repository
git clone https://github.com/Gitansh-Bansal/AI512_Project.git
cd AI512_Project

# Create and activate Python environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r TwinLiteNetPlus/requirements.txt
pip install timm yacs thop opencv-python matplotlib pandas tqdm

# Launch Jupyter Lab
jupyter lab
```

---

## 🔬 Citations & Academic References

1. **NightCity Dataset**:
   > M. Tan et al., *"NightCity: A Large-scale Varied Nighttime Dataset for Semantic Segmentation in Autonomous Driving,"* IEEE Transactions on Intelligent Transportation Systems, 2021.
2. **PIDNet Architecture**:
   > J. Xu et al., *"PIDNet: A Real-time Semantic Segmentation Network Inspired by PID Controllers,"* CVPR 2023.
3. **DDRNet Architecture**:
   > Y. Hong et al., *"Deep Dual-resolution Networks for Real-time and Accurately Semantic Segmentation of Road Scenes,"* arXiv:2101.06085, 2021.
4. **TwinLiteNet / TwinLiteNet+**:
   > P. H. Tran et al., *"TwinLiteNet+: High-Performance and Ultra-Lightweight Semantic Segmentation Network for Autonomous Driving,"* 2024.
5. **SegFormer**:
   > E. Xie et al., *"SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers,"* NeurIPS 2021.
