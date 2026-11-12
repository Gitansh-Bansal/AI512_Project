# Baseline Results (NightCity & Daytime Reproductions)

**Dataset:** NightCity (2,998 train / 1,299 val, `512x1024`, 19 Cityscapes classes, ignore label 255)  
**Training Setup:** 30 epochs from scratch on CPU (`max_train_samples=8`, `batch_size=2`, `512x512` crop), using each model's best configuration from hyperparameter tuning, evaluated on **100 validation scenes** at `512x1024`.  
**Models:** PIDNet-S, DDRNet-23-slim, TwinLiteNet+ (`small`, 19-class head)

---

## 1. Benchmark Summary (NightCity, 100 Val Scenes)

| Metric | PIDNet-S | DDRNet-23-slim | TwinLiteNet+ (`small`) |
| :--- | :---: | :---: | :---: |
| Architecture | 3-branch (P / I / D) | Bilateral + DAPPM | ESPNet + CAAM |
| Best Config | `AdamW lr=0.001, bd=1.0` | `SGD lr=0.005, aux=0.8` | `small, AdamW lr=0.001` |
| Parameters | 7.63 M | 5.70 M | **0.128 M** |
| FLOPs | 95.46 G (`1024x2048`) | 73.34 G (`1024x2048`) | 13.12 G (`512x1024`) |
| CPU Latency (`512x1024`) | 444.9 ms (2.25 FPS) | **280.7 ms (3.56 FPS)** | 357.0 ms (2.80 FPS) |
| **Pixel Accuracy** | **42.70%** | 37.11% | 36.83% |
| **mIoU** | **6.60%** | 5.52% | 5.78% |
| **Mean Dice (F1)** | **9.76%** | 9.01% | 9.13% |
| **Mean Sensitivity** | **11.50%** | 9.26% | 10.99% |
| **Mean Specificity** | **94.90%** | 94.03% | 93.79% |

### Daytime Reproductions vs. NightCity Baseline

| Model | Daytime Reproduction (CPU Scratch) | Official Ckpt / Paper Target | NightCity Baseline (30 ep, 100 val) | Main Reason for Drop |
| :--- | :--- | :--- | :--- | :--- |
| **PIDNet-S** | **36.38% mIoU**, 76.91% Acc (CamVid `960x720`, 20 ep) | 80.10% mIoU (CamVid, Cityscapes pretrain) | **6.60% mIoU** / 42.70% Acc | Low illumination, glare, loss of edge contrast |
| **DDRNet-23-slim** | 26.15% mIoU, 66.29% Acc (CamVid `960x720`, 8 ep) | 56.02% mIoU (Official CPU) / 74.7%–78.6% Paper | **5.52% mIoU** / 37.11% Acc | 1/32 downsampling erases faint dark objects |
| **TwinLiteNet+ (`small`)** | **78.51% DA mIoU**, 64.67% LL Acc (BDD100K `640x384`, 20 ep) | 90.81% DA mIoU (Official) / 90.6% Paper | **5.78% mIoU** / 36.83% Acc (19 classes) | 0.128M capacity too small for 19 dark classes |

---

## 2. Training Convergence (30 Epochs, Best Hyperparameters)

- **PIDNet-S (`AdamW lr=0.001, bd=1.0`):** Reached **12.36% mIoU** on the 10-scene validation subset at Epoch 18, and **6.60% mIoU** across 100 validation scenes. Late-epoch boundary loss oscillates because sharp ground-truth edge labels conflict with blurred/noisy low-light gradients.
- **DDRNet-23-slim (`SGD lr=0.005, aux=0.8`):** Reached **10.15% mIoU** on the 10-scene validation subset and **5.52% mIoU** across 100 validation scenes. The auxiliary head (`aux=0.8`) helps preserve gradient flow into early bilateral blocks.
- **TwinLiteNet+ (`small, AdamW lr=0.001`):** Reached **13.15% mIoU** on the 10-scene validation subset at Epoch 28 and **5.78% mIoU** across 100 validation scenes.

---

## 3. Per-Class IoU Breakdown (100 NightCity Val Scenes)

| ID | Class | PIDNet-S IoU (%) | DDRNet-23-slim IoU (%) | TwinLiteNet+ IoU (%) |
| :---: | :--- | :---: | :---: | :---: |
| 0 | Road | **34.04** | 19.31 | 24.80 |
| 2 | Building | **44.58** | 36.55 | 33.67 |
| 13 | Car | 13.75 | 8.82 | **20.89** |
| 8 | Vegetation | 0.00 | **6.57** | 0.00 |
| 10 | Sky | 0.00 | **4.98** | 0.58 |
| 3 | Wall | 0.00 | **0.99** | 0.00 |
| 5 | Pole | 0.00 | 0.08 | **0.93** |
| 4 | Fence | 0.00 | 0.01 | **0.04** |
| 6 | Traffic Light | 0.00 | 0.00 | **0.01** |
| 1, 7, 9, 11, 12, 14–18 | Sidewalk, Sign, Terrain, Person, Rider, Heavy/2-Wheel | 0.00 | 0.00 | 0.00 |

- All three models only learn high-frequency, large-area classes (`building`, `road`, `car`) and collapse to `0.00%` IoU on small/rare classes (`sidewalk`, `traffic sign`, `person`, `rider`, `terrain`).
- **PIDNet-S** leads on `building` (44.58%) and `road` (34.04%), **TwinLiteNet+** leads on `car` (20.89%) and `pole` (0.93%), and **DDRNet-23-slim** is the only baseline to capture `vegetation` (6.57%) and `sky` (4.98%).

---

**Artifacts**
- NightCity notebooks: `AI512_Project/task3_pidnet_nightcity_baseline.ipynb`, `AI512_Project/task3_ddrnet_nightcity_baseline.ipynb`, `AI512_Project/task3_twinlitenetplus_nightcity_baseline.ipynb`
- Daytime notebooks: `AI512_Project/task3_pidnet_camvid_reproduction.ipynb`, `AI512_Project/task3_ddrnet_camvid_reproduction.ipynb`, `AI512_Project/task3_twinlitenetplus_bdd100k_reproduction.ipynb`
- Failure analysis report: `AI512_Project/reports/failure_analysis_report.md`
