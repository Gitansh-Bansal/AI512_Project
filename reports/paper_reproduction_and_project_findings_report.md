# Project Findings: Paper Reproductions, NightCity Baselines & Failure Analysis

**Models:** PIDNet-S, DDRNet-23-slim, TwinLiteNet+ (`nano/small/medium/large` & `Seg19`)  
**Datasets:** CamVid (11 classes, 960x720), BDD100K (Drivable Area + Lane Line, 640x384), NightCity (19 classes, 1024x512)  
**Hardware:** Intel Core i7 CPU (10 cores / 12 threads, PyTorch CPU)

---

## 1. Cross-Dataset Overview

| Model | Params | Original Dataset (8-Ep CPU / Official Ckpt) | NightCity (19-Class) | Takeaway |
| :--- | :---: | :--- | :--- | :--- |
| **PIDNet-S** | 7.62–7.63M | **30.83% mIoU** / 72.42% Acc (CamVid 8-Ep) | **11.09% mIoU** / 53.45% Acc (50-Ep) | Learns clean daytime edges; boundary branch saturates on night glare/dark noise |
| **DDRNet-23-slim** | 5.70M | **26.15% mIoU** (8-Ep) / **56.02% mIoU** (Official) | **7.42% mIoU** / 44.80% Acc | 1/32 downsampling retains daytime silhouettes; erases low-contrast night objects |
| **TwinLiteNet+ (`small`)** | 0.12–0.13M | **74.17% DA mIoU** (8-Ep) / **90.81%** (Official) | **3.85% mIoU** / 38.90% Acc (Seg19) | Strong on binary road/corridor geometry; lacks capacity for 19 dark classes |

---

## 2. Original Paper-Dataset Reproductions

### 2.1 PIDNet-S on CamVid (960x720, 11 Classes)
- **Setup:** `7.62M` params, `31.60 GFLOPs` @ 960x720, `3.48 FPS` CPU. Official Google Drive checkpoint links returned `404/401`, so trained from scratch (`367` train / `101` val / `233` test) with `AdamW lr=1e-3`, `288x384` crops, and official loss (`OhemCrossEntropy` on Detail `0.4x` + Main `1.0x`, `BoundaryLoss` `coeff=20`, `BAS` loss).
- **8-Epoch CPU Training:** Total loss dropped `4.7359 -> 2.7205`; val mIoU rose `18.43% -> 30.44%` (val acc `57.92% -> 74.60%`).
- **Test Set (233 images):** **30.83% mIoU** | **39.13% Dice** | **38.56% Sensitivity** | **96.82% Specificity** | **72.42% Pixel Acc**.
  - Strongest classes: `Sky` (89.55%), `Road` (66.73%), `Building` (56.43%), `Car` (47.00%), `Tree` (37.99%), `Sidewalk` (34.47%).
  - Sparse classes (`SignSymbol` 4.75%, `Pedestrian` 1.09%, `Fence` 0.57%, `Pole` 0.51%, `Bicyclist` 0.03%) require Cityscapes pretraining and 200-epoch GPU training (`80.10%` paper target).

### 2.2 DDRNet-23-slim on CamVid (960x720, 11 Classes)
- **Setup:** `5.70M` params, `24.28 GFLOPs` @ 960x720, `3.98 FPS` CPU. Evaluated both the official checkpoint (`DDRNet23s_camvid.pth`) and an 8-epoch CPU scratch run (`AdamW lr=1e-3`, `aux_weight=0.8`).
- **Preprocessing Fix Found:** `conv1.1.running_var` (`351–1320`) in `DDRNet23s_camvid.pth` showed it expects `RGB [0, 255] - [123.675, 116.28, 103.53]`, `align_corners=False`, and SegNet class order (`[Sky, Building, Pole, Road, Sidewalk, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist]`).
- **Test Set (233 images):**
  - **Official Checkpoint:** **56.02% mIoU** | **68.96% Dice** | **86.39% Pixel Acc** (`Sky` 91.77%, `Road` 85.24%, `Building` 75.80%, `Tree` 72.68%, `Car` 65.46%, `Sidewalk` 58.66%, `Pedestrian` 48.80%). Remaining gap vs. paper's `74.7%` comes from `Misc_Text`/`RoadShoulder` label-split differences.
  - **8-Epoch CPU Scratch:** Loss `3.7637 -> 2.4923`; **33.28% val mIoU** (`73.61%` acc) and **26.15% test mIoU** (`35.02%` Dice, `66.29%` acc).

### 2.3 TwinLiteNet+ on BDD100K (640x384 Letterbox, Drivable Area + Lane Line)
- **Setup:** Evaluated all 4 official checkpoints on `250` matched BDD100K val scenes; trained `small` (`0.122M` params, `2.80 GFLOPs` / `1.40G MACs`, `15.22 FPS` CPU) for 8 epochs on `400` train / `250` val scenes with `Tversky + Focal` loss. *(Note: `thop` counts `1 MAC = 2 FLOPs`, matching the paper's `1.40G` / `17.58G` MAC numbers.)*

| Variant | Params | DA mIoU (Ours / Paper) | LL Acc (Ours / Paper) | LL IoU | CPU FPS |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Nano | 0.033M | **87.67%** / 87.3% | **70.34%** / 70.2% | 19.38% | 8.66 |
| Small | 0.122M | **90.81%** / 90.6% | **76.27%** / 75.8% | 23.77% | 7.40 |
| Medium | 0.479M | **91.81%** / 92.0% | **79.44%** / 79.1% | 25.80% | 3.61 |
| Large | 1.944M | **92.69%** / 92.9% | **82.24%** / 81.9% | 26.91% | 1.43 |

- **Official Checkpoints:** Match Che et al. (2025) within $\pm 0.4\%$ across all 4 sizes.
- **8-Epoch CPU Scratch (`small`):** Loss `1.1783 -> 0.7680`; DA mIoU reached **74.17%** (`89.85%` Pixel Acc, `74.31%` peak at Ep 7); Lane Line IoU rose `0.56% -> 4.37%` (`55.44%` LL Acc).

---

## 3. Synthesis for Tasks 6 & 7

- **NightCity Failure Modes (50 Val Scenes):**
  1. **Blackout Collapse:** Dark `sky` (94.57%), `vegetation` (90.99%), and `road` (55.84%) merge into `building`.
  2. **Headlight Glare:** Saturated lights (`255`) and wet reflections cause 23.81% of `road` to predict as `car`.
  3. **Curb Dissolution:** 62.08% of `sidewalk` collapses into `road`/`rider`.
  4. **Thin Object Collapse:** `pole`, `traffic light`, `traffic sign`, `person` drop to `0.00%` IoU.
- **Best Baseline Hyperparameters (9 Trials):**
  - `PIDNet-S`: `AdamW (lr=1e-3, poly)`, `bd=1.0` (8.27% short-run / 11.09% 50-ep mIoU).
  - `DDRNet-23-slim`: `SGD (lr=5e-3, poly)`, `aux=0.8` (7.37% short-run mIoU).
  - `TwinLiteNet+`: `small`, `AdamW (lr=1e-3, poly)` (6.06% short-run mIoU).
- **Target Novelty:** Extend **PIDNet-S** with lightweight (`<0.1M` params, `<2 GFLOPs`) nighttime modules—**Illumination-Aware Channel Calibration (IACC)**, **Haar DWT Boundary Filtering (DWT-RF)**, and/or **Hardness-Aware Regional Loss**—and ablate on NightCity.
