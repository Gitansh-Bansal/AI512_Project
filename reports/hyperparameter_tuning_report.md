# Hyperparameter Tuning Report (NightCity)

**Dataset:** NightCity (1024x512, 19 classes)  
**Models:** PIDNet-S, DDRNet-23-slim, TwinLiteNet+ (Seg19)  
**Trials:** 9 controlled runs testing optimizer (SGD vs. AdamW), LR & scheduler (Poly vs. Cosine), loss/branch weights, and model width.

---

## 1. Trial Results

| Model | Trial | Configuration | Final Loss | Val mIoU | Pixel Acc | Val Dice | Rank |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| PIDNet-S | 1 | SGD (`lr=0.01`, poly), `bd=1.0`, unweighted | 3.4363 | 8.09% | 55.63% | 11.85% | 2 |
| **PIDNet-S** | **2** | **AdamW (`lr=0.001`, poly), `bd=1.0`, unweighted** | **3.6597** | **8.27%** | **57.64%** | **12.14%** | **1** |
| PIDNet-S | 3 | SGD (`lr=0.005`, poly), `bd=0.4`, class-weighted | 3.7373 | 5.74% | 50.67% | 8.92% | 3 |
| DDRNet-23-slim | 1 | SGD (`lr=0.01`, poly), `aux=0.4`, unweighted | 2.0596 | 6.76% | 55.20% | 10.45% | 2 |
| DDRNet-23-slim | 2 | AdamW (`lr=0.001`, cosine), `aux=0.4`, unweighted | 2.0903 | 6.14% | 56.61% | 9.88% | 3 |
| **DDRNet-23-slim** | **3** | **SGD (`lr=0.005`, poly), `aux=0.8`, class-weighted** | **3.0082** | **7.37%** | **55.89%** | **11.20%** | **1** |
| **TwinLiteNet+** | **1** | **`small`, AdamW (`lr=0.001`, poly), unweighted** | **2.3017** | **6.06%** | **42.37%** | **9.15%** | **1** |
| TwinLiteNet+ | 2 | `small`, AdamW (`lr=0.0003`, cosine), unweighted | 2.7169 | 4.52% | 15.15% | 6.42% | 3 |
| TwinLiteNet+ | 3 | `medium`, AdamW (`lr=0.0005`, poly), class-weighted | 2.6902 | 4.89% | 26.54% | 7.10% | 2 |

---

## 2. Key Findings

- **PIDNet-S:**
  - `AdamW (lr=1e-3, wd=1e-4)` gave the best result overall (8.27% mIoU, 57.64% pixel acc), handling gradient scale differences between the Detail and Boundary branches better than SGD.
  - Lowering boundary weight (`1.0 -> 0.4`) dropped mIoU by 2.53 points; aggressive inverse-frequency class weighting hurt dominant road/building accuracy (50.67% vs. 57.64% pixel acc).
- **DDRNet-23-slim:**
  - Raising auxiliary supervision (`0.4 -> 0.8`) produced the best mIoU (7.37%), helping preserve low-contrast features before 1/32 downsampling.
  - SGD + poly decay reached a lower training loss than AdamW (2.0596 vs. 2.0903).
- **TwinLiteNet+ (Seg19):**
  - Sensitive to low learning rates: `lr=3e-4` underfit badly (15.15% pixel acc); `AdamW lr=1e-3` on `small` worked best (6.06% mIoU).
  - Scaling to `medium` did not improve early mIoU (4.89% vs. 6.06%).

---

## 3. Recommended Full-Training Settings

| Setting | PIDNet-S | DDRNet-23-slim |
| :--- | :--- | :--- |
| Optimizer | AdamW | SGD (momentum 0.9) |
| Base LR | `1e-3` | `5e-3` |
| Weight Decay | `1e-4` | `5e-4` |
| LR Schedule | Poly (`power=0.9`) | Poly (`power=0.9`) |
| Branch Weight | `bd = 1.0` | `aux = 0.8` |
| Loss | OHEM Cross-Entropy (`thresh=0.7`) | OHEM Cross-Entropy (`thresh=0.7`) |
| Crop / Batch / Epochs | `512x1024` / `4–8` / `50` | `512x1024` / `4–8` / `50` |

*(For TwinLiteNet+ Seg19: `small` config, AdamW `lr=1e-3`, poly decay.)*

---

**Artifacts**
- Results JSON: `output/hyperparameter_study/hyperparameter_study_results.json`
- Loss plot: `output/hyperparameter_study/hyperparameter_loss_curves_comparison.png`
- Metric plot: `output/hyperparameter_study/hyperparameter_metrics_comparison.png`
