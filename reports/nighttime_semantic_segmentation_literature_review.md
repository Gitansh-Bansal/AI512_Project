# Literature Review: Nighttime Semantic Segmentation

**Task:** Task 4 Review & Gap Analysis  
**Papers:** NightLab (CVPR 2022), Disentangle then Parse / DTP (ICCV 2023), FDLNet (IEEE TIP 2023)

---

## 1. Why Daytime Models Drop at Night

1. **Underexposure:** Unlit regions compress into narrow pixel ranges (`[0, 20]`), erasing local texture and object boundaries.
2. **Glare & Saturation:** Headlights, streetlamps, and wet-road reflections clip to `255` and bloom over nearby pixels.
3. **Color Loss:** Chromatic cues that separate sky, road, and sidewalk in daylight disappear into uniform dark tones.

---

## 2. Paper Breakdown

### 2.1 NightLab (CVPR 2022)
*Dual-Level Architecture With Hardness Detection for Segmentation at Night* — Deng et al. ([GitHub](https://github.com/xdeng7/NightLab))

- **Idea:** Nighttime degradation varies across an image; process easy regions coarsely and zoom into hard dark/small-object regions.
- **Architecture:**
  1. **Image-Level Network:** Global segmentation on the downscaled scene.
  2. **Hardness Detection Module (HDM):** Uses prediction entropy/margin uncertainty to propose bounding boxes around ambiguous patches without box labels.
  3. **Regional-Level Network + Fusion:** Crops and runs high-resolution segmentation on hard patches, then merges back into the global prediction.
- **Loss:** $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{image}} + \alpha \mathcal{L}_{\text{region}} + \beta \mathcal{L}_{\text{hardness}}$.
- **Limitation:** Two sequential forward passes plus dynamic cropping keep throughput under 5 FPS on high-end GPUs.

### 2.2 Disentangle then Parse / DTP (ICCV 2023)
*Night-time Semantic Segmentation with Illumination Disentanglement* — Wei et al. ([GitHub](https://github.com/w1oves/DTP))

- **Idea:** Use Retinex theory ($S = R \times I$) to separate lighting-invariant surface reflectance from illumination, then reuse lighting geometry (e.g., headlight cones) as spatial cues.
- **Architecture:**
  1. **Semantic-Oriented Disentanglement (SOD):** Splits features into Reflectance ($R$) and Illumination ($I$) using mutual-information minimization and reconstruction consistency.
  2. **Illumination-Aware Parser (IAParser):** Cross-attention between $R$ and $I$ so lighting patterns guide road and vehicle boundary parsing.
- **Loss:** Semantic cross-entropy + Retinex reconstruction ($\mathcal{L}_{\text{recon}}$) + illumination smoothness ($\mathcal{L}_{\text{smooth}}$).
- **Limitation:** Heavy dual-branch + cross-attention overhead (<15 FPS); Retinex division ($R = S / I$) degrades into noise in pitch-black areas where $I \approx 0$.

### 2.3 FDLNet (IEEE TIP 2023)
*Boosting Night-time Scene Parsing with Learnable Frequency* — Xie et al. ([GitHub](https://github.com/wangsen99/FDLNet))

- **Idea:** Lighting variations sit mainly in low-frequency spectral bands, while structural edges sit in high frequencies; filter features in the frequency domain instead of relying purely on spatial convs.
- **Architecture:**
  1. **Learnable Frequency Encoder (LFE):** Maps features via 2D DCT and applies learnable spectral channel weights to boost high frequencies and suppress lighting drift.
  2. **Spatial-Frequency Fusion (SFF):** Multi-scale cross-attention injects spectral edge cues back into the spatial backbone.
- **Loss:** Cross-entropy + Frequency Consistency Loss ($\mathcal{L}_{\text{freq}}$) aligning nighttime spectra with daytime priors.
- **Limitation:** Repeated multi-scale 2D DCT/FFT transforms add latency (<20 FPS); sharp glare spikes can cause ringing artifacts.

---

## 3. Comparison vs. Real-Time Baselines

| Method | Mechanism | Size | Real-Time Ready? | Main Limitation |
| :--- | :--- | :---: | :---: | :--- |
| **NightLab** (CVPR '22) | Global + HDM uncertainty crop | >60M | No (<5 FPS GPU) | Dual forward passes, sequential bottleneck |
| **DTP** (ICCV '23) | Retinex SOD + IAParser cross-attn | >40M | No (<15 FPS GPU) | Unstable when $I \approx 0$; heavy parser |
| **FDLNet** (TIP '23) | 2D DCT learnable spectral filter | >30M | No (<20 FPS GPU) | DCT/FFT latency; glare ringing |
| **PIDNet-S** | Detail / Context / Boundary branches | 7.63M | Yes (93 FPS GPU) | Boundary branch outputs noise in zero contrast |
| **DDRNet-23-slim** | Bilateral + DAPPM | 5.70M | Yes (102 FPS GPU) | 1/32 downsampling erases small dark objects |
| **TwinLiteNet+** | Depthwise ESPNet + CAAM | 0.13M | Yes (400+ FPS GPU) | Too little capacity for 19 nighttime classes |

---

## 4. Research Gap & Direction

- **Gap:** Specialized nighttime models (NightLab, DTP, FDLNet) rely on heavy multi-stage nets, Retinex cross-attention, or DCT transforms that break real-time budgets. Lightweight models (PIDNet-S, DDRNet-23-slim) run fast, but drop from ~78% daytime mIoU to <=11.1% on NightCity.
- **Target:** Add lightweight (<0.1M params, <2 GFLOPs) nighttime modules directly into PIDNet-S:
  1. **Illumination-Aware Channel Calibration (IACC):** Fast pooled luminance stats to reweight feature channels (lightweight version of DTP's idea).
  2. **Haar DWT Residual Boundary Filter (DWT-RF):** Add/subtract-only wavelet high-pass filtering on the Boundary branch (lightweight alternative to FDLNet's DCT).
  3. **Hardness-Aware Regional Loss:** Training-only loss upweighting dark/hard regions (lightweight alternative to NightLab's second pass, zero inference cost).
