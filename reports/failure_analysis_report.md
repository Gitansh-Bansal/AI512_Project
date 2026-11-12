# Failure Analysis Report (NightCity)

**Dataset:** NightCity (1,299 val, `512x1024`, 19 Cityscapes classes, ignore label 255)  
**Models:** PIDNet-S, DDRNet-23-slim, TwinLiteNet+ (`small`, 19-class head)

---

## 1. Failure Case Analysis (50 Validation Scenes)

### Top Confusion Pairs

| Rank | Ground Truth | Predicted As | Error Pixels | Class Misclassification Rate |
| :---: | :--- | :--- | :---: | :---: |
| 1 | `road` | `building` | 1,890,361 | 55.84% |
| 2 | `building` | `car` | 1,697,160 | 13.54% |
| 3 | `sky` | `building` | 1,587,786 | 94.57% |
| 4 | `car` | `building` | 1,019,723 | 50.23% |
| 5 | `road` | `car` | 805,998 | 23.81% |
| 6 | `sidewalk` | `rider` / `road` | 609,730 | 62.08% |
| 7 | `vegetation` | `building` | 400,485 | 90.99% |
| 8 | `pole` | `building` | 184,290 | 78.50% |
| 9 | `fence` | `building` | 162,065 | 84.79% |

### Representative Validation Scenes

- **`Chicago_0099` (4.93% mIoU, 31.2% Acc):** Unlit rural road; pitch-black upper half causes sky and road to be predicted as `building`.
- **`Chicago_0042` (5.61% mIoU, 35.8% Acc):** Strong oncoming headlights and wet-asphalt glare cause false `car` predictions on the road.
- **`Chicago_0017` (6.12% mIoU, 39.4% Acc):** Dark alley; dark vegetation and chain-link fences get absorbed into `building` (0% recall).
- **`Chicago_0058` (22.74% mIoU, 68.1% Acc):** Lit commercial avenue; `building` and `car` exceed 60% IoU, while sidewalks stay fragmented.

### Root Causes

1. **Blackout Collapse:** Night sky, dark trees, and unlit concrete share near-zero RGB values (`[0, 15]`), collapsing into `building` (94.6% of sky, 91.0% of vegetation).
2. **Specular Glare & Bloom:** Saturated headlights (`255`) and wet-road reflections trigger false object predictions (23.8% of road misclassified as `car`).
3. **Thin Object Loss:** Narrow classes (poles, lights, pedestrians, 2–8 px wide) fall below activation contrast and are washed out by 1/8–1/32 downsampling.
4. **Curbside Dissolution:** Without daytime curb shadows, sidewalks blend into road/rider (62.1% misclassified).

---

## 2. Model Bottlenecks & Next Directions

**Model Bottlenecks**
- **PIDNet-S:** Best baseline (**6.60% mIoU**, **42.70% Acc** on NightCity; **36.38% mIoU** on CamVid 20-ep scratch), but slowest on CPU (`2.25 FPS`) and its Boundary branch outputs noise in pitch-black or glare-saturated areas.
- **DDRNet-23-slim:** Fastest on full-resolution CPU inference (`3.56 FPS` on NightCity, `3.76 FPS` on CamVid), but 1/32 downsampling drops low-contrast small objects (**5.52% mIoU** NightCity).
- **TwinLiteNet+ (`small`):** Ultra-light (`0.128 M` params) and fast on `640x384` BDD100K (`17.35 FPS`, **78.51% DA mIoU**), but slows down at `512x1024` (`2.80 FPS`) and lacks channel capacity for 19 nighttime classes (**5.78% mIoU** NightCity).

**Candidate Directions (Tasks 6 & 7)**
1. **Illumination-Aware Channel Calibration (IACC):** Pool local luminance statistics to rescale feature channels across dark/glare regions (<0.05M params).
2. **Frequency/Wavelet Edge Filtering (DWT-RF):** Add a fast Haar wavelet high-pass filter to PIDNet's boundary branch to keep structural edges under glare/low contrast.
3. **Hardness-Aware Regional Loss:** Upweight under-exposed regions during training (zero inference cost).

---

**Artifacts**
- Visual comparison: `output/failure_analysis/failure_cases_visual_comparison.png`
- Summary JSON: `output/failure_analysis/failure_analysis_summary.json`
- Baseline results report: `AI512_Project/reports/baseline_results_report.md`
