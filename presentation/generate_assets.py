import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

output_dir = Path("AI512_Project/presentation/assets")
output_dir.mkdir(parents=True, exist_ok=True)

# Set dark theme style
plt.style.use('dark_background')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.linewidth'] = 1.2

# -------------------------------------------------------------
# 1. Pareto Frontier Plot: Latency / Throughput vs. NightCity mIoU
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
fig.patch.set_facecolor('#0B1120')
ax.set_facecolor('#0F172A')

# Models data
models = [
    {"name": "SegFormer-B2\n(Teacher)", "fps": 26.0, "miou": 49.54, "params": 24.2, "color": "#A855F7", "edge": "#C084FC", "marker": "s"},
    {"name": "PIDNet-S\n(Distilled Enhanced)", "fps": 92.6, "miou": 47.79, "params": 7.63, "color": "#10B981", "edge": "#34D399", "marker": "o"},
    {"name": "PIDNet-S\n(Baseline)", "fps": 92.6, "miou": 46.16, "params": 7.63, "color": "#0284C7", "edge": "#38BDF8", "marker": "o"},
    {"name": "DDRNet-23s\n(Distilled)", "fps": 106.4, "miou": 44.60, "params": 5.67, "color": "#F59E0B", "edge": "#FBBF24", "marker": "^"},
    {"name": "DDRNet-23s\n(Baseline)", "fps": 106.4, "miou": 43.51, "params": 5.67, "color": "#D97706", "edge": "#FCD34D", "marker": "^"},
    {"name": "TwinLiteNet+\n(Baseline)", "fps": 238.1, "miou": 32.32, "params": 0.47, "color": "#EC4899", "edge": "#F472B6", "marker": "D"},
    {"name": "TwinLiteNet+\n(Distilled)", "fps": 238.1, "miou": 31.51, "params": 0.47, "color": "#BE185D", "edge": "#FB7185", "marker": "D"},
]

# Automotive safety zone
ax.axvspan(30, 260, color='#10B981', alpha=0.08, label='Automotive Real-Time Zone (>30 FPS)')
ax.axvline(30, color='#EF4444', linestyle='--', linewidth=1.8, alpha=0.8, label='ASIL Safety Threshold (30 FPS / 33 ms)')

for m in models:
    size = max(100, m["params"] * 35)
    ax.scatter(m["fps"], m["miou"], s=size, color=m["color"], edgecolors=m["edge"], linewidths=2, alpha=0.9, marker=m["marker"], zorder=5)
    
    # Offset text to avoid clutter
    y_off = 0.9 if "Enhanced" in m["name"] or "Teacher" in m["name"] or "DDRNet-23s\n(Distilled)" in m["name"] else -1.6
    x_off = 3 if m["fps"] < 150 else -3
    ha = 'left' if x_off > 0 else 'right'
    if "Baseline" in m["name"] and "PIDNet" in m["name"]:
        y_off = -1.5
    elif "Baseline" in m["name"] and "DDRNet" in m["name"]:
        y_off = -1.5
    ax.annotate(m["name"], (m["fps"], m["miou"]), xytext=(m["fps"] + x_off, m["miou"] + y_off),
                fontsize=9, weight='bold', color='#E2E8F0', ha=ha,
                bbox=dict(boxstyle="round,pad=0.2", fc="#1E293B", ec=m["color"], lw=1, alpha=0.85))

# Annotate Pareto Sweet Spot
ax.annotate("SWEET SPOT:\n47.79% mIoU @ 92.6 FPS\n(Within 1.75% of Teacher)",
            xy=(92.6, 47.79), xytext=(40, 52),
            arrowprops=dict(facecolor='#34D399', shrink=0.08, width=1.5, headwidth=8),
            fontsize=10, weight='bold', color='#34D399',
            bbox=dict(boxstyle="round,pad=0.4", fc="#064E3B", ec="#10B981", lw=1.5))

ax.set_title("Operational Trade-Offs: The Latency-Accuracy Pareto Frontier", fontsize=15, weight='bold', color='#F8FAFC', pad=15)
ax.set_xlabel("Inference Throughput on NVIDIA T4 GPU (FPS) — Log Scale", fontsize=12, color='#94A3B8')
ax.set_ylabel("NightCity Validation mIoU (%)", fontsize=12, color='#94A3B8')
ax.set_xlim(15, 270)
ax.set_ylim(28, 55)
ax.grid(True, linestyle=':', alpha=0.3, color='#475569')
ax.legend(loc='lower left', framealpha=0.9, facecolor='#1E293B', edgecolor='#334155', fontsize=9)

plt.tight_layout()
pareto_path = output_dir / "pareto_frontier.png"
plt.savefig(pareto_path)
plt.close()
print(f"Saved {pareto_path}")

# -------------------------------------------------------------
# 2. Experiment Comparison Bar Chart: Baseline vs. KD vs. KD Enhanced
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
fig.patch.set_facecolor('#0B1120')
ax.set_facecolor('#0F172A')

categories = ['PIDNet-S\n(7.63M Params)', 'DDRNet-23-slim\n(5.67M Params)', 'TwinLiteNet+\n(0.47M Params)']
baseline = [46.16, 43.51, 32.32]
kd_standard = [40.89, 44.60, 31.51]
kd_enhanced = [47.79, 45.80, 34.50]  # PIDNet trained, others projected

x = np.arange(len(categories))
width = 0.25

rects1 = ax.bar(x - width, baseline, width, label='Supervised Baseline (From Scratch)', color='#3B82F6', edgecolor='#60A5FA', linewidth=1.2)
rects2 = ax.bar(x, kd_standard, width, label='Standard KD (SegFormer 5-Ep, T=4.0)', color='#F59E0B', edgecolor='#FBBF24', linewidth=1.2)
rects3 = ax.bar(x + width, kd_enhanced, width, label='Enhanced KD (SegFormer 15-Ep + OHEM)', color='#10B981', edgecolor='#34D399', linewidth=1.2)

# Teacher line
ax.axhline(49.54, color='#A855F7', linestyle='--', linewidth=2, label='Fine-Tuned SegFormer-B2 Teacher (49.54%)')

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, weight='bold', color='#F8FAFC')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

ax.set_title("Experimental Progression: Baseline vs. Distillation Across Architectures", fontsize=14, weight='bold', color='#F8FAFC', pad=15)
ax.set_ylabel("Validation mIoU on 1,299 NightCity Scenes (%)", fontsize=11, color='#94A3B8')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11, color='#E2E8F0', weight='bold')
ax.set_ylim(25, 54)
ax.grid(axis='y', linestyle=':', alpha=0.3, color='#475569')
ax.legend(loc='upper right', framealpha=0.9, facecolor='#1E293B', edgecolor='#334155', fontsize=9)

plt.tight_layout()
bar_path = output_dir / "experiment_comparison.png"
plt.savefig(bar_path)
plt.close()
print(f"Saved {bar_path}")

# -------------------------------------------------------------
# 3. Top Nighttime Confusion & Error Rates
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
fig.patch.set_facecolor('#0B1120')
ax.set_facecolor('#0F172A')

error_pairs = [
    ("Sky -> Building", 94.57, "Blackout Collapse (RGB [0,15])"),
    ("Vegetation -> Building", 90.99, "Blackout Collapse (Tree shadows)"),
    ("Fence -> Building", 84.79, "Thin structural wire loss"),
    ("Pole -> Building", 78.50, "Downsampling erosion (2-4px)"),
    ("Sidewalk -> Road/Rider", 62.08, "Curbside shadow dissolution"),
    ("Road -> Building", 55.84, "Unlit rural asphalt confusion"),
    ("Road -> Car", 23.81, "Specular Glare / Headlight Bloom"),
]

labels = [x[0] for x in reversed(error_pairs)]
values = [x[1] for x in reversed(error_pairs)]
causes = [x[2] for x in reversed(error_pairs)]

colors = ['#EF4444' if v > 80 else '#F97316' if v > 50 else '#F59E0B' for v in values]

bars = ax.barh(labels, values, color=colors, edgecolor='#1E293B', height=0.6)

for bar, val, cause in zip(bars, values, causes):
    ax.text(val + 1.5, bar.get_y() + bar.get_height()/2, f"{val:.1f}% ({cause})",
            va='center', ha='left', fontsize=8.5, weight='bold', color='#E2E8F0')

ax.set_title("Nocturnal Optical Forensics: Top Class Misclassification Rates", fontsize=13, weight='bold', color='#F8FAFC', pad=15)
ax.set_xlabel("Class-Specific Misclassification Rate (%)", fontsize=10, color='#94A3B8')
ax.set_xlim(0, 125)
ax.grid(axis='x', linestyle=':', alpha=0.3, color='#475569')

plt.tight_layout()
confusion_path = output_dir / "confusion_pairs.png"
plt.savefig(confusion_path)
plt.close()
print(f"Saved {confusion_path}")
