import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

output_dir = Path("AI512_Project/presentation/assets")
output_dir.mkdir(parents=True, exist_ok=True)

plt.style.use('dark_background')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

# ----------------------------------------------------------------------
# 1. PIDNet-S Architecture Diagram
# ----------------------------------------------------------------------
def draw_pidnet():
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=300)
    fig.patch.set_facecolor('#0B1120')
    ax.set_facecolor('#0F172A')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Title
    ax.text(5, 6.6, "PIDNet-S: Three-Branch Control Architecture", ha='center', va='center',
            fontsize=12, weight='bold', color='#38BDF8')

    # Input image
    box_in = patches.FancyBboxPatch((0.2, 2.5), 1.0, 1.6, boxstyle="round,pad=0.1", fc='#1E293B', ec='#64748B', lw=1.5)
    ax.add_patch(box_in)
    ax.text(0.7, 3.3, "Input\nImage\n(H, W, 3)", ha='center', va='center', fontsize=8, color='#F8FAFC', weight='bold')

    # Arrow from input to branches
    ax.annotate('', xy=(1.8, 4.8), xytext=(1.3, 3.7), arrowprops=dict(arrowstyle="->", color='#38BDF8', lw=1.5))
    ax.annotate('', xy=(1.8, 3.3), xytext=(1.3, 3.3), arrowprops=dict(arrowstyle="->", color='#F59E0B', lw=1.5))
    ax.annotate('', xy=(1.8, 1.8), xytext=(1.3, 2.9), arrowprops=dict(arrowstyle="->", color='#10B981', lw=1.5))

    # P-Branch (Detail / Proportional)
    box_p = patches.FancyBboxPatch((1.9, 4.2), 2.2, 1.2, boxstyle="round,pad=0.1", fc='#0C4A6E', ec='#38BDF8', lw=1.8)
    ax.add_patch(box_p)
    ax.text(3.0, 4.8, "P-Branch (Detail)", ha='center', va='center', fontsize=9, color='#38BDF8', weight='bold')
    ax.text(3.0, 4.4, "H/4 x W/4 (High-Res)\nSpatial Details", ha='center', va='center', fontsize=7.5, color='#E2E8F0')

    # D-Branch (Boundary / Derivative)
    box_d = patches.FancyBboxPatch((1.9, 2.7), 2.2, 1.2, boxstyle="round,pad=0.1", fc='#78350F', ec='#F59E0B', lw=1.8)
    ax.add_patch(box_d)
    ax.text(3.0, 3.3, "D-Branch (Boundary)", ha='center', va='center', fontsize=9, color='#FBBF24', weight='bold')
    ax.text(3.0, 2.9, "H/8 x W/8 (Edges)\nHigh-Freq Gradients", ha='center', va='center', fontsize=7.5, color='#E2E8F0')

    # I-Branch (Context / Integral)
    box_i = patches.FancyBboxPatch((1.9, 1.2), 2.2, 1.2, boxstyle="round,pad=0.1", fc='#064E3B', ec='#10B981', lw=1.8)
    ax.add_patch(box_i)
    ax.text(3.0, 1.8, "I-Branch (Context)", ha='center', va='center', fontsize=9, color='#34D399', weight='bold')
    ax.text(3.0, 1.4, "H/32 x W/32 (Low-Res)\nGlobal Semantics", ha='center', va='center', fontsize=7.5, color='#E2E8F0')

    # PAG Module 1 (between P and I)
    box_pag = patches.FancyBboxPatch((4.7, 3.6), 1.8, 1.2, boxstyle="round,pad=0.1", fc='#1E293B', ec='#A855F7', lw=1.5)
    ax.add_patch(box_pag)
    ax.text(5.6, 4.2, "PAG Module", ha='center', va='center', fontsize=8.5, color='#C084FC', weight='bold')
    ax.text(5.6, 3.8, "Pixel-Attention\nGuided Fusion", ha='center', va='center', fontsize=7.5, color='#E2E8F0')

    # Connect P and I to PAG
    ax.annotate('', xy=(4.6, 4.4), xytext=(4.2, 4.6), arrowprops=dict(arrowstyle="->", color='#38BDF8', lw=1.2))
    ax.annotate('', xy=(4.6, 3.9), xytext=(4.2, 1.9), arrowprops=dict(arrowstyle="->", color='#10B981', lw=1.2))

    # Boundary Head
    box_bdhead = patches.FancyBboxPatch((4.7, 2.3), 1.8, 0.9, boxstyle="round,pad=0.1", fc='#451A03', ec='#F59E0B', lw=1.2)
    ax.add_patch(box_bdhead)
    ax.text(5.6, 2.75, "Boundary Loss Head\n(CE + Dice)", ha='center', va='center', fontsize=7.5, color='#FCD34D', weight='bold')
    ax.annotate('', xy=(4.6, 2.75), xytext=(4.2, 3.1), arrowprops=dict(arrowstyle="->", color='#F59E0B', lw=1.2))

    # PAP / BDF Module (Boundary-Guided Fusion)
    box_bdf = patches.FancyBboxPatch((7.0, 3.0), 1.5, 1.8, boxstyle="round,pad=0.1", fc='#1E293B', ec='#E11D48', lw=1.5)
    ax.add_patch(box_bdf)
    ax.text(7.75, 4.0, "PAP Module", ha='center', va='center', fontsize=8.5, color='#FB7185', weight='bold')
    ax.text(7.75, 3.4, "Boundary-Guided\nAggregation\n(P + I + D)", ha='center', va='center', fontsize=7, color='#E2E8F0')

    # Connect PAG and D-Branch to PAP
    ax.annotate('', xy=(6.9, 4.1), xytext=(6.6, 4.1), arrowprops=dict(arrowstyle="->", color='#C084FC', lw=1.2))
    ax.annotate('', xy=(6.9, 3.5), xytext=(4.2, 3.2), arrowprops=dict(arrowstyle="->", color='#F59E0B', lw=1.2))

    # Final Output
    box_out = patches.FancyBboxPatch((8.8, 3.3), 1.0, 1.2, boxstyle="round,pad=0.1", fc='#065F46', ec='#10B981', lw=1.5)
    ax.add_patch(box_out)
    ax.text(9.3, 3.9, "Final\nLogits\n(19 classes)", ha='center', va='center', fontsize=7.5, color='#F8FAFC', weight='bold')
    ax.annotate('', xy=(8.7, 3.9), xytext=(8.6, 3.9), arrowprops=dict(arrowstyle="->", color='#10B981', lw=1.5))

    # Caption box
    ax.text(5, 0.4, "Mathematical Analogy: u(t) = Kp*e(t) + Ki*int(e(tau)) + Kd*(de/dt)\nDerivative branch explicitly corrects dark-region boundary bleed.",
            ha='center', va='center', fontsize=8, color='#94A3B8', style='italic',
            bbox=dict(boxstyle="round,pad=0.3", fc="#1E293B", ec="#334155", lw=1))

    plt.tight_layout()
    plt.savefig(output_dir / "arch_pidnet.png")
    plt.close()
    print("Saved arch_pidnet.png")

# ----------------------------------------------------------------------
# 2. DDRNet-23-slim Architecture Diagram
# ----------------------------------------------------------------------
def draw_ddrnet():
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=300)
    fig.patch.set_facecolor('#0B1120')
    ax.set_facecolor('#0F172A')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    ax.text(5, 6.6, "DDRNet-23-slim: Deep Dual-Resolution Architecture", ha='center', va='center',
            fontsize=12, weight='bold', color='#F59E0B')

    # Input image
    box_in = patches.FancyBboxPatch((0.2, 2.7), 1.0, 1.5, boxstyle="round,pad=0.1", fc='#1E293B', ec='#64748B', lw=1.5)
    ax.add_patch(box_in)
    ax.text(0.7, 3.45, "Input\nImage\n(H, W, 3)", ha='center', va='center', fontsize=8, color='#F8FAFC', weight='bold')

    # Stem
    box_stem = patches.FancyBboxPatch((1.5, 2.7), 1.1, 1.5, boxstyle="round,pad=0.1", fc='#1E293B', ec='#38BDF8', lw=1.2)
    ax.add_patch(box_stem)
    ax.text(2.05, 3.45, "Stem\nConv\n(H/4, W/4)", ha='center', va='center', fontsize=7.5, color='#38BDF8', weight='bold')
    ax.annotate('', xy=(1.4, 3.45), xytext=(1.3, 3.45), arrowprops=dict(arrowstyle="->", color='#64748B', lw=1.2))

    # High-Res Branch
    box_high1 = patches.FancyBboxPatch((3.0, 4.3), 1.6, 1.1, boxstyle="round,pad=0.1", fc='#0C4A6E', ec='#38BDF8', lw=1.5)
    ax.add_patch(box_high1)
    ax.text(3.8, 4.85, "High-Res Branch 1\n(H/8 x W/8, 32ch)", ha='center', va='center', fontsize=7.5, color='#F8FAFC', weight='bold')

    box_high2 = patches.FancyBboxPatch((5.4, 4.3), 1.6, 1.1, boxstyle="round,pad=0.1", fc='#0C4A6E', ec='#38BDF8', lw=1.5)
    ax.add_patch(box_high2)
    ax.text(6.2, 4.85, "High-Res Branch 2\n(H/8 x W/8, 64ch)", ha='center', va='center', fontsize=7.5, color='#F8FAFC', weight='bold')

    # Low-Res Branch
    box_low1 = patches.FancyBboxPatch((3.0, 1.7), 1.6, 1.1, boxstyle="round,pad=0.1", fc='#78350F', ec='#F59E0B', lw=1.5)
    ax.add_patch(box_low1)
    ax.text(3.8, 2.25, "Low-Res Branch 1\n(H/16 x W/16, 64ch)", ha='center', va='center', fontsize=7.5, color='#F8FAFC', weight='bold')

    box_low2 = patches.FancyBboxPatch((5.4, 1.7), 1.6, 1.1, boxstyle="round,pad=0.1", fc='#78350F', ec='#F59E0B', lw=1.5)
    ax.add_patch(box_low2)
    ax.text(6.2, 2.25, "Low-Res Branch 2\n(H/32 x W/32, 128ch)", ha='center', va='center', fontsize=7.5, color='#F8FAFC', weight='bold')

    # Connect Stem to both branches
    ax.annotate('', xy=(2.9, 4.85), xytext=(2.7, 3.7), arrowprops=dict(arrowstyle="->", color='#38BDF8', lw=1.2))
    ax.annotate('', xy=(2.9, 2.25), xytext=(2.7, 3.2), arrowprops=dict(arrowstyle="->", color='#F59E0B', lw=1.2))

    # Bilateral Cross Connections
    ax.annotate('', xy=(4.6, 4.6), xytext=(3.8, 2.9), arrowprops=dict(arrowstyle="<->", color='#10B981', lw=1.8))
    ax.text(4.7, 3.5, "Bilateral\nFusion", ha='center', va='center', fontsize=7, color='#34D399', weight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc="#064E3B", ec="#10B981", lw=1))
    ax.annotate('', xy=(5.3, 4.85), xytext=(4.7, 4.85), arrowprops=dict(arrowstyle="->", color='#38BDF8', lw=1.2))
    ax.annotate('', xy=(5.3, 2.25), xytext=(4.7, 2.25), arrowprops=dict(arrowstyle="->", color='#F59E0B', lw=1.2))

    # DAPPM Module
    box_dappm = patches.FancyBboxPatch((7.3, 1.4), 1.5, 1.6, boxstyle="round,pad=0.1", fc='#4C1D95', ec='#A855F7', lw=1.5)
    ax.add_patch(box_dappm)
    ax.text(8.05, 2.4, "DAPPM", ha='center', va='center', fontsize=9, color='#C084FC', weight='bold')
    ax.text(8.05, 1.85, "Deep Aggregation\nPyramid Pooling\n(k=1, 2, 3, 6)", ha='center', va='center', fontsize=7, color='#E2E8F0')
    ax.annotate('', xy=(7.2, 2.25), xytext=(7.1, 2.25), arrowprops=dict(arrowstyle="->", color='#F59E0B', lw=1.2))

    # Final Aggregation & Output
    box_out = patches.FancyBboxPatch((7.3, 3.8), 1.5, 1.6, boxstyle="round,pad=0.1", fc='#065F46', ec='#10B981', lw=1.5)
    ax.add_patch(box_out)
    ax.text(8.05, 4.8, "Final Head", ha='center', va='center', fontsize=8.5, color='#34D399', weight='bold')
    ax.text(8.05, 4.3, "High-Res +\nDAPPM Upsample\n(106 FPS)", ha='center', va='center', fontsize=7, color='#E2E8F0')

    ax.annotate('', xy=(7.2, 4.85), xytext=(7.1, 4.85), arrowprops=dict(arrowstyle="->", color='#38BDF8', lw=1.2))
    ax.annotate('', xy=(8.05, 3.7), xytext=(8.05, 3.1), arrowprops=dict(arrowstyle="->", color='#A855F7', lw=1.2))

    # Caption box
    ax.text(5, 0.4, "Dual-Resolution Design: Keeps high-res spatial stream uncompressed while DAPPM aggregates context.\nThroughput: 106.4 FPS on T4 GPU (Maximum speed among 19-class models).",
            ha='center', va='center', fontsize=8, color='#94A3B8', style='italic',
            bbox=dict(boxstyle="round,pad=0.3", fc="#1E293B", ec="#334155", lw=1))

    plt.tight_layout()
    plt.savefig(output_dir / "arch_ddrnet.png")
    plt.close()
    print("Saved arch_ddrnet.png")

# ----------------------------------------------------------------------
# 3. TwinLiteNet+ Architecture Diagram
# ----------------------------------------------------------------------
def draw_twinlite():
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=300)
    fig.patch.set_facecolor('#0B1120')
    ax.set_facecolor('#0F172A')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    ax.text(5, 6.6, "TwinLiteNet+: Depthwise ESPNet & Channel Attention", ha='center', va='center',
            fontsize=12, weight='bold', color='#EC4899')

    # Input image
    box_in = patches.FancyBboxPatch((0.2, 2.7), 1.0, 1.5, boxstyle="round,pad=0.1", fc='#1E293B', ec='#64748B', lw=1.5)
    ax.add_patch(box_in)
    ax.text(0.7, 3.45, "Input\nImage\n(H, W, 3)", ha='center', va='center', fontsize=8, color='#F8FAFC', weight='bold')

    # Encoder Stages (ESPNet-style)
    box_enc1 = patches.FancyBboxPatch((1.6, 2.7), 1.6, 1.5, boxstyle="round,pad=0.1", fc='#831843', ec='#EC4899', lw=1.2)
    ax.add_patch(box_enc1)
    ax.text(2.4, 3.65, "Encoder Stage 1", ha='center', va='center', fontsize=8, color='#F472B6', weight='bold')
    ax.text(2.4, 3.2, "Depthwise Conv\nDownsample /2", ha='center', va='center', fontsize=7, color='#E2E8F0')
    ax.annotate('', xy=(1.5, 3.45), xytext=(1.3, 3.45), arrowprops=dict(arrowstyle="->", color='#64748B', lw=1.2))

    box_esp = patches.FancyBboxPatch((3.6, 2.2), 2.2, 2.5, boxstyle="round,pad=0.1", fc='#1E293B', ec='#EC4899', lw=1.5)
    ax.add_patch(box_esp)
    ax.text(4.7, 4.3, "ESPNet Hierarchical Module", ha='center', va='center', fontsize=8.5, color='#F472B6', weight='bold')
    ax.text(4.7, 3.7, "Dilated Depthwise Convs\nParallel Dilation: d=1, 2, 4, 8\nHierarchical Additive Fusion\n(9x FLOP reduction)",
            ha='center', va='center', fontsize=7, color='#E2E8F0')
    ax.annotate('', xy=(3.5, 3.45), xytext=(3.3, 3.45), arrowprops=dict(arrowstyle="->", color='#EC4899', lw=1.2))

    # CAAM Module (Channel Attention Adaptive Module)
    box_caam = patches.FancyBboxPatch((6.2, 2.5), 1.8, 1.9, boxstyle="round,pad=0.1", fc='#500724', ec='#F43F5E', lw=1.5)
    ax.add_patch(box_caam)
    ax.text(7.1, 3.8, "CAAM Module", ha='center', va='center', fontsize=8.5, color='#FB7185', weight='bold')
    ax.text(7.1, 3.1, "Channel Attention\nGlobal Pooling\nMLP Recalibration\nSigmoid Gate",
            ha='center', va='center', fontsize=7, color='#E2E8F0')
    ax.annotate('', xy=(6.1, 3.45), xytext=(5.9, 3.45), arrowprops=dict(arrowstyle="->", color='#EC4899', lw=1.2))

    # Dual Head / 19-class Head
    box_head = patches.FancyBboxPatch((8.4, 2.7), 1.3, 1.5, boxstyle="round,pad=0.1", fc='#065F46', ec='#10B981', lw=1.5)
    ax.add_patch(box_head)
    ax.text(9.05, 3.65, "Output Head", ha='center', va='center', fontsize=8, color='#34D399', weight='bold')
    ax.text(9.05, 3.2, "19-Class Seg\n(238.1 FPS)\n0.47M Params", ha='center', va='center', fontsize=7, color='#E2E8F0')
    ax.annotate('', xy=(8.3, 3.45), xytext=(8.1, 3.45), arrowprops=dict(arrowstyle="->", color='#F43F5E', lw=1.2))

    # Caption box
    ax.text(5, 0.4, "Ultra-Lightweight Design: Factorized depthwise convolutions keep parameter budget under 0.5M.\nInference: 4.2ms latency (238 FPS) — optimal for ADAS microcontrollers.",
            ha='center', va='center', fontsize=8, color='#94A3B8', style='italic',
            bbox=dict(boxstyle="round,pad=0.3", fc="#1E293B", ec="#334155", lw=1))

    plt.tight_layout()
    plt.savefig(output_dir / "arch_twinlite.png")
    plt.close()
    print("Saved arch_twinlite.png")

# ----------------------------------------------------------------------
# 4. SegFormer-B2 Architecture Diagram (Teacher)
# ----------------------------------------------------------------------
def draw_segformer():
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=300)
    fig.patch.set_facecolor('#0B1120')
    ax.set_facecolor('#0F172A')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    ax.text(5, 6.6, "SegFormer-B2: Hierarchical Vision Transformer Teacher", ha='center', va='center',
            fontsize=12, weight='bold', color='#A855F7')

    # Input image
    box_in = patches.FancyBboxPatch((0.2, 2.6), 1.0, 1.6, boxstyle="round,pad=0.1", fc='#1E293B', ec='#64748B', lw=1.5)
    ax.add_patch(box_in)
    ax.text(0.7, 3.4, "Input\nImage\n(H, W, 3)", ha='center', va='center', fontsize=8, color='#F8FAFC', weight='bold')

    # 4 Transformer Encoder Stages (MiT-B2)
    stages = [
        ("Stage 1", "H/4 x W/4\nC=64", 1.5),
        ("Stage 2", "H/8 x W/8\nC=128", 2.9),
        ("Stage 3", "H/16 x W/16\nC=320", 4.3),
        ("Stage 4", "H/32 x W/32\nC=512", 5.7),
    ]
    for name, shape, x in stages:
        box_st = patches.FancyBboxPatch((x, 2.4), 1.2, 2.0, boxstyle="round,pad=0.1", fc='#3B0764', ec='#A855F7', lw=1.2)
        ax.add_patch(box_st)
        ax.text(x + 0.6, 3.9, name, ha='center', va='center', fontsize=8, color='#C084FC', weight='bold')
        ax.text(x + 0.6, 3.4, "Overlapped\nPatch Merge\n+\nSpatial Red.\nSelf-Attn", ha='center', va='center', fontsize=6.2, color='#E2E8F0')
        ax.text(x + 0.6, 2.7, shape, ha='center', va='center', fontsize=6.5, color='#38BDF8', weight='bold')

    ax.annotate('', xy=(1.4, 3.4), xytext=(1.3, 3.4), arrowprops=dict(arrowstyle="->", color='#64748B', lw=1.2))
    ax.annotate('', xy=(2.8, 3.4), xytext=(2.7, 3.4), arrowprops=dict(arrowstyle="->", color='#A855F7', lw=1.2))
    ax.annotate('', xy=(4.2, 3.4), xytext=(4.1, 3.4), arrowprops=dict(arrowstyle="->", color='#A855F7', lw=1.2))
    ax.annotate('', xy=(5.6, 3.4), xytext=(5.5, 3.4), arrowprops=dict(arrowstyle="->", color='#A855F7', lw=1.2))

    # All-MLP Decoder
    box_dec = patches.FancyBboxPatch((7.2, 2.2), 1.5, 2.4, boxstyle="round,pad=0.1", fc='#1E293B', ec='#06B6D4', lw=1.5)
    ax.add_patch(box_dec)
    ax.text(7.95, 4.1, "All-MLP Decoder", ha='center', va='center', fontsize=8.5, color='#22D3EE', weight='bold')
    ax.text(7.95, 3.3, "1. Linear Layers\n2. Upsample to H/4\n3. Concat Multi-Level\n4. Predict 19 Classes",
            ha='center', va='center', fontsize=7, color='#E2E8F0')

    # Connections to decoder
    ax.annotate('', xy=(7.1, 3.6), xytext=(6.9, 3.4), arrowprops=dict(arrowstyle="->", color='#A855F7', lw=1.2))

    # Final Soft Teacher Logits
    box_logits = patches.FancyBboxPatch((8.9, 2.8), 0.9, 1.2, boxstyle="round,pad=0.1", fc='#4C1D95', ec='#C084FC', lw=1.5)
    ax.add_patch(box_logits)
    ax.text(9.35, 3.4, "Soft\nLogits\n(T=4.0)", ha='center', va='center', fontsize=7.5, color='#F8FAFC', weight='bold')
    ax.annotate('', xy=(8.8, 3.4), xytext=(8.7, 3.4), arrowprops=dict(arrowstyle="->", color='#06B6D4', lw=1.2))

    # Caption box
    ax.text(5, 0.4, "Global Context Engine: Mix Transformer eliminates fixed positional embeddings & captures long-range dark dependencies.\nPerformance: 49.54% mIoU on NightCity (Peak benchmark quality, but 26 FPS limits direct deployment).",
            ha='center', va='center', fontsize=8, color='#94A3B8', style='italic',
            bbox=dict(boxstyle="round,pad=0.3", fc="#1E293B", ec="#334155", lw=1))

    plt.tight_layout()
    plt.savefig(output_dir / "arch_segformer.png")
    plt.close()
    print("Saved arch_segformer.png")

if __name__ == "__main__":
    draw_pidnet()
    draw_ddrnet()
    draw_twinlite()
    draw_segformer()
