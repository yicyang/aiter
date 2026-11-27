"""Fusion / selection diagrams for the three gfx942 GDN PRs (PNG, light background)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = "/root/workspace/review_gdn/assets"
INK, MUTED = "#1f2328", "#57606a"
KERNEL, FUSED, DATA, HBM = "#ddf4ff", "#dafbe1", "#f6f8fa", "#fff1e5"
EDGE_K, EDGE_F, EDGE_D, EDGE_H = "#0969da", "#1a7f37", "#8c959f", "#bc4c00"


def box(ax, x, y, w, h, text, fill, edge, size=9.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fill, ec=edge, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            color=INK, weight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, label=None, color=MUTED, dy=0.12):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                                 lw=1.3, color=color))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, label, ha="center", va="bottom",
                fontsize=8, color=color)


def lane_title(ax, x, y, text):
    ax.text(x, y, text, ha="left", va="center", fontsize=11, weight="bold", color=INK)


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    ax.set_xlim(0, w * 2)
    ax.set_ylim(0, h * 2)
    ax.axis("off")
    return fig, ax


# ---------------------------------------------------------------- PR-B
def pr_b():
    fig, ax = canvas(13, 8.6)
    # ---- decode
    lane_title(ax, 0.3, 16.6, "Decode (1 token per request) - Qwen3.8 TP2, batch 8, fp32 state")
    ax.text(0.3, 15.9, "Before: 2 launches", fontsize=10, color=MUTED)
    box(ax, 0.3, 13.6, 3.2, 1.7, "in_proj rows\nqkvz = [q|k|v|z]\nba = [b|a]", DATA, EDGE_D, 9)
    box(ax, 4.6, 13.6, 4.6, 1.7, "causal_conv1d_update\n_sglang_flydsl\nconv(4) + SiLU", KERNEL, EDGE_K)
    box(ax, 10.3, 13.6, 3.0, 1.7, "q, k, v\nconv output\n(HBM)", HBM, EDGE_H, 9)
    box(ax, 14.4, 13.6, 5.2, 1.7, "flydsl_gdr_decode\ngating + q/k L2 norm\n+ delta-rule recurrence", KERNEL, EDGE_K)
    box(ax, 20.7, 13.6, 4.6, 1.7, "o  [B, 1, HV, V]\nSSM state updated\nz: separate copy", DATA, EDGE_D, 9)
    for x0, x1 in ((3.5, 4.6), (9.2, 10.3), (13.3, 14.4), (19.6, 20.7)):
        arrow(ax, x0, 14.45, x1, 14.45)
    ax.text(11.8, 13.2, "write + read back", ha="center", fontsize=8, color=EDGE_H)
    ax.text(25.6, 14.45, "32.2 us", ha="left", va="center", fontsize=10, color=INK, weight="bold")

    ax.text(0.3, 12.2, "After: 1 launch", fontsize=10, color=MUTED)
    box(ax, 0.3, 9.9, 3.2, 1.7, "in_proj rows\nqkvz = [q|k|v|z]\nba = [b|a]", DATA, EDGE_D, 9)
    box(ax, 4.6, 9.4, 15.0, 2.7,
        "fused_qkvzba_conv1d_gated_delta_rule_decode_flydsl\n"
        "(one workgroup per request x v-head)\n"
        "unpack -> conv(4) + SiLU in registers -> q/k/v via LDS\n"
        "-> gating + q/k L2 norm -> recurrence (fp32 state, nt)\n"
        "conv state shifted in place; shared q/k channels: last reader writes",
        FUSED, EDGE_F, 8.8)
    box(ax, 20.7, 9.9, 4.6, 1.7, "o  [B, 1, HV, V]\nSSM state updated\ncontiguous z", DATA, EDGE_D, 9)
    arrow(ax, 3.5, 10.75, 4.6, 10.75)
    arrow(ax, 19.6, 10.75, 20.7, 10.75)
    ax.text(25.6, 10.75, "21.2 us\n(-34%)", ha="left", va="center", fontsize=10, color=EDGE_F, weight="bold")

    ax.plot([0.3, 27.5], [8.6, 8.6], color="#d0d7de", lw=1)
    # ---- verify
    lane_title(ax, 0.3, 7.9, "MTP target verify (T = 4 draft tokens, chain) - Qwen3.8 TP2, batch 8")
    ax.text(0.3, 7.2, "Before: 2 launches", fontsize=10, color=MUTED)
    box(ax, 0.3, 4.9, 3.2, 1.7, "in_proj rows\n[B*T, ...]", DATA, EDGE_D, 9)
    box(ax, 4.6, 4.9, 4.6, 1.7, "causal_conv1d_update\n_sglang_flydsl\n+ T conv windows", KERNEL, EDGE_K)
    box(ax, 10.3, 4.9, 3.0, 1.7, "q, k, v of\nT steps\n(HBM)", HBM, EDGE_H, 9)
    box(ax, 14.4, 4.9, 5.2, 1.7, "flydsl_gdr_mtp_sglang\n(disable_state_update)\nT steps, snapshot per step", KERNEL, EDGE_K)
    box(ax, 20.7, 4.9, 4.6, 1.7, "o of T steps\nintermediate states\n+ conv windows", DATA, EDGE_D, 9)
    for x0, x1 in ((3.5, 4.6), (9.2, 10.3), (13.3, 14.4), (19.6, 20.7)):
        arrow(ax, x0, 5.75, x1, 5.75)
    ax.text(25.6, 5.75, "48.4 us", ha="left", va="center", fontsize=10, color=INK, weight="bold")

    ax.text(0.3, 3.5, "After: 1 launch", fontsize=10, color=MUTED)
    box(ax, 0.3, 1.2, 3.2, 1.7, "in_proj rows\n[B*T, ...]", DATA, EDGE_D, 9)
    box(ax, 4.6, 0.8, 15.0, 2.5,
        "fused_qkvzba_conv1d_gated_delta_rule_verify_flydsl\n"
        "conv(4) + SiLU of all T tokens -> per step: gating + L2 norm + recurrence\n"
        "SSM state read once for all T steps (not written back)\n"
        "per-step state + conv window -> intermediate caches",
        FUSED, EDGE_F, 8.8)
    box(ax, 20.7, 1.2, 4.6, 1.7, "o of T steps\nintermediate states\n+ conv windows", DATA, EDGE_D, 9)
    arrow(ax, 3.5, 2.05, 4.6, 2.05)
    arrow(ax, 19.6, 2.05, 20.7, 2.05)
    ax.text(25.6, 2.05, "43.4 us\n(-10%)", ha="left", va="center", fontsize=10, color=EDGE_F, weight="bold")
    ax.text(0.3, 0.05, "MI308X (gfx942, 80 CU), cold L2/MALL, median kernel time. gfx942 + Qwen3.8 heads (16/48, 8/24, 4/12) only; "
            "other archs and models keep their path.", fontsize=8, color=MUTED)
    fig.savefig(f"{OUT}/pr-b-gdn-decode-verify-fusion.png", bbox_inches="tight", facecolor="white")


# ---------------------------------------------------------------- PR-C
def pr_c():
    fig, ax = canvas(13, 4.9)
    lane_title(ax, 0.3, 9.2, "GDN prefill front - Qwen3.8 TP2, 1 x 12000 tokens")
    ax.text(0.3, 8.5, "Before: 3 launches", fontsize=10, color=MUTED)
    box(ax, 0.3, 6.3, 3.0, 1.6, "in_proj rows\nqkvz [T, q|k|v|z]\n(channel-last)", DATA, EDGE_D, 9)
    box(ax, 4.2, 6.3, 4.6, 1.6, "causal_conv1d_split_qkv\n_hip_fn\nconv(4) + SiLU + split", KERNEL, EDGE_K)
    box(ax, 9.7, 6.3, 2.6, 1.6, "q, k, v\n(HBM)", HBM, EDGE_H, 9)
    box(ax, 13.2, 6.3, 3.4, 1.6, "l2norm_fwd(q)\nread + write q", KERNEL, EDGE_K)
    box(ax, 17.5, 6.3, 3.4, 1.6, "l2norm_fwd(k)\nread + write k", KERNEL, EDGE_K)
    box(ax, 21.8, 6.3, 3.4, 1.6, "chunked\ndelta rule", DATA, EDGE_D, 9)
    for x0, x1 in ((3.3, 4.2), (8.8, 9.7), (12.3, 13.2), (16.6, 17.5), (20.9, 21.8)):
        arrow(ax, x0, 7.1, x1, 7.1)
    ax.text(25.6, 7.1, "321.9 us", ha="left", va="center", fontsize=10, color=INK, weight="bold")
    ax.text(10.9, 5.9, "q, k written and read back twice", ha="center", fontsize=8, color=EDGE_H)

    ax.text(0.3, 4.6, "After: 1 launch", fontsize=10, color=MUTED)
    box(ax, 0.3, 2.3, 3.0, 1.6, "in_proj rows\nqkvz [T, q|k|v|z]\n(channel-last)", DATA, EDGE_D, 9)
    box(ax, 4.2, 2.1, 16.7, 2.0,
        "flydsl_gdn_prefill_front  (one workgroup per head x 128-token chunk x sequence)\n"
        "16-byte row loads + 3-token halo in registers  ->  conv(4) + SiLU  ->  split\n"
        "->  q/k L2 norm with a 16-lane DPP all-reduce (no LDS)  ->  q, k, v  |  conv state written in place",
        FUSED, EDGE_F, 9)
    box(ax, 21.8, 2.3, 3.4, 1.6, "chunked\ndelta rule\n(l2norm off)", DATA, EDGE_D, 9)
    arrow(ax, 3.3, 3.1, 4.2, 3.1)
    arrow(ax, 20.9, 3.1, 21.8, 3.1)
    ax.text(25.6, 3.1, "255.3 us\n(-21%)", ha="left", va="center", fontsize=10, color=EDGE_F, weight="bold")
    ax.text(0.3, 0.6, "MI308X (gfx942, 80 CU), cold L2/MALL, median kernel time. gfx942 + Qwen3.8 TP1/TP2 heads (16/48, 8/24) only; "
            "TP4, other archs and models keep their path.", fontsize=8, color=MUTED)
    fig.savefig(f"{OUT}/pr-c-gdn-prefill-front-fusion.png", bbox_inches="tight", facecolor="white")


# ---------------------------------------------------------------- PR-A
def pr_a():
    fig, ax = canvas(12, 4.4)
    lane_title(ax, 0.3, 8.3, "chunk_fwd_kernel_o_opt_vk: which tile runs (autotuning off, the default)")
    box(ax, 0.3, 5.2, 4.2, 1.8, "launch\nchunk_fwd_kernel\n_o_opt_vk", DATA, EDGE_D, 9.5)
    box(ax, 5.6, 5.2, 6.0, 1.8, "this device has\ngated_delta_rule \"dispatch\" rules?\n(only configs/gfx942/80cu/)", KERNEL, EDGE_K, 9)
    box(ax, 12.9, 5.2, 5.8, 1.8, "(H, Hg, K, V, BT) equals a\nQwen3.8 rule exactly?\n48/16, 24/8, 12/4, 6/2; 128; 64", KERNEL, EDGE_K, 9)
    box(ax, 20.0, 5.2, 3.8, 1.8, "BK=64, BV=128\n4 warps, 1 stage", FUSED, EDGE_F, 9.5, bold=True)
    box(ax, 9.0, 1.4, 6.4, 1.8, "first candidate (unchanged default)\nBK=32, BV=32, 2 warps, 2 stages", DATA, EDGE_D, 9.5)
    arrow(ax, 4.5, 6.1, 5.6, 6.1)
    arrow(ax, 11.6, 6.1, 12.9, 6.1, "yes", EDGE_F)
    arrow(ax, 18.7, 6.1, 20.0, 6.1, "yes", EDGE_F)
    arrow(ax, 8.6, 5.2, 10.8, 3.2)
    arrow(ax, 15.8, 5.2, 13.6, 3.2)
    ax.text(7.2, 3.9, "no: one config,\nno prune", ha="center", fontsize=8, color=MUTED)
    ax.text(17.2, 3.9, "no\n(e.g. Qwen3-Next)", ha="center", fontsize=8, color=MUTED)
    ax.text(20.0, 3.4, "Qwen3.8 on MI308X:\n731 -> 350 us (TP2, 12000 tok),\nbit-identical output", fontsize=8.5, color=EDGE_F)
    ax.text(0.3, 0.3, "MI300X (304 CU), gfx950 and other parts read no rules; every other model's shape misses the rules. "
            "GATED_DELTA_RULE_TRITON_AUTOTUNE=1 still tunes all candidates.", fontsize=8, color=MUTED)
    fig.savefig(f"{OUT}/pr-a-chunk-o-tile-selection.png", bbox_inches="tight", facecolor="white")


pr_a()
pr_b()
pr_c()
print("ok")
