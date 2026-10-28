"""Figure 1 of the manuscript, drawn from Results_v5/all_results_v5.json (nested protocol).

Usage: python make_figure1_v5.py Results_v5/all_results_v5.json Figures_v5/Figure1.png
"""
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

R = json.load(open(sys.argv[1]))
OUT = sys.argv[2]

C_BLUE, C_GREEN, C_ORANGE, C_DARK = "#3498DB", "#27AE60", "#E67E22", "#2C3E50"
C_RED_L, C_GRAY, C_PURPLE, C_URINE = "#F5B7B1", "#95A5A6", "#9B59B6", "#F1C40F"
C_ENS, C_TABPFN, C_SVR, C_OTHER = "#1F6391", "#2E8B57", "#4A90D9", "#A9C9EC"
C_REDLINE = "#C0392B"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

# ---------------- data ----------------
rec = R["recon"]
names = {"WeightedEnsemble": "Weighted\nEnsemble", "TabPFN": "TabPFN-3.5", "BaggingSVR+": "BaggingSVR+",
         "BaggingKNN": "BaggingKNN", "ExtraTrees": "ExtraTrees", "ResidualMLP": "ResidualMLP",
         "MLP_Wide": "MLP Wide", "LightGBM": "LightGBM", "MLP_Deep": "MLP Deep", "XGBoost": "XGBoost",
         "TabNet": "TabNet", "1D_CNN": "1D-CNN", "SAN_MLP": "SAN-MLP"}
stable = [k for k, v in sorted(rec.items(), key=lambda x: -x[1]["r2_test_mean"])
          if k not in ("StackingMeta",) and v["r2_test_mean"] > 0.05]
models = [(names[k], rec[k]["r2_test_mean"], rec[k]["r2_test_std"]) for k in stable]
n_individual = len([k for k in rec if k not in ("WeightedEnsemble", "StackingMeta")])
n_stable = len([k for k in stable if k != "WeightedEnsemble"])
baseline = rec["BaggingSVR+"]["r2_test_mean"]
ens = rec["WeightedEnsemble"]["r2_test_mean"]
gain = (ens / baseline - 1) * 100
r2_stable = [rec[k]["r2_test_mean"] for k in stable if k != "WeightedEnsemble"]

endpoints = ["COVID", "GGT", "Creatinine", "HDL"]
best = {(b["endpoint"], b["input"]): b for b in R["best_nested"]}
f1 = {inp: [best[(e, inp)]["f1_mean"] for e in endpoints] for inp in ("recon", "real", "urine")}
cep = {c["endpoint"]: c for c in R["cspr_ep_nested"]}
cspr_auc = [100 * cep[e]["CSPR_AUC"] for e in endpoints]
cspr_f1 = [100 * cep[e]["CSPR_F1"] for e in endpoints]

mw = R["ensemble"]["mean_weights"]
main_w = [("TabPFN", C_TABPFN), ("BaggingSVR+", C_SVR), ("BaggingKNN", C_PURPLE),
          ("MLP_Wide", C_ORANGE), ("ResidualMLP", C_GRAY)]
weights = [(names[m].replace("\n", " "), 100 * mw[m], c) for m, c in main_w]
weights.append(("Others", 100 * sum(v for m, v in mw.items() if m not in dict(main_w)), "#D5D8DC"))

info = R["info_nested"]
ite = info["ite_per_comp"]
ite_global = info["ite_global"]
ite_mean = float(np.mean(ite))
pc1_r = R["per_component"][0]["Pearson_r"]
snr1 = info["snr_per_comp"][0]

# ---------------- figure ----------------
fig = plt.figure(figsize=(16, 11.2), facecolor="white")
gs = fig.add_gridspec(2, 3, left=0.085, right=0.975, top=0.95, bottom=0.06, wspace=0.38, hspace=0.42)

def title(ax, s):
    ax.set_title(s, fontsize=13, fontweight="bold", loc="left", pad=10)

# A
ax = fig.add_subplot(gs[0, 0])
y = np.arange(len(models))[::-1]
cols = [C_ENS, C_TABPFN, C_SVR] + [C_OTHER] * (len(models) - 3)
for yi, (name, r2, sd), c in zip(y, models, cols):
    ax.barh(yi, r2, color=c, height=0.62, zorder=3)
    ax.errorbar(r2, yi, xerr=sd, fmt="none", ecolor="#555555", elinewidth=0.9, capsize=2.5, zorder=4)
    bold = yi >= y[2]
    ax.text(r2 + sd + 0.006, yi, f"{r2:.3f}", va="center", fontsize=9,
            fontweight="bold" if bold else "normal", color=C_DARK)
ax.set_yticks(y); ax.set_yticklabels([m[0] for m in models], fontsize=9.5)
ax.axvline(baseline, color=C_REDLINE, ls="--", lw=1.2, zorder=5)
ax.text(baseline - 0.004, y[0] + 0.85, f"BaggingSVR+ baseline ({baseline:.3f})",
        color=C_REDLINE, fontsize=8.5, ha="right", va="center")
ax.annotate(f"+{gain:.1f}%", xy=(ens + 0.002, y[0] + 0.2), xytext=(0.375, y[0] + 0.85),
            fontsize=11, fontweight="bold", color=C_ENS, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="#EAF2FA", ec=C_ENS, lw=1),
            arrowprops=dict(arrowstyle="-", color=C_ENS, lw=0.9))
ax.set_xlim(0, 0.43); ax.set_ylim(-0.6, y[0] + 1.3)
ax.set_xlabel("R² (test, outer folds of nested CV)", fontsize=11)
ax.grid(axis="x", alpha=0.25, zorder=0)
title(ax, "A) Serum reconstruction performance")

# B
ax = fig.add_subplot(gs[0, 1])
x = np.arange(len(endpoints)); w = 0.26
for k, (inp, col, lab) in enumerate([("recon", C_ORANGE, "Reconstructed serum"),
                                     ("real", C_DARK, "Real serum"),
                                     ("urine", C_URINE, "Direct urine")]):
    xs = x + (k - 1) * w
    ax.bar(xs, f1[inp], w, color=col, label=lab, zorder=3)
    for xi, v in zip(xs, f1[inp]):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=7.6, fontweight="bold",
                color=col if inp != "urine" else "#9A7D0A", rotation=90, va="bottom")
ax.axhline(0.5, color=C_GRAY, ls=":", lw=1, zorder=2)
ax.text(3.6, 0.505, "Chance", fontsize=8, color=C_GRAY, ha="right", va="bottom")
ax.set_xticks(x); ax.set_xticklabels(endpoints, fontsize=10.5)
ax.set_xlim(-0.55, 3.65); ax.set_ylim(0, 1.18)
ax.set_ylabel("F1-macro (best classifier)", fontsize=11)
ax.legend(loc="upper right", fontsize=9, frameon=True, ncol=1)
ax.grid(axis="y", alpha=0.25, zorder=0)
title(ax, "B) Clinical classification performance")

# C
ax = fig.add_subplot(gs[0, 2])
ax.axhspan(75, 105, color=C_GREEN, alpha=0.07, zorder=0)
ax.text(-0.45, 104, "Good preservation (>75%)", fontsize=8, color=C_GREEN, va="top")
w = 0.34
ax.bar(x - w / 2, cspr_auc, w, color=C_BLUE, label="CSPR (AUC)", zorder=3)
ax.bar(x + w / 2, cspr_f1, w, color=C_ORANGE, label="CSPR (F1)", zorder=3)
for xi, a, b in zip(x, cspr_auc, cspr_f1):
    ax.text(xi - w / 2, a + 0.5, f"{a:.1f}", ha="center", fontsize=8.5, color=C_BLUE)
    ax.text(xi + w / 2, b + 0.5, f"{b:.1f}", ha="center", fontsize=8.5, color=C_ORANGE)
m_auc, m_f1 = np.mean(cspr_auc), np.mean(cspr_f1)
ax.axhline(m_auc, color=C_BLUE, ls="--", lw=1, alpha=0.7, zorder=2)
ax.axhline(m_f1, color=C_ORANGE, ls="--", lw=1, alpha=0.7, zorder=2)
hi, lo = (m_auc, m_f1) if m_auc >= m_f1 else (m_f1, m_auc)
ax.text(3.68, hi + 0.4, f"Mean {'AUC' if hi == m_auc else 'F1'}\n{hi:.1f}%", fontsize=8,
        color=C_BLUE if hi == m_auc else C_ORANGE, ha="left", va="bottom")
ax.text(3.68, lo - 0.4, f"Mean {'F1' if lo == m_f1 else 'AUC'}\n{lo:.1f}%", fontsize=8,
        color=C_ORANGE if lo == m_f1 else C_BLUE, ha="left", va="top")
ax.set_xticks(x); ax.set_xticklabels(endpoints, fontsize=10.5)
ax.set_ylim(50, 105); ax.set_xlim(-0.55, 4.35)
ax.set_ylabel("CSPR (%)", fontsize=11)
ax.legend(loc="upper right", fontsize=9, frameon=True, bbox_to_anchor=(0.8, 1.0))
title(ax, "C) Clinical signal preservation ratio")

# D
ax = fig.add_subplot(gs[1, 0])
vals = [v for _, v, _ in weights]
wedges, _ = ax.pie(vals, colors=[c for *_, c in weights], startangle=90, counterclock=False,
                   wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2))
small_pos = {"MLP Wide": (-0.75, 1.13), "ResidualMLP": (0.22, 1.17), "Others": (0.72, 1.0),
             "BaggingKNN": (-1.1, 0.8)}
for wdg, (name, v, _) in zip(wedges, weights):
    ang = np.deg2rad((wdg.theta2 + wdg.theta1) / 2)
    ex, ey = 0.92 * np.cos(ang), 0.92 * np.sin(ang)
    if v < 8:
        tx, ty = small_pos.get(name, (1.2 * np.cos(ang), 1.2 * np.sin(ang)))
        ax.annotate(f"{name} {v:.1f}%", xy=(ex, ey), xytext=(tx, ty), fontsize=9,
                    ha="left" if tx > 0 else "right", va="center",
                    arrowprops=dict(arrowstyle="-", color="#777777", lw=0.8))
        continue
    xx, yy = 1.17 * np.cos(ang), 1.17 * np.sin(ang)
    ax.text(xx, yy, f"{name}\n{v:.1f}%", ha="left" if xx > 0 else "right", va="center", fontsize=9.5)
ax.text(0, 0.08, f"R² = {ens:.3f}", ha="center", va="center", fontsize=15, fontweight="bold", color=C_ENS)
ax.text(0, -0.17, "Weighted\nEnsemble", ha="center", va="center", fontsize=10.5, color="#444444")
ax.set_aspect("equal")
title(ax, "D) Ensemble composition (mean weights)")

# E
ax = fig.add_subplot(gs[1, 1])
pcs = np.arange(1, len(ite) + 1)
ax.bar(pcs, ite, color=[C_GREEN if v > ite_mean else C_RED_L for v in ite], width=0.7, zorder=3)
for p, v in zip(pcs, ite):
    if v > ite_mean:
        ax.text(p, v + 0.003, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
ax.axhline(ite_global, color=C_GREEN, ls="--", lw=1, alpha=0.8, zorder=2)
ax.text(10.45, ite_global + 0.002, f"Global ITE\n(variance-weighted): {ite_global:.3f}",
        fontsize=8, color=C_GREEN, ha="right", va="bottom")
ax.axhline(ite_mean, color=C_GRAY, ls=":", lw=1, zorder=2)
ax.text(7.0, ite_mean + 0.002, f"Mean ITE: {ite_mean:.3f}", fontsize=8, color="#6C7A7D",
        ha="center", va="bottom", zorder=6)
ax.annotate(f"r = {pc1_r:.3f}\nSNR = {snr1:.2f}", xy=(1.2, ite[0]), xytext=(2.6, ite[0] - 0.012),
            fontsize=9, color=C_GREEN, fontweight="bold", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="#E9F7EF", ec=C_GREEN, lw=1),
            arrowprops=dict(arrowstyle="-", color=C_GREEN, lw=0.9))
ax.set_xticks(pcs); ax.set_xlim(0.4, 10.6); ax.set_ylim(0, 0.23)
ax.set_xlabel("Serum PCA component", fontsize=11); ax.set_ylabel("ITE", fontsize=11)
ax.grid(axis="y", alpha=0.25, zorder=0)
title(ax, "E) Information transfer efficiency")

# F
hi_ep = max(endpoints, key=lambda e: cep[e]["CSPR_AUC"])
lo_ep = min(endpoints, key=lambda e: cep[e]["CSPR_AUC"])
ax = fig.add_subplot(gs[1, 2]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0.0,rounding_size=0.01",
                            fc="white", ec="#D5D8DC", lw=1.2, transform=ax.transAxes))
rows = [
    ("RECONSTRUCTION", None, None),
    ("Best individual model (TabPFN-3.5)", f"R² = {rec['TabPFN']['r2_test_mean']:.3f}", C_TABPFN),
    ("Weighted ensemble", f"R² = {ens:.3f} (+{gain:.1f}%)", C_ENS),
    ("Individual architectures evaluated", f"{n_individual} models", C_DARK),
    (f"Convergence range ({n_stable} stable)", f"R² = {min(r2_stable):.2f} – {max(r2_stable):.2f}", C_DARK),
    ("INFORMATION TRANSFER", None, None),
    ("Channel capacity", f"{info['total_capacity_bits']:.2f} bits", C_ENS),
    ("Global ITE", f"{ite_global:.3f}", C_GREEN),
    ("PC1 Pearson r", f"{pc1_r:.3f}", C_GREEN),
    ("CLINICAL SIGNAL PRESERVATION", None, None),
    ("Mean CSPR (AUC)", f"{m_auc:.1f}%", C_BLUE),
    ("Mean CSPR (F1)", f"{m_f1:.1f}%", C_ORANGE),
    ("Highest CSPR (AUC)", f"{hi_ep}: {100 * cep[hi_ep]['CSPR_AUC']:.1f}%", C_GREEN),
    ("Lowest CSPR (AUC)", f"{lo_ep}: {100 * cep[lo_ep]['CSPR_AUC']:.1f}%", C_ORANGE),
]
yy = 0.95
for label, val, col in rows:
    if val is None:
        if yy < 0.9:
            yy -= 0.02
        ax.text(0.04, yy, label, fontsize=11, fontweight="bold", color=C_DARK, va="center")
        ax.plot([0.04, 0.96], [yy - 0.03, yy - 0.03], color="#D5D8DC", lw=0.8)
        yy -= 0.068
    else:
        ax.text(0.07, yy, label, fontsize=9.5, color="#333333", va="center")
        ax.text(0.96, yy, val, fontsize=10, fontweight="bold", color=col, ha="right", va="center")
        yy -= 0.058
title(ax, "F) Summary metrics")

fig.savefig(OUT, dpi=300, facecolor="white")
print("saved", OUT, f"gain={gain:.2f}% ite_mean={ite_mean:.4f} n_stable={n_stable}")
