"""
sim_data_analysis.py - graphs and statistical summary for the simulation study.
Reads the summary csv of the restricted run. Runs in a few seconds.

Usage:
    python sim_data_analysis.py

Writes into figures/ :
    fig_mse.png, fig_cp.png, fig_ratio.png
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(BASE_DIR, "results", "restricted", "summary_all.csv")
DEFAULT_OUT = os.path.join(BASE_DIR, "figures")

CSV = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
OUT = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
B = 1000                                          # replications per setting (for the coverage band)
os.makedirs(OUT, exist_ok=True)
S = pd.read_csv(CSV)

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e3e2de"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False})

SIZES = ["n30m20", "n50m30"]
PLANS = ["last", "even", "first"]
CENS = ["moderate", "heavy"]
CONF = ["ordered", "equal", "reversed"]
CONF_T = {"ordered": r"true $(\theta_1,\theta_2)=(1,2)$", "equal": "true $(1.5,1.5)$", "reversed": "true $(2,1)$"}
PARS = [("t1", r"$\theta_1$"), ("t2", r"$\theta_2$"), ("lam", r"$\lambda$")]
SET = [(s, p, c) for s in SIZES for p in PLANS for c in CENS]       # the 12 points on every line
XL = [f"{p[0].upper()}{c[0].upper()}" for _, p, c in SET]            # e.g. LM = Last, Moderate
# method -> (label, colour, linestyle, marker, filled)
M = {"mle": ("MLE", BLUE, "-", "o", True), "ormle": ("ORMLE", BLUE, "--", "o", False),
     "bayes_vague": ("BE", ORANGE, "-", "s", True), "orbe_vague": ("OR-BE", ORANGE, "--", "s", False)}


def series(method, conf, par, col):
    """one value per setting, in the order of SET"""
    d = S[(S.method == method) & (S.config == conf) & (S.par == par)].set_index(["size", "plan", "censoring"])[col]
    return np.array([d[k] for k in SET])


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------- figures
def grid_fig(col, methods, name, ylabel, ylog, band=None):
    """3 x 3 panels: rows = parameter, columns = true shapes; one line per method over the 12 settings"""
    fig, axs = plt.subplots(3, 3, figsize=(7.2, 6.2), sharex=True, sharey=False if band else "row")
    for i, (par, pn) in enumerate(PARS):
        for j, conf in enumerate(CONF):
            ax = axs[i, j]
            for m in methods:
                lab, c, ls, mk, filled = M[m]
                ax.plot(range(len(SET)), series(m, conf, par, col), color=c, ls=ls, lw=1.4, marker=mk, ms=4.5,
                        mfc=c if filled else "white", mec=c, mew=1.2, label=lab)
            if band:                                           # Monte Carlo band around the nominal 0.95
                ax.axhspan(band[0], band[1], color=GRID, alpha=0.8, lw=0)
                ax.axhline(0.95, color=MUTED, lw=0.8, ls=":")
            ax.set_xticks(range(len(SET)))
            ax.set_xticklabels(XL, fontsize=6.5)
            ax.axvline(len(SET) / 2 - 0.5, color=MUTED, lw=0.8)    # separates n = 30 from n = 50
            if ylog:
                ax.set_yscale("log")
            if i == 0:
                ax.set_title(CONF_T[conf], fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{ylabel} of {pn}")
            if i == 2:
                ax.set_xlabel("plan / censoring")
            for x, t in ((0.25, "n = 30"), (0.75, "n = 50")):
                ax.text(x, 0.02, t, transform=ax.transAxes, fontsize=6.5, color=MUTED, va="bottom", ha="center")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(methods), bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, name)


def ratio_fig(name):
    """MSE of the restricted over the unrestricted estimator, one dot per setting, bar = median"""
    pairs = [("ormle", "mle", BLUE, "o", "ORMLE / MLE"), ("orbe_vague", "bayes_vague", ORANGE, "s", "OR-BE / BE")]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
    for ax, (par, pn) in zip(axs, PARS):
        for ci, conf in enumerate(CONF):
            for k, (num, den, c, mk, lab) in enumerate(pairs):
                r = series(num, conf, par, "mse") / series(den, conf, par, "mse")
                x0 = ci + (k - 0.5) * 0.3
                ax.scatter(x0 + np.linspace(-0.07, 0.07, len(r)), r, s=14, color=c, marker=mk, alpha=0.85,
                           edgecolor="white", linewidth=0.4, label=lab if ci == 0 else None)
                ax.hlines(np.median(r), x0 - 0.12, x0 + 0.12, color=INK, lw=1.2)
        ax.axhline(1, color=MUTED, lw=1)
        ax.set_yscale("log")
        ax.set_yticks([0.3, 0.5, 1, 2, 3])
        ax.set_yticklabels(["0.3", "0.5", "1", "2", "3"])
        ax.set_xticks(range(3))
        ax.set_xticklabels(["ordered\n(1, 2)", "equal\n(1.5, 1.5)", "reversed\n(2, 1)"], fontsize=7)
        ax.set_title(pn)
    axs[0].set_ylabel("MSE ratio (restricted / unrestricted)")
    axs[1].legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    save(fig, name)


if __name__ == "__main__":
    band = [0.95 - 1.96 * np.sqrt(0.95 * 0.05 / B), 0.95 + 1.96 * np.sqrt(0.95 * 0.05 / B)]
    grid_fig("mse", ["mle", "ormle", "bayes_vague", "orbe_vague"], "fig_mse", "MSE", True)
    grid_fig("CP", ["mle", "bayes_vague", "orbe_vague"], "fig_cp", "CP", False, band=band)
    ratio_fig("fig_ratio")
    print(f"Generated figures in '{OUT}': fig_mse.png, fig_cp.png, fig_ratio.png")
