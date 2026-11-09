"""Figures for the two cartpole ablations, from the json files written by
ablation_retraining.py and ablation_dictionary.py.

    python scripts/plot_ablations.py
    python scripts/plot_ablations.py --retraining runs/ablation_retraining --dictionary runs/ablation_dictionary/results.json

Error bars are one standard deviation over the training seeds. All test
parameters are shown.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GREEN, RED, BLUE = "#3a9e5c", "#d1484f", "#3f7fb5"

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})


def by_alpha(records, n_alpha):
    """(n_alpha, n_seeds) array of mean errors."""
    return np.array([[r["mean_err"] for r in records if r["alpha_idx"] == i] for i in range(n_alpha)])


def grouped_bars(ax, a, b, colors, labels, log=False):
    x = np.arange(len(a))
    w = 0.38
    for off, errs, c, lab in [(-w / 2, a, colors[0], labels[0]), (w / 2, b, colors[1], labels[1])]:
        mean, std = errs.mean(axis=1), errs.std(axis=1)
        lower = np.minimum(std, 0.5 * mean) if log else std   # keep bars visible on a log axis
        ax.bar(x + off, mean, w, color=c, label=lab, edgecolor="none")
        ax.errorbar(x + off, mean, yerr=[lower, std], fmt="none", ecolor="black",
                    elinewidth=0.6, capsize=1.5, capthick=0.6)
    if log:
        ax.set_yscale("log")
    ax.set_xticks(x)
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
              handlelength=1, borderaxespad=0.1, columnspacing=1.0)


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    print("wrote", os.path.join(out, name + ".pdf"))


def plot_retraining(paths, out):
    runs = [json.load(open(p)) for p in sorted(paths)]
    n_alpha = len(runs[0]["alphas"])
    cond = by_alpha([r for run in runs for r in run["conditioned"]], n_alpha)
    uncond = by_alpha([r for run in runs for r in run["unconditioned"]], n_alpha)
    retrain = np.mean([r["data_time"] + r["train_time"] for run in runs for r in run["unconditioned"]])
    query = np.mean([r["query_time"] for run in runs for r in run["conditioned"]])
    print(f"ablation A, {len(runs)} seeds: retraining {retrain:.0f} s per parameter, K(theta) query {query:.2e} s")
    print("  conditioned  ", np.round(cond.mean(axis=1), 3), f"mean {cond.mean():.3f}")
    print("  per-parameter", np.round(uncond.mean(axis=1), 3), f"mean {uncond.mean():.3f}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.5, 1.55), gridspec_kw=dict(width_ratios=[1.0, 1.6], wspace=0.42))
    a1.bar([0, 1], [retrain, query], 0.6, color=[RED, GREEN], edgecolor="none")
    a1.set_yscale("log")
    for x, v in [(0, retrain), (1, query)]:
        a1.text(x, v * 2.2, f"{v:.4g} s", ha="center", va="bottom", fontsize=7)
    a1.set_ylim(min(query, 1e-4) / 3, retrain * 50)
    a1.set_xlim(-0.6, 1.6)
    a1.set_xticks([0, 1])
    a1.set_xticklabels(["Retrain\nper $\\theta$", "Conditioned\nmodel"], fontsize=6.5)
    a1.set_ylabel("Time for a new $\\theta$ (s)")
    a1.minorticks_off()

    grouped_bars(a2, cond, uncond, (GREEN, RED), ("Conditioned (one model)", "Per-parameter models"))
    a2.set_xticklabels([f"$\\theta^{{({i + 1})}}$" for i in range(n_alpha)])
    a2.set_ylabel("Worst-case error (%)")
    save(fig, out, "ablation_retraining")


def plot_dictionary(path, out):
    res = json.load(open(path))
    n_alpha = len(res["alphas"])
    cond = by_alpha(res["conditioned"], n_alpha)
    fixed = by_alpha(res["fixed_dictionary"], n_alpha)
    wins = cond.mean(axis=1) < fixed.mean(axis=1)
    print(f"ablation B, {cond.shape[1]} seeds, {n_alpha} parameters: conditioned {cond.mean():.3f}%, "
          f"fixed dictionary {fixed.mean():.3f}%, conditioned lower at {wins.sum()}/{n_alpha}")
    print(f"  seed-to-seed std (mean over parameters): {cond.std(axis=1).mean():.3f} vs {fixed.std(axis=1).mean():.3f}")

    fig, ax = plt.subplots(figsize=(7.0, 1.7))
    grouped_bars(ax, cond, fixed, (GREEN, BLUE),
                 ("Conditioned dictionary $\\phi_\\theta$", "Fixed dictionary $\\phi$"), log=True)
    ax.set_xticklabels([str(i) for i in range(n_alpha)])
    ax.set_xlim(-0.6, n_alpha - 0.4)
    ax.axvline(res["n_anchors"] - 0.5, color="0.6", lw=0.5, ls="--")
    ax.set_xlabel(f"Index $i$ of the test parameter $\\theta^{{(i)}}$ "
                  f"(0-{res['n_anchors'] - 1} hand-picked, rest Latin hypercube)")
    ax.set_ylabel("Worst-case error (%)")
    save(fig, out, "ablation_dictionary_accuracy")

    lat = res["latent"]
    pcs = np.array(lat["conditioned"]["pca"]) * 1e3
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    ax.scatter(pcs[:, 0], pcs[:, 1], s=14, color=GREEN, edgecolor="black", linewidth=0.4, zorder=3)
    for i, (x, y) in enumerate(pcs):
        ax.annotate(str(i), (x, y), xytext=(2.5, 2.5), textcoords="offset points", fontsize=6)
    ax.set_xlabel("PC1 ($\\times10^{-3}$)")
    ax.set_ylabel("PC2 ($\\times10^{-3}$)")
    ax.set_title(f"spread {lat['conditioned']['spread_pca']:.3e} (fixed dictionary: "
                 f"{lat['fixed_dictionary']['spread_raw']:g}), seed {lat['seed']}", fontsize=7)
    ax.grid(lw=0.3, alpha=0.5)
    ax.set_axisbelow(True)
    save(fig, out, "ablation_dictionary_latent")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retraining", default="runs/ablation_retraining", help="directory with seed_*.json")
    parser.add_argument("--dictionary", default="runs/ablation_dictionary/results.json")
    parser.add_argument("--out", default="runs/plot_ablations")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    paths = glob.glob(os.path.join(args.retraining, "seed_*.json"))
    if paths:
        plot_retraining(paths, args.out)
    if os.path.exists(args.dictionary):
        plot_dictionary(args.dictionary, args.out)


if __name__ == "__main__":
    main()
