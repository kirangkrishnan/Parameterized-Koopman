"""Fig. 7: prediction error of all methods, from the json files of prediction_benchmark.py.

Top row: worst-case error over 12 steps (bar = mean, whisker = one std, black
tick = median). Bottom row: mean error at step k, k = 1..20.

    python scripts/plot_prediction.py
    python scripts/plot_prediction.py --systems vdp cartpole --out fig7.png
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
METHODS = ["Ours", "PKNN", "EDMD", "Deep Koopman", "Kernel-EDMD"]
COLORS = {"Ours": "#3a9e5c", "PKNN": "#35a9c4", "EDMD": "#7d5ba6",
          "Deep Koopman": "#e8892b", "Kernel-EDMD": "#8c6151"}
TITLES = {"vdp": "Van der Pol", "slow_manifold": "Slow manifold (u in x1)",
          "slow_manifold_u2": "Slow manifold", "manipulator": "Manipulator", "cartpole": "Cartpole"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", nargs="+", default=["vdp", "slow_manifold_u2", "manipulator", "cartpole"])
    parser.add_argument("--runs", default=os.path.join(ROOT, "runs", "prediction_benchmark"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    ncol = len(args.systems)
    fig, axes = plt.subplots(2, ncol, figsize=(2.0 * ncol, 4.2), squeeze=False)
    for j, name in enumerate(args.systems):
        with open(os.path.join(args.runs, f"{name}.json")) as f:
            res = json.load(f)
        top, bottom = axes[0, j], axes[1, j]
        H = str(res["train_horizon"])

        for i, m in enumerate(METHODS):
            s = res["methods"][m]["worst_case"][H]
            top.bar(i, s["mean"], 0.7, color=COLORS[m])
            top.errorbar(i, s["mean"], yerr=[[min(s["std"], s["mean"])], [s["std"]]],
                         fmt="none", ecolor="black", elinewidth=0.7, capsize=2)
            top.plot([i - 0.25, i + 0.25], [s["median"]] * 2, color="black", lw=1.2)

            curve = res["methods"][m]["mean_step_error"]
            k = np.arange(1, len(curve) + 1)
            bottom.plot(k, curve, color=COLORS[m], lw=1.2, marker="o", ms=2,
                        ls="--" if m == "PKNN" else "-", label=m)

        bottom.axhline(15, color="red", lw=0.8, ls="--", label="15% threshold")
        bottom.axvline(res["train_horizon"], color="black", lw=0.6, ls=":")
        top.set_xticks([])
        top.set_title(TITLES.get(name, name), fontsize=9)
        bottom.set_xlabel("prediction step k")
        bottom.set_xlim(1, len(curve))
        # the 4-d systems have baselines in the hundreds of percent
        for ax in (top, bottom):
            if max(ax.get_ylim()) > 150:
                ax.set_yscale("symlog", linthresh=15)
            ax.set_ylim(bottom=0)

    axes[0, 0].set_ylabel(f"{res['train_horizon']}-step error (%)")
    axes[1, 0].set_ylabel("error at step k (%)")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out = args.out or os.path.join(args.runs, "prediction_error.pdf")
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
