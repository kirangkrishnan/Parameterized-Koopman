"""Plot how the verified region shrinks (Fig. 3 style) from the files written by train.py.

    python scripts/plot_region.py runs/train/vdp_region_history.npz
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("history")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    data = np.load(args.history)
    explored = data["explored"]
    stages = [data[f"arr_{i}"] for i in range(len(data.files) - 1)]
    final = stages[-1]

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    for ax, (i, j) in zip(axes, [(0, 1), (2, 3)]):
        ax.scatter(explored[:, i], explored[:, j], s=4, c="0.85", label="explored")
        ax.scatter(stages[0][:, i], stages[0][:, j], s=4, c="tab:orange", label="passed (first test)")
        ax.scatter(final[:, i], final[:, j], s=4, c="tab:green", label="converged")
        ax.set_xlabel(rf"$\theta_{i + 1}$")
        ax.set_ylabel(rf"$\theta_{j + 1}$")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    out = args.out or args.history.replace("_history.npz", ".pdf")
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
