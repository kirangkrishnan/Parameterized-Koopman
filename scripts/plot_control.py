"""Fig. 7 style summary of the control comparison: ||x(t)|| per system (top row)
and online computation t_tot averaged over the available seeds (bottom row, log scale).

    python scripts/plot_control.py
    python scripts/plot_control.py --systems vdp cartpole --seed 8
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = {"vdp": "Van der Pol", "slow_manifold": "Slow manifold", "manipulator": "Manipulator",
           "cartpole": "Cartpole", "mujoco_ip": "MuJoCo IP", "mujoco_idp": "MuJoCo IDP"}
# same colours as the paper figures
METHODS = {"ours_z": ("Ours-Z", "#86c995", "--"), "ours_x": ("Ours-X", "#3a9e5c", "-"),
           "nmpc": ("NMPC", "#3f7fb5", "-"), "kmpc_deep": ("K-MPC (Deep)", "#e8892b", "-"),
           "kmpc_edmd": ("K-MPC (EDMD)", "#7d5ba6", "-")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=str(ROOT / "runs" / "control_comparison"))
    parser.add_argument("--systems", nargs="+", default=list(SYSTEMS))
    parser.add_argument("--seed", type=int, default=8, help="seed whose trajectory is shown")
    parser.add_argument("--out", default=str(ROOT / "runs" / "plot_control" / "control_comparison.pdf"))
    args = parser.parse_args()
    runs = Path(args.runs)

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False})
    ncol = len(args.systems)
    fig, axes = plt.subplots(2, ncol, figsize=(2.3 * ncol, 4.2), squeeze=False)
    for j, name in enumerate(args.systems):
        top, bottom = axes[0, j], axes[1, j]
        labels, times, colors = [], [], []
        for method, (label, color, ls) in METHODS.items():
            traj_file = runs / f"{name}_{method}_seed{args.seed}.npz"
            if traj_file.exists():
                traj = np.load(traj_file)["traj"]
                dt = json.loads(traj_file.with_suffix(".json").read_text())["dt"]
                t = dt * np.arange(len(traj))
                top.plot(t, np.linalg.norm(traj, axis=1), color=color, ls=ls, lw=1.0, label=label)
            t_tot = [json.loads(f.read_text())["t_tot"] for f in runs.glob(f"{name}_{method}_seed*.json")]
            if t_tot:
                labels.append(label)
                times.append(np.mean(t_tot))
                colors.append(color)
        top.set_title(SYSTEMS[name])
        top.set_xlabel("time (s)")
        bottom.bar(range(len(times)), times, color=colors, width=0.7)
        bottom.set_yscale("log")
        bottom.set_ylim(bottom=5e-3)  # common baseline so bar heights compare across panels
        bottom.set_xticks(range(len(times)))
        bottom.set_xticklabels(labels, rotation=60, ha="right", fontsize=6.5)
        bottom.grid(axis="y", lw=0.3, color="#c8c8c8")
        bottom.set_axisbelow(True)
        if j == 0:
            top.set_ylabel(r"$\|x(t)\|$")
            bottom.set_ylabel(r"$t_{\rm tot}$ (s)")
    handles, names = [], []
    for ax in axes[0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in names:
                handles.append(h)
                names.append(l)
    fig.legend(handles, names, loc="upper center", ncol=len(names), bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
