"""How well do the selection objectives rank controllers on the true plant?

For each initial state, 100 parameters are drawn from the verified region C. Each
one is scored with the latent objective J_z and the decoded objective J_x of the
model (Ours-Z and Ours-X, settings from pkoopman/selection.py), and with the same
decoded objective on the true closed loop (J_true). The script reports the Spearman
rank correlation of J_z and J_x with J_true, and how close the best candidate under
each objective is to the best one under J_true.

It also runs the one-shot selection (SLSQP, as for Table 3) and checks the selected
parameter on the true plant: its true cost compared with the centroid of C, and the
worst-case 12-step prediction error of the model at that parameter.

The initial states are the first N_X0 of the 50-state benchmark draw.

    python scripts/selection_ranking.py vdp
    python scripts/selection_ranking.py mujoco_idp --n-x0 10 --n-cand 100

Writes runs/selection_ranking/<system>.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import Delaunay
from scipy.stats import spearmanr

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from pkoopman.model import load_model
from pkoopman.region import sample_in_hull
from pkoopman.selection import (FAILED_ROLLOUT_COST, SETTINGS, NumpyModel, select_theta,
                                simulator_cost, surrogate_cost)
from pkoopman.systems import experiment_system

# same boxes and seed as the 50-initial-state control benchmark
X0_BOX = {
    "vdp": [1.0, 1.0],
    "slow_manifold": [1.0, 1.0],
    "manipulator": [0.3, 0.3, 0.2, 0.2],
    "cartpole": [0.2, 0.2, 0.1, 0.2],
    "mujoco_ip": [0.1, 0.1, 0.05, 0.1],
    "mujoco_idp": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
}
PRED_STEPS = 12


def initial_states(name, n):
    b = np.array(X0_BOX[name])
    return np.random.default_rng(2029).uniform(-b, b, size=(50, len(b)))[:n]


def prediction_error(net, system, x0, theta):
    """Worst-case normalized error over PRED_STEPS steps (%), None if the true
    closed loop fails."""
    true = system.simulate(x0, theta, PRED_STEPS)
    if true is None or not np.all(np.isfinite(true)):
        return None
    pred = net.decode(net.latent_rollout(x0, theta, PRED_STEPS), theta)
    err = np.linalg.norm(pred - true[1:], axis=1) / np.maximum(np.linalg.norm(true[1:], axis=1), 0.1)
    return float(100 * err.max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("system", choices=list(X0_BOX))
    parser.add_argument("--n-x0", type=int, default=10)
    parser.add_argument("--n-cand", type=int, default=100)
    parser.add_argument("--checkpoints", default=os.path.join(ROOT, "checkpoints"))
    parser.add_argument("--out", default=os.path.join(ROOT, "runs", "selection_ranking"))
    args = parser.parse_args()

    name = args.system
    system = experiment_system(name)
    s = SETTINGS[name]
    model = load_model(os.path.join(args.checkpoints, f"{name}.pt"))
    net = NumpyModel(model)
    region = np.load(os.path.join(args.checkpoints, f"{name}_region.npy"))
    cands = sample_in_hull(Delaunay(region), region, args.n_cand, np.random.default_rng(2032))
    centroid = region.mean(axis=0)
    mujoco = name.startswith("mujoco")

    records = []
    t0 = time.perf_counter()
    for i, x0 in enumerate(initial_states(name, args.n_x0)):
        x0 = system.reset(x0) if mujoco else x0
        jz = np.array([surrogate_cost(net, system, x0, th, "z", s) for th in cands])
        jx = np.array([surrogate_cost(net, system, x0, th, "x", s) for th in cands])
        jt = np.array([simulator_cost(system, x0, th, s) for th in cands])
        ok = jt < FAILED_ROLLOUT_COST
        rec = dict(x0=np.asarray(x0).tolist(), n_failed_true=int((~ok).sum()))
        if ok.sum() >= 3:
            best = jt[ok].min()
            for tag, j in (("z", jz), ("x", jx)):
                rec[f"spearman_{tag}"] = float(spearmanr(j[ok], jt[ok]).correlation)
                # true cost of the candidate the surrogate ranks first, relative to the best
                rec[f"regret_{tag}"] = float(jt[ok][np.argmin(j[ok])] / best - 1.0)
        j_centroid = simulator_cost(system, x0, centroid, s)
        rec["j_true_centroid"] = float(j_centroid)
        rec["j_true_best_candidate"] = float(jt[ok].min()) if ok.any() else None
        for tag in ("z", "x"):
            theta, info = select_theta(model, region, x0, system, objective=tag, settings=s)
            rec[f"selected_{tag}"] = dict(theta=np.asarray(theta).tolist(), t_select=info["t_select"],
                                          slsqp_success=info["success"],
                                          j_true=float(simulator_cost(system, x0, theta, s)),
                                          pred_error=prediction_error(net, system, x0, theta))
        rec["pred_error_centroid"] = prediction_error(net, system, x0, centroid)
        records.append(rec)
        print(f"[{name}] x0 {i}: rho_z {rec.get('spearman_z', float('nan')):+.3f}  "
              f"rho_x {rec.get('spearman_x', float('nan')):+.3f}  "
              f"J sel_x {rec['selected_x']['j_true']:.4g}  J centroid {j_centroid:.4g}  "
              f"pred err sel_x {rec['selected_x']['pred_error']}  ({time.perf_counter() - t0:.0f} s)",
              flush=True)

    def stat(key):
        v = [r[key] for r in records if r.get(key) is not None]
        return dict(mean=float(np.mean(v)), median=float(np.median(v)), min=float(np.min(v))) if v else None

    summary = dict(spearman_z=stat("spearman_z"), spearman_x=stat("spearman_x"),
                   regret_z=stat("regret_z"), regret_x=stat("regret_x"))
    for tag in ("z", "x"):
        rel = [r[f"selected_{tag}"]["j_true"] / r["j_true_centroid"] - 1.0 for r in records
               if r["j_true_centroid"] < FAILED_ROLLOUT_COST]
        pe = [r[f"selected_{tag}"]["pred_error"] for r in records if r[f"selected_{tag}"]["pred_error"] is not None]
        summary[f"selected_{tag}_vs_centroid"] = dict(mean=float(np.mean(rel)), median=float(np.median(rel))) if rel else None
        summary[f"selected_{tag}_pred_error"] = dict(mean=float(np.mean(pe)), max=float(np.max(pe))) if pe else None
    pc = [r["pred_error_centroid"] for r in records if r["pred_error_centroid"] is not None]
    summary["centroid_pred_error"] = dict(mean=float(np.mean(pc)), max=float(np.max(pc))) if pc else None

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{name}.json"), "w") as f:
        json.dump(dict(system=name, n_x0=args.n_x0, n_cand=args.n_cand, settings=str(s),
                       summary=summary, records=records), f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
