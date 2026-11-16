"""Check a trained model on fresh parameters from its verified region (Phase 3).

New parameters are drawn uniformly from the convex hull of checkpoints/<system>_region.npy
with a seed that was not used anywhere in training. For each one the true closed
loop is simulated from the verification initial conditions of the system and the
worst-case prediction error is recorded at several horizons. The summary
(collect_results.py) turns this into survival rates at several tolerances with
Clopper-Pearson intervals. The fraction of the box Theta covered by the hull is
estimated by Monte Carlo.

    python scripts/region_check.py vdp
    python scripts/region_check.py mujoco_idp --chunk 3 --n-chunks 20     # one piece of a cluster array
    python scripts/region_check.py cartpole --checkpoints runs/train_tol10 --tag cartpole_tol10

Each run writes runs/region_check/<tag>/chunk<k>of<n>.npz (the tag defaults to the system name).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from scipy.spatial import Delaunay

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pkoopman.model import load_model
from pkoopman.region import sample_in_hull, simulate_all
from pkoopman.systems import experiment_system

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SEED = 2027          # the reported checks of the Phase 3 regions use --seed 2031


def errors_at_horizons(model, system, thetas, x0s, horizons):
    """Worst-case normalized error (%) over all x0s at every horizon in horizons,
    one row per theta, and whether the true closed loop passed system.is_valid().
    A failed simulation gives an infinite error."""
    steps = max(horizons)
    all_trajs, valid = simulate_all(system, thetas, x0s, steps)
    errors = np.full((len(thetas), len(horizons)), np.inf)
    x0_t = torch.tensor(np.asarray(x0s), dtype=torch.float32)
    for row, (theta, trajs) in enumerate(zip(thetas, all_trajs)):
        if any(tr is None for tr in trajs):
            continue
        th = torch.tensor(theta, dtype=torch.float32).expand(len(x0s), -1)
        with torch.no_grad():
            pred = model.predict(x0_t, th, steps).numpy()
        true = np.stack(trajs)[:, 1:]
        scale = np.maximum(np.linalg.norm(true, axis=-1), 0.1)
        err = 100.0 * np.linalg.norm(pred - true, axis=-1) / scale        # (n_x0, steps)
        errors[row] = [err[:, :h].max() for h in horizons]
    return errors, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("system")
    parser.add_argument("--n-samples", type=int, default=1500, help="N_s in the paper")
    parser.add_argument("--horizons", type=int, nargs="+", default=[6, 12])
    parser.add_argument("--max-ic", type=int, default=None,
                        help="use only the first max-ic verification initial conditions")
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--n-chunks", type=int, default=1)
    parser.add_argument("--checkpoints", default=os.path.join(ROOT, "checkpoints"),
                        help="folder with <system>.pt and <system>_region.npy")
    parser.add_argument("--tag", default=None, help="name of the output folder")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default=os.path.join(ROOT, "runs", "region_check"))
    args = parser.parse_args()

    name = args.system
    system = experiment_system(name)
    model = load_model(os.path.join(args.checkpoints, f"{name}.pt"))
    region = np.load(os.path.join(args.checkpoints, f"{name}_region.npy"))
    hull = Delaunay(region)

    # every chunk draws the same full sample, then keeps its own slice
    thetas = sample_in_hull(hull, region, args.n_samples, np.random.default_rng(args.seed))
    idx = np.array_split(np.arange(args.n_samples), args.n_chunks)[args.chunk]
    x0s = system.test_initial_conditions()
    if args.max_ic is not None:
        x0s = x0s[:args.max_ic]

    print(f"[{name}] chunk {args.chunk + 1}/{args.n_chunks}: {len(idx)} parameters, "
          f"{len(x0s)} initial conditions, horizons {args.horizons}", flush=True)
    t0 = time.time()
    errors, valid = errors_at_horizons(model, system, thetas[idx], x0s, args.horizons)

    extra = {}
    if args.chunk == 0:
        # fraction of the box Theta inside the hull, from 200k uniform samples
        box = system.sample_theta(np.random.default_rng(args.seed + 1), 200_000)
        extra["volume_fraction"] = float(np.mean(hull.find_simplex(box) >= 0))

    out_dir = os.path.join(args.out, args.tag or name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"chunk{args.chunk}of{args.n_chunks}.npz")
    np.savez(path, index=idx, thetas=thetas[idx], errors=errors, valid=valid,
             horizons=np.array(args.horizons), n_ic=len(x0s), **extra)
    print(f"saved {path} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
