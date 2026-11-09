"""Ablation B (paper Q1, Figs. 5-6): fully conditioned model versus a fixed
dictionary phi(x), psi(z) with the same parameter-conditioned operator K(theta).

Both models are trained on the same 20000 random-parameter cartpole trajectories
with identical settings and evaluated at 30 test parameters (the five
hand-picked ones of ablation A plus 25 Latin-hypercube points, seed 2026).
Every parameter is reported. The latent diagnostic encodes one state under every
test parameter with the first seed's models and measures the spread of the codes
in their first two principal components.

    python scripts/ablation_dictionary.py
    python scripts/ablation_dictionary.py --load runs/ablation_dictionary/checkpoints   # evaluate only

Writes runs/ablation_dictionary/results.json and the model checkpoints.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import qmc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ablation_retraining import ANCHORS, DZ, TRAIN_OPTS, error_record, evaluate, make_dataset, test_initial_conditions
from pkoopman.model import PolicyKoopman, load_model, save_model
from pkoopman.systems import get_system
from pkoopman.training import train

LATENT_X0 = np.array([0.1, 0.0, 0.05, 0.0])
VARIANTS = ("conditioned", "fixed_dictionary")


def test_alphas(n_lhs=25, seed=2026):
    lhs = qmc.LatinHypercube(d=4, seed=seed).random(n_lhs)
    return np.vstack([ANCHORS, qmc.scale(lhs, -np.ones(4), np.ones(4))])


def train_pair(seed, args, system):
    rs = np.random.RandomState(seed)
    X, T = make_dataset(system, rs, n=args.n_traj)
    models = {}
    # each model gets its own torch seed, as in the paper runs
    for name, conditioning, offset in [("conditioned", "full", 100000), ("fixed_dictionary", "operator", 200000)]:
        print(f"seed {seed}: training {name} model on {len(X)} trajectories", flush=True)
        torch.manual_seed(seed + offset)
        model = PolicyKoopman(4, 4, DZ, conditioning=conditioning, small_init=True)
        models[name], _ = train(model, X, T, epochs=args.epochs, **TRAIN_OPTS)
        save_model(models[name], os.path.join(args.out, "checkpoints", f"{name}_seed_{seed}.pt"))
    return models


def latent_spread(model, alphas, x0=LATENT_X0):
    """Codes of one state under every alpha, their 2D PCA coordinates and the
    spread (sum of the standard deviations of the two coordinates)."""
    x0_t = torch.tensor(x0, dtype=torch.float32).expand(len(alphas), -1)
    with torch.no_grad():
        z = model.encode(x0_t, torch.tensor(alphas, dtype=torch.float32)).numpy().astype(float)
    u, s, _ = np.linalg.svd(z - z.mean(axis=0), full_matrices=False)
    pcs = u[:, :2] * s[:2]
    return dict(codes=z.tolist(), pca=pcs.tolist(), spread_pca=float(pcs.std(axis=0).sum()),
                spread_raw=float(z.std(axis=0).sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 8, 17, 42, 99])
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--n-traj", type=int, default=20000)
    parser.add_argument("--load", default=None, help="directory with <variant>_seed_<seed>.pt, skips training")
    parser.add_argument("--out", default="runs/ablation_dictionary")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out, "checkpoints"), exist_ok=True)
    system = get_system("cartpole")
    x0s = test_initial_conditions()
    alphas = test_alphas()
    results = dict(config=vars(args), alphas=alphas.tolist(), n_anchors=len(ANCHORS),
                   conditioned=[], fixed_dictionary=[])

    for seed in args.seeds:
        if args.load:
            models = {v: load_model(os.path.join(args.load, f"{v}_seed_{seed}.pt")) for v in VARIANTS}
        else:
            models = train_pair(seed, args, system)
        for v in VARIANTS:
            models[v].cpu()
            for i, alpha in enumerate(alphas):
                rec = dict(seed=seed, alpha_idx=i, **error_record(evaluate(models[v], system, alpha, x0s)))
                results[v].append(rec)
            errs = [r["mean_err"] for r in results[v] if r["seed"] == seed]
            print(f"seed {seed}  {v:16s}  mean over {len(alphas)} parameters {np.mean(errs):.3f}%", flush=True)

        if seed == args.seeds[0]:
            results["latent"] = dict(seed=seed, x0=LATENT_X0.tolist(),
                                     **{v: latent_spread(models[v], alphas) for v in VARIANTS})
            print(f"latent spread (PCA plane): conditioned {results['latent']['conditioned']['spread_pca']:.4e}, "
                  f"fixed dictionary {results['latent']['fixed_dictionary']['spread_raw']:.4e}")

        with open(os.path.join(args.out, "results.json"), "w") as f:
            json.dump(results, f, indent=1)

    for v in VARIANTS:
        per_alpha = np.array([[r["mean_err"] for r in results[v] if r["alpha_idx"] == i]
                              for i in range(len(alphas))])
        print(f"{v:16s}  mean {per_alpha.mean():.3f}%  seed-to-seed std (mean over parameters) "
              f"{per_alpha.std(axis=1).mean():.3f}")


if __name__ == "__main__":
    main()
