"""Ablation B (Fig. 5): full conditioning versus a fixed dictionary phi(x), psi(z)
with the same parameter-conditioned operator K(theta).

Both models are trained on the same random-parameter data, with the settings of
train.py, and tested at 30 parameters drawn from the hull built in Phases 1 and 2
(checkpoints/<system>_region_pre.npy). The script also stores a latent diagnostic:
one state is encoded under every test parameter and the spread of the codes is
measured in their first two principal components.

--test-params paper gives the 30 box parameters of the first preprint.

    python scripts/ablation_dictionary.py
    python scripts/ablation_dictionary.py --load runs/ablation_dictionary/checkpoints   # evaluate only
    python scripts/ablation_dictionary.py --system slow_manifold --seeds 0

Writes runs/ablation_dictionary/results.json and the model checkpoints.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.spatial import Delaunay
from scipy.stats import qmc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ablation_retraining import ANCHORS, DZ, TRAIN_OPTS, error_record, evaluate, make_dataset, test_initial_conditions
from pkoopman.model import PolicyKoopman, load_model, save_model
from pkoopman.region import sample_in_hull
from pkoopman.systems import experiment_system, get_system
from pkoopman.training import train
import train as train_script

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

LATENT_X0 = np.array([0.1, 0.0, 0.05, 0.0])
VARIANTS = ("conditioned", "fixed_dictionary")


def test_alphas(n_lhs=25, seed=2026):
    lhs = qmc.LatinHypercube(d=4, seed=seed).random(n_lhs)
    return np.vstack([ANCHORS, qmc.scale(lhs, -np.ones(4), np.ones(4))])


def region_alphas(name, n=30, seed=2026):
    # the hull before the final check (Phases 1 and 2)
    region = np.load(os.path.join(ROOT, "checkpoints", f"{name}_region_pre.npy"))
    return sample_in_hull(Delaunay(region), region, n, np.random.default_rng(seed))


def generic_setup(system, seed, args):
    """Data, model constructor and training options for a system other than the
    cartpole, taken from the settings in train.py."""
    cfg = train_script.CONFIGS[system.name]
    X, T = train_script.make_dataset(system, np.random.default_rng(seed),
                                     system.sample_theta(np.random.default_rng(seed + 1), args.n_traj or cfg["n_traj"]))
    opts = dict(batch_size=cfg.get("batch_size", 128), weights=cfg.get("weights"), clip=cfg.get("clip"))

    def build(conditioning):
        return PolicyKoopman(system.n, system.p, args.dz or cfg.get("dz", 10), hidden=cfg.get("hidden", (32,)),
                             conditioning=conditioning, small_init=cfg.get("small_init", False),
                             theta_scale=getattr(system, "theta_scale", None))
    return X, T, build, opts


def train_pair(seed, args, system):
    if args.paper_setup:
        X, T = make_dataset(system, np.random.RandomState(seed), n=args.n_traj or 20000)
        build = lambda conditioning: PolicyKoopman(4, 4, DZ, conditioning=conditioning, small_init=True)
        opts = TRAIN_OPTS
    else:
        X, T, build, opts = generic_setup(system, seed, args)
    models = {}
    # each model gets its own torch seed
    for name, conditioning, offset in [("conditioned", "full", 100000), ("fixed_dictionary", "operator", 200000)]:
        print(f"seed {seed}: training {name} model on {len(X)} trajectories", flush=True)
        torch.manual_seed(seed + offset)
        models[name], _ = train(build(conditioning), X, T, epochs=args.epochs, **opts)
        save_model(models[name], os.path.join(args.out, "checkpoints", f"{name}_seed_{seed}.pt"))
    return models


def latent_spread(model, alphas, x0):
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
    parser.add_argument("--n-traj", type=int, default=None,
                        help="training trajectories (default 20000 for the paper setup, else as in train.py)")
    parser.add_argument("--system", default="cartpole")
    parser.add_argument("--dz", type=int, default=None,
                        help="latent dimension for both variants (default as in train.py)")
    parser.add_argument("--test-params", choices=["paper", "region"], default=None,
                        help="paper: the 30 box parameters of the first preprint (cartpole only); "
                             "region: 30 fresh parameters from the deployment region (default for other systems)")
    parser.add_argument("--load", default=None, help="directory with <variant>_seed_<seed>.pt, skips training")
    parser.add_argument("--out", default="runs/ablation_dictionary")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out, "checkpoints"), exist_ok=True)
    if args.test_params is None:
        args.test_params = "region"
    args.paper_setup = args.system == "cartpole" and args.test_params == "paper"
    if args.paper_setup:
        system = get_system("cartpole")
        x0s, alphas, latent_x0 = test_initial_conditions(), test_alphas(), LATENT_X0
    else:
        system = experiment_system(args.system)
        x0s, alphas = system.test_initial_conditions(), region_alphas(args.system)
        latent_x0 = 0.25 * system.x0_high
    results = dict(config=vars(args), system=system.name, alphas=alphas.tolist(),
                   n_anchors=len(ANCHORS) if args.paper_setup else 0,
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
            results["latent"] = dict(seed=seed, x0=np.asarray(latent_x0).tolist(),
                                     **{v: latent_spread(models[v], alphas, latent_x0) for v in VARIANTS})
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
