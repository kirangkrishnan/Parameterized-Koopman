"""Train a policy-conditioned Koopman model and build its verified region.

The pipeline has two stages (Sections IV and V of the paper):
  1. train a scout model on random parameters from the whole box Theta,
     then use it to find the region C where its predictions pass the test;
  2. regenerate data with parameters from C and train the final model.

    python scripts/train.py vdp
    python scripts/train.py cartpole --scout-epochs 200 --final-epochs 200   # quick test
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pkoopman.model import PolicyKoopman, save_model
from pkoopman.region import find_region
from pkoopman.systems import get_system
from pkoopman.training import train

# Settings used for the models in checkpoints/. Everything not listed uses the
# defaults: latent dim 10, one hidden layer of 32, batch 128, loss weights
# rec 1, pred 2, lin 1, cons 0.1, reg 1e-4, 5000 epochs per stage.
CONFIGS = {
    "vdp": dict(n_traj=4000, n_explore=4000, n_final=4000),
    "slow_manifold": dict(n_traj=4000, n_explore=3000, n_final=4000, n_test=500,
                          weights=dict(pred=4.0), require_valid=True),
    "slow_manifold_u2": dict(n_traj=4000, n_explore=3000, n_final=4000, n_test=500,
                             weights=dict(pred=4.0), require_valid=True),
    "manipulator": dict(n_traj=5000, n_explore=5000, n_final=5000,
                        weights=dict(pred=4.0), small_init=True, require_valid=True),
    "cartpole": dict(n_traj=20000, n_explore=10000, n_final=10000, batch_size=256,
                     weights=dict(pred=8.0, lin=4.0), clip=1.0, small_init=True),
    # the MuJoCo models were checked with a loose 100% threshold
    "mujoco_ip": dict(n_traj=20000, n_explore=10000, n_final=10000, batch_size=256,
                      weights=dict(pred=8.0, lin=4.0), clip=1.0, small_init=True,
                      check_steps=4, threshold=100.0),
    "mujoco_idp": dict(n_traj=20000, n_explore=10000, n_final=10000, batch_size=256, dz=20,
                       hidden=(128, 128), weights=dict(pred=8.0, lin=4.0), clip=1.0, small_init=True,
                       check_steps=12, threshold=100.0, require_valid=True),
}

HORIZON = 12        # rollout length in the training loss
CHECK_STEPS = 6     # steps used in the region test (unless a config says otherwise)
THRESHOLD = 15.0    # percent


def keep(system, traj):
    """Training trajectories that leave the region of interest are thrown away."""
    if traj is None:
        return False
    if system.name == "cartpole":
        return system.in_bounds(traj)
    if system.name == "manipulator":
        return np.abs(traj).max() <= system.state_limit
    return True


def make_dataset(system, rng, thetas):
    """One trajectory of length HORIZON per parameter, from a random initial state."""
    X, T = [], []
    for theta in thetas:
        while True:
            x0 = system.sample_x0(rng, 1)[0]
            traj = system.simulate(x0, theta, HORIZON)
            if keep(system, traj):
                break
        X.append(traj)
        T.append(theta)
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(T), dtype=torch.float32)


def new_model(system, cfg):
    return PolicyKoopman(system.n, system.p, cfg.get("dz", 10), hidden=cfg.get("hidden", (32,)),
                         small_init=cfg.get("small_init", False),
                         theta_scale=getattr(system, "theta_scale", None))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("system", choices=sorted(CONFIGS))
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--scout-epochs", type=int, default=5000)
    parser.add_argument("--final-epochs", type=int, default=5000)
    parser.add_argument("--threshold", type=float, default=None,
                        help="region tolerance in percent (overrides the setting of the system)")
    parser.add_argument("--check-steps", type=int, default=None,
                        help="steps in the region test (overrides the setting of the system)")
    parser.add_argument("--out", default="runs/train")
    args = parser.parse_args()

    cfg = dict(CONFIGS[args.system])
    if args.threshold is not None:
        cfg["threshold"] = args.threshold
    if args.check_steps is not None:
        cfg["check_steps"] = args.check_steps
    system = get_system(args.system)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    train_opts = dict(batch_size=cfg.get("batch_size", 128), weights=cfg.get("weights"),
                      clip=cfg.get("clip"))

    print(f"[{system.name}] scout data: {cfg['n_traj']} trajectories")
    X, T = make_dataset(system, rng, system.sample_theta(rng, cfg["n_traj"]))
    scout, _ = train(new_model(system, cfg), X, T, epochs=args.scout_epochs, **train_opts)
    save_model(scout, f"{args.out}/{system.name}_scout.pt")

    print(f"[{system.name}] building the verified region")
    region = find_region(scout, system, rng, n_explore=cfg["n_explore"], n_test=cfg.get("n_test", 1500),
                         n_final=cfg["n_final"], steps=cfg.get("check_steps", CHECK_STEPS),
                         threshold=cfg.get("threshold", THRESHOLD),
                         require_valid=cfg.get("require_valid", False))
    np.save(f"{args.out}/{system.name}_region.npy", region["final"])
    np.savez(f"{args.out}/{system.name}_region_history.npz", *region["history"],
             explored=region["explored"])

    print(f"[{system.name}] final data: {len(region['final'])} trajectories")
    X, T = make_dataset(system, rng, region["final"])
    final, history = train(new_model(system, cfg), X, T, epochs=args.final_epochs, **train_opts)
    save_model(final, f"{args.out}/{system.name}.pt")
    np.save(f"{args.out}/{system.name}_loss.npy", np.array([h["total"] for h in history]))
    print(f"saved {args.out}/{system.name}.pt")


if __name__ == "__main__":
    main()
