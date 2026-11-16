"""Train a policy-conditioned Koopman model and build its verified region.

The pipeline has two stages (Sections IV and V of the paper):
  1. train a scout model on random parameters from the whole box Theta,
     then use it to find the region C where its predictions pass the test;
  2. regenerate data with parameters from C and train the final model,
     then check the final model on fresh parameters from C and shrink C if
     too many fail (Phase 3, pkoopman.region.verify_region).

    python scripts/train.py vdp
    python scripts/train.py cartpole --scout-epochs 200 --final-epochs 200   # quick test
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pkoopman.model import PolicyKoopman, save_model
from pkoopman.region import find_region, verify_region
from pkoopman.systems import experiment_system
from pkoopman.training import train

# Settings used for the models in checkpoints/. Everything not listed uses the
# defaults: latent dim 10, one hidden layer of 32, batch 128, loss weights
# rec 1, pred 2, lin 1, cons 0.1, reg 1e-4, 5000 epochs per stage.
CONFIGS = {
    "vdp": dict(n_traj=4000, n_explore=4000, n_final=4000),
    "slow_manifold": dict(n_traj=4000, n_explore=3000, n_final=4000, n_test=500,
                          weights=dict(pred=4.0), require_valid=True),
    # no region passes at 15% for the manipulator; the paper uses this model, trained
    # on a 1.2x wider box of initial states, with its region built and checked at 20%
    "manipulator": dict(n_traj=5000, n_explore=5000, n_final=5000,
                        weights=dict(pred=4.0), small_init=True, require_valid=True,
                        threshold=20.0, check_steps=12, x0_margin=1.2, verify_threshold=20.0),
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


def make_dataset(system, rng, thetas, x0_margin=1.0):
    """One trajectory of length HORIZON per parameter, from a random initial state.
    x0_margin > 1 widens the box of initial states around its centre, so that the
    corners of the verification grid lie inside the training data."""
    centre = 0.5 * (system.x0_low + system.x0_high)
    half = 0.5 * (system.x0_high - system.x0_low) * x0_margin
    X, T = [], []
    for theta in thetas:
        while True:
            x0 = rng.uniform(centre - half, centre + half) if x0_margin != 1.0 else system.sample_x0(rng, 1)[0]
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
    parser.add_argument("--x0-margin", type=float, default=None,
                        help="widen the box of training initial states by this factor")
    parser.add_argument("--verify-threshold", type=float, default=None,
                        help="Phase 3 tolerance in percent (default 15, 20 for the manipulator)")
    parser.add_argument("--verify-steps", type=int, default=12, help="Phase 3 horizon")
    parser.add_argument("--acceptance", type=float, default=0.95,
                        help="Phase 3 accepts the region when this fraction of fresh parameters passes")
    parser.add_argument("--no-phase3", action="store_true", help="stop after training the final model")
    parser.add_argument("--out", default="runs/train")
    args = parser.parse_args()

    cfg = dict(CONFIGS[args.system])
    if args.threshold is not None:
        cfg["threshold"] = args.threshold
    if args.check_steps is not None:
        cfg["check_steps"] = args.check_steps
    if args.x0_margin is None:
        args.x0_margin = cfg.get("x0_margin", 1.0)
    if args.verify_threshold is None:
        args.verify_threshold = cfg.get("verify_threshold", 15.0)
    system = experiment_system(args.system)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    train_opts = dict(batch_size=cfg.get("batch_size", 128), weights=cfg.get("weights"),
                      clip=cfg.get("clip"))

    print(f"[{system.name}] scout data: {cfg['n_traj']} trajectories")
    X, T = make_dataset(system, rng, system.sample_theta(rng, cfg["n_traj"]), args.x0_margin)
    scout, _ = train(new_model(system, cfg), X, T, epochs=args.scout_epochs, **train_opts)
    save_model(scout, f"{args.out}/{system.name}_scout.pt")

    print(f"[{system.name}] building the verified region")
    region = find_region(scout, system, rng, n_explore=cfg["n_explore"], n_test=cfg.get("n_test", 1500),
                         n_final=cfg["n_final"], steps=cfg.get("check_steps", CHECK_STEPS),
                         threshold=cfg.get("threshold", THRESHOLD),
                         require_valid=cfg.get("require_valid", False))
    np.save(f"{args.out}/{system.name}_region_pre.npy", region["final"])
    np.savez(f"{args.out}/{system.name}_region_history.npz", *region["history"],
             explored=region["explored"])

    print(f"[{system.name}] final data: {len(region['final'])} trajectories")
    X, T = make_dataset(system, rng, region["final"], args.x0_margin)
    final, history = train(new_model(system, cfg), X, T, epochs=args.final_epochs, **train_opts)
    save_model(final, f"{args.out}/{system.name}.pt")
    np.save(f"{args.out}/{system.name}_loss.npy", np.array([h["total"] for h in history]))
    print(f"saved {args.out}/{system.name}.pt")

    region_file = f"{args.out}/{system.name}_region.npy"
    if args.no_phase3:
        np.save(region_file, region["final"])
        return
    print(f"[{system.name}] phase 3: final model on fresh parameters "
          f"({args.verify_threshold:g}%, {args.verify_steps} steps, accept at {args.acceptance:.0%})")
    check = verify_region(final, system, region["final"], rng, n_samples=cfg.get("n_test", 1500),
                          steps=args.verify_steps, threshold=args.verify_threshold,
                          acceptance=args.acceptance, require_valid=cfg.get("require_valid", False))
    # if phase 3 fails the unverified region is kept so that later scripts still run,
    # and phase3.json says so
    np.save(region_file, check["final"] if check["accepted"] else region["final"])
    with open(f"{args.out}/{system.name}_phase3.json", "w") as f:
        json.dump(dict(accepted=check["accepted"], threshold=args.verify_threshold,
                       steps=args.verify_steps, acceptance=args.acceptance,
                       x0_margin=args.x0_margin, construction_threshold=cfg.get("threshold", THRESHOLD),
                       construction_steps=cfg.get("check_steps", CHECK_STEPS),
                       history=check["history"]), f, indent=1)
    print(f"[{system.name}] phase 3 {'accepted' if check['accepted'] else 'FAILED'}; region in {region_file}")


if __name__ == "__main__":
    main()
