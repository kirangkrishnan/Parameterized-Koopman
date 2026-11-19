"""Ablation A (Fig. 4): one policy-conditioned model versus a plain Koopman model
retrained from scratch at every test parameter, on the cartpole.

For each seed the conditioned model is trained once on random parameters, and one
unconditioned model is trained per test parameter on data from that parameter only.
Both are scored with the mean, over 16 initial conditions, of the worst-case 12-step
error. Data generation time, training time and the time the conditioned model needs
to produce K(theta) are saved as well.

    python scripts/ablation_retraining.py --seeds 0 8 17 42 99
    python scripts/ablation_retraining.py --seeds 0 --epochs 2 --n-traj 300 --n-traj-single 100   # quick test

One json per seed is written to runs/ablation_retraining/.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pkoopman.metrics import worst_case_error
from pkoopman.model import PolicyKoopman, mlp
from pkoopman.systems import get_system
from pkoopman.training import train

# the five hand-picked parameters (normalized alpha in [-1, 1]^4)
ANCHORS = np.array([
    [0.0, 0.0, 0.0, 0.0],
    [0.7, 0.7, 0.7, 0.7],
    [-0.7, -0.7, -0.7, -0.7],
    [0.7, -0.7, 0.7, -0.7],
    [-0.5, 0.5, -0.5, 0.5],
])

HORIZON = 12
DZ = 10
TRAIN_OPTS = dict(batch_size=256, weights=dict(pred=8.0, lin=4.0), clip=1.0)


def test_initial_conditions():
    """The 2^4 corner grid used for every ablation evaluation."""
    axes = [np.linspace(-0.4, 0.4, 2), np.linspace(-0.4, 0.4, 2),
            np.linspace(-0.25, 0.25, 2), np.linspace(-0.6, 0.6, 2)]
    return np.array(np.meshgrid(*axes, indexing="ij")).reshape(4, -1).T


def make_dataset(system, rs, n=None, alphas=None):
    """Cartpole trajectories of length HORIZON, rejecting the ones that leave
    |p| <= 10, |phi| <= 20 deg.

    rs is a np.random.RandomState. The draws are made in a fixed order (x0, then a
    random alpha or a random index into `alphas`), so a seed always gives the same
    data set."""
    if alphas is not None:
        n = len(alphas)
    X, T = [], []
    attempts = 0
    while len(X) < n:
        attempts += 1
        if attempts > 20 * n:
            raise RuntimeError(f"only {len(X)} of {n} trajectories stayed in bounds")
        x0 = rs.uniform(system.x0_low, system.x0_high)
        if alphas is None:
            alpha = rs.uniform(-1.0, 1.0, size=4)
        else:
            alpha = alphas[rs.randint(len(alphas))]
        traj = system.simulate(x0, alpha, HORIZON)
        if system.in_bounds(traj):
            X.append(traj)
            T.append(alpha)
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(T), dtype=torch.float32)


class FixedKoopman(PolicyKoopman):
    """Unconditioned Koopman model for one parameter: phi(x), psi(z) and a single
    learned pair (K_fwd, K_bwd). theta is accepted and ignored, so the same loss
    and training loop as for PolicyKoopman apply."""

    def __init__(self, n, dz, hidden=(32,)):
        nn.Module.__init__(self)   # skip PolicyKoopman.__init__, there is no operator network
        self.n, self.p, self.dz, self.hidden = n, 0, dz, tuple(hidden)
        self.conditioning = "none"
        self.register_buffer("theta_scale", torch.ones(0))
        self.encoder = mlp(n, dz, hidden, nn.Tanh)
        self.decoder = mlp(dz, n, hidden, nn.Tanh)
        self.K_fwd = nn.Parameter(torch.eye(dz) + 0.01 * torch.randn(dz, dz))
        self.K_bwd = nn.Parameter(torch.eye(dz) + 0.01 * torch.randn(dz, dz))

    def operators(self, theta):
        batch = theta.shape[0]
        return self.K_fwd.expand(batch, -1, -1), self.K_bwd.expand(batch, -1, -1)


def evaluate(model, system, alpha, x0s):
    """Worst-case error over the horizon for every initial condition."""
    trajs = [system.simulate(x0, alpha, HORIZON) for x0 in x0s]
    x0s = np.asarray(x0s)[[tr is not None for tr in trajs]]      # skip closed loops that failed
    true = np.stack([tr[1:] for tr in trajs if tr is not None])
    device = next(model.parameters()).device
    x0_t = torch.tensor(x0s, dtype=torch.float32, device=device)
    theta = torch.tensor(alpha, dtype=torch.float32, device=device).expand(len(x0s), -1)
    with torch.no_grad():
        pred = model.predict(x0_t, theta, HORIZON).cpu().numpy()
    return worst_case_error(pred, true)


def error_record(errors):
    return dict(mean_err=float(errors.mean()), std_err=float(errors.std()),
                median_err=float(np.median(errors)), errors=errors.tolist())


def query_time(model, alpha, repeats=200):
    """Seconds needed to produce K(theta) for one new parameter."""
    device = next(model.parameters()).device
    theta = torch.tensor(alpha, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        for _ in range(20):
            model.K(theta)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            model.K(theta)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats


def run_seed(seed, args, system, alphas, x0s):
    # one numpy stream and one torch stream per seed, used in this order:
    # conditioned data, conditioned training, then data and training per alpha
    rs = np.random.RandomState(seed)
    torch.manual_seed(seed)
    out = dict(seed=seed, config=vars(args), alphas=alphas.tolist(), conditioned=[], unconditioned=[])

    t0 = time.perf_counter()
    X, T = make_dataset(system, rs, n=args.n_traj)
    out["conditioned_data_time"] = time.perf_counter() - t0
    print(f"seed {seed}: conditioned model, {len(X)} trajectories", flush=True)
    t0 = time.perf_counter()
    cond, _ = train(PolicyKoopman(4, 4, DZ, small_init=True), X, T, epochs=args.epochs, **TRAIN_OPTS)
    out["conditioned_train_time"] = time.perf_counter() - t0

    for i, alpha in enumerate(alphas):
        rec = dict(alpha_idx=i, alpha=alpha.tolist(), **error_record(evaluate(cond, system, alpha, x0s)))
        rec["query_time"] = query_time(cond, alpha)
        out["conditioned"].append(rec)
        print(f"  conditioned    alpha {i}: {rec['mean_err']:.3f}%  (K(theta) in {rec['query_time']:.2e} s)")

    path = os.path.join(args.out, f"seed_{seed}.json")
    for i, alpha in enumerate(alphas):
        t0 = time.perf_counter()
        X, T = make_dataset(system, rs, alphas=np.tile(alpha, (args.n_traj_single, 1)))
        data_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        model, _ = train(FixedKoopman(4, DZ), X, T, epochs=args.epochs, **TRAIN_OPTS)
        train_time = time.perf_counter() - t0

        rec = dict(alpha_idx=i, alpha=alpha.tolist(), data_time=data_time, train_time=train_time,
                   **error_record(evaluate(model, system, alpha, x0s)))
        out["unconditioned"].append(rec)
        print(f"  unconditioned  alpha {i}: {rec['mean_err']:.3f}%  "
              f"(data {data_time:.0f} s, training {train_time:.0f} s)", flush=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 8, 17, 42, 99])
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--n-traj", type=int, default=20000, help="data for the conditioned model")
    parser.add_argument("--n-traj-single", type=int, default=4000, help="data per unconditioned model")
    parser.add_argument("--out", default="runs/ablation_retraining")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    system = get_system("cartpole")
    x0s = test_initial_conditions()
    for seed in args.seeds:
        print("saved", run_seed(seed, args, system, ANCHORS, x0s))


if __name__ == "__main__":
    main()
