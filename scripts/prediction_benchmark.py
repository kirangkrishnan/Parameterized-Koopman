"""Closed-loop prediction benchmark (Table II and Fig. 7 of the paper).

By default: 30 parameters drawn from the samples of the verified region C (the
same samples the final model was trained on), 20 initial conditions each.
--thetas fresh draws new parameters uniformly from the hull of C with an unused
seed, and --thetas box draws them uniformly from the whole box Theta, which also
tests every method outside the region chosen for our model.
Every method predicts 20 steps from x0; the table reports the worst-case error
over the first 12 steps. Methods:
  Ours          policy-conditioned model from checkpoints/<system>.pt
  PKNN          Guo et al., trained on random theta from the box, pure lifted rollout
  PKNN (re-lift) same model, state re-lifted after every step (not in the table)
  Deep Koopman  one fixed K, trained on the same data, ignores theta
  EDMD, Kernel-EDMD  fitted at the centroid of C, ignore theta

    python scripts/prediction_benchmark.py vdp
    python scripts/prediction_benchmark.py cartpole --retrain
    python scripts/prediction_benchmark.py manipulator --thetas fresh
    python scripts/prediction_benchmark.py mujoco_ip --thetas fresh --retrain

By default the PKNN and Deep Koopman weights from checkpoints/baselines/ are
used. --retrain trains them again with the same data, seeds and settings. The
MuJoCo systems have no stored baselines: without --retrain only Ours, EDMD and
Kernel-EDMD are run there.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.spatial import Delaunay

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from baselines import deep_koopman, pknn
from baselines.edmd import EDMD, KernelEDMD
from pkoopman.metrics import summarize
from pkoopman.model import load_model
from pkoopman.region import sample_in_hull
from pkoopman.systems import experiment_system, get_system

# x0_scale: initial conditions are uniform in [-x0_scale, x0_scale]^n (test and
# baseline training). n_train: trajectories for PKNN / Deep Koopman.
# degree, trig: EDMD dictionary.
SETTINGS = {
    "vdp": dict(x0_scale=0.5, n_train=4000, degree=3, trig=()),
    "slow_manifold": dict(x0_scale=0.5, n_train=4000, degree=3, trig=()),
    "slow_manifold_u2": dict(x0_scale=0.5, n_train=4000, degree=3, trig=()),
    "manipulator": dict(x0_scale=0.3, n_train=5000, degree=2, trig=(2, 3)),
    "cartpole": dict(x0_scale=0.2, n_train=10000, degree=2, trig=(2, 3)),
    # MuJoCo: test initial conditions in the box used to build the region
    "mujoco_ip": dict(x0_scale=np.array([0.2, 0.2, 0.08, 0.3]), n_train=10000, degree=2, trig=(2,)),
    "mujoco_idp": dict(x0_scale=np.array([0.15, 0.15, 0.06, 0.2, 0.06, 0.2]), n_train=10000,
                       degree=2, trig=(2, 4)),
}
N_THETA, N_IC = 30, 20          # paper values; --n-theta and --n-ic change them
TRAIN_HORIZON = 12      # baseline training rollout, and the horizon of Table II
MAX_STEPS = 20          # Fig. 7 extends the rollout to k = 20
REPORT_STEPS = (1, 3, 6, 9, 12, 15, 20)
METHODS = ["Ours", "PKNN", "PKNN (re-lift)", "Deep Koopman", "EDMD", "Kernel-EDMD"]


def simulate(system, x0, theta, steps):
    """Integrate one sampling interval at a time (RK45, max_step dt/4).
    Returns (steps+1, n), or None once a state component exceeds 15 in magnitude.
    MuJoCo systems are stepped by the simulator; None if the episode terminates."""
    if system.name.startswith("mujoco"):
        traj = system.simulate(x0, theta, steps)
        if traj is None or not np.all(np.isfinite(traj)) or np.any(np.abs(traj) > 15.0):
            return None
        return traj
    traj = [np.array(x0, dtype=float)]
    for _ in range(steps):
        sol = solve_ivp(system.closed_loop, (0.0, system.dt), traj[-1], args=(theta,),
                        max_step=system.dt / 4)
        x = sol.y[:, -1]
        if not np.all(np.isfinite(x)) or np.any(np.abs(x) > 15.0):
            return None
        traj.append(x)
    return np.array(traj)


def test_set(name, system, x0_scale, mode="region", n_theta=N_THETA, n_ic=N_IC, ckpt=None):
    """Test parameters and initial conditions. mode "region" draws them exactly as
    in the paper runs; "fresh" and "box" use a seed that training never saw."""
    region = np.load(os.path.join(ckpt or os.path.join(ROOT, "checkpoints"), f"{name}_region.npy"))
    if mode == "region":
        thetas = region[np.random.default_rng(42).choice(len(region), n_theta, replace=False)]
    elif mode == "fresh":
        thetas = sample_in_hull(Delaunay(region), region, n_theta, np.random.default_rng(2028))
    else:
        thetas = system.sample_theta(np.random.default_rng(2028), n_theta)
    x0s = np.random.RandomState(42).uniform(-x0_scale, x0_scale, size=(n_theta, n_ic, system.n))
    return thetas, x0s, region.mean(axis=0)


def training_data(system, x0_scale, n_traj, seed=0):
    """Trajectories of length TRAIN_HORIZON with theta uniform on the whole box Theta."""
    rs = np.random.RandomState(seed)
    low, high = system.theta_low, system.theta_high
    X, T = [], []
    attempts = 0
    while len(X) < n_traj and attempts < 8 * n_traj:
        attempts += 1
        theta = low + rs.rand(system.p) * (high - low)
        x0 = rs.uniform(-x0_scale, x0_scale, system.n)
        traj = simulate(system, x0, theta, TRAIN_HORIZON)
        if traj is not None:
            X.append(traj)
            T.append(theta)
    print(f"  collected {len(X)} trajectories ({attempts} attempts)")
    return torch.tensor(np.array(X, dtype=np.float32)), torch.tensor(np.array(T, dtype=np.float32))


def centroid_data(system, x0_scale, theta, n_traj=400, n_try=600):
    trajs = []
    for i in range(n_try):
        x0 = np.random.default_rng(i).uniform(-x0_scale, x0_scale, system.n)
        traj = simulate(system, x0, theta, TRAIN_HORIZON)
        if traj is not None:
            trajs.append(traj)
        if len(trajs) >= n_traj:
            break
    return trajs


def baselines(name, system, cfg, centroid, retrain, out_dir, neural=True):
    if not neural:
        pknn_model = dk_model = None
    elif retrain:
        X, T = training_data(system, cfg["x0_scale"], cfg["n_train"])
        pknn_model = pknn.train_pknn(X, T)
        dk_model = deep_koopman.train_deep_koopman(X)
        pknn.save_pknn(pknn_model, os.path.join(out_dir, f"pknn_{name}.pt"))
        deep_koopman.save_deep_koopman(dk_model, os.path.join(out_dir, f"deep_koopman_{name}.pt"))
    else:
        ckpt_dir = os.path.join(ROOT, "checkpoints", "baselines")
        pknn_model = pknn.load_pknn(os.path.join(ckpt_dir, f"pknn_{name}.pt"))
        dk_model = deep_koopman.load_deep_koopman(os.path.join(ckpt_dir, f"deep_koopman_{name}.pt"))

    trajs = centroid_data(system, cfg["x0_scale"], centroid)
    edmd = EDMD(system.n, cfg["degree"], cfg["trig"]).fit(trajs)
    kedmd = KernelEDMD().fit(trajs[:100])
    print(f"  EDMD: {len(trajs)} centroid trajectories, {edmd.K.shape[0]} observables; "
          f"Kernel-EDMD: gamma {kedmd.gamma:.4f}")
    return pknn_model, dk_model, edmd, kedmd


def predictors(name, system, cfg, centroid, retrain, out_dir, methods, ckpt=None):
    """Each predictor maps (x0, theta, steps) to an array (steps, n) of x_1..x_steps."""
    # the paper evaluation ran our model through numpy in float64
    ours = load_model(os.path.join(ckpt or os.path.join(ROOT, "checkpoints"), f"{name}.pt")).double()
    neural = any(m in methods for m in ("PKNN", "PKNN (re-lift)", "Deep Koopman"))
    pknn_model, dk_model, edmd, kedmd = baselines(name, system, cfg, centroid, retrain, out_dir, neural)

    # one case at a time like the paper runs; batching changes the float32 rounding
    def torch_call(fn, model, dtype):
        def predict(x0, theta, steps):
            x = torch.tensor(x0, dtype=dtype)[None]
            th = torch.tensor(theta, dtype=dtype)[None]
            with torch.no_grad():
                return fn(model, x, th, steps)[0].numpy()
        return predict

    fns = {
        "Ours": torch_call(lambda m, x, th, s: m.predict(x, th, s), ours, torch.float64),
        "PKNN": torch_call(pknn.predict_pure, pknn_model, torch.float32),
        "PKNN (re-lift)": torch_call(pknn.predict_relift, pknn_model, torch.float32),
        "Deep Koopman": torch_call(lambda m, x, th, s: deep_koopman.predict(m, x, s), dk_model, torch.float32),
        "EDMD": lambda x0, theta, steps: edmd.predict(x0, steps),
        "Kernel-EDMD": lambda x0, theta, steps: kedmd.predict(x0, steps),
    }
    return {m: fns[m] for m in methods}


def step_errors(x_pred, x_true, floor=0.1):
    """Normalized error in percent at every step; its running max is the worst-case metric of the paper."""
    scale = np.maximum(np.linalg.norm(x_true, axis=-1), floor)
    return 100.0 * np.linalg.norm(x_pred - x_true, axis=-1) / scale


def evaluate(system, thetas, x0s, predict_fns):
    """Errors of shape (N_THETA, N_IC, MAX_STEPS) per method. Cases where the true
    trajectory diverges stay NaN; a non-finite prediction counts as 100%."""
    n_theta, n_ic = x0s.shape[:2]
    errors = {m: np.full((n_theta, n_ic, MAX_STEPS), np.nan) for m in predict_fns}
    for i, theta in enumerate(thetas):
        for j, x0 in enumerate(x0s[i]):
            traj = simulate(system, x0, theta, MAX_STEPS)
            if traj is None:
                continue
            for m, predict in predict_fns.items():
                x_pred = predict(x0, theta, MAX_STEPS)
                ok = np.all(np.isfinite(x_pred))
                errors[m][i, j] = step_errors(x_pred, traj[1:]) if ok else 100.0
        print(f"  theta {i + 1:2d}/{n_theta}", flush=True)
    return errors


def stats(values):
    values = values[np.isfinite(values)]
    if values.size == 0:
        return dict(mean=float("nan"), median=float("nan"), std=float("nan"),
                    max=float("nan"), below_threshold=float("nan"))
    return summarize(values)


def results_dict(name, errors, mode="region"):
    shape = next(iter(errors.values())).shape
    out = dict(system=name, thetas=mode, n_theta=shape[0], n_ic=shape[1], train_horizon=TRAIN_HORIZON,
               max_steps=MAX_STEPS, methods={})
    for m, e in errors.items():
        worst = np.maximum.accumulate(e, axis=2)    # worst-case error over steps 1..k
        out["methods"][m] = dict(
            n_cases=int(np.isfinite(e[:, :, 0]).sum()),
            # cases whose true trajectory left the bounds and were not scored
            n_true_failed=int((~np.isfinite(e[:, :, 0])).sum()),
            mean_step_error=np.nanmean(e, axis=(0, 1)).tolist(),
            mean_worst_case_error=np.nanmean(worst, axis=(0, 1)).tolist(),
            worst_case=dict((str(k), stats(worst[:, :, k - 1])) for k in REPORT_STEPS),
            step_error=dict((str(k), stats(e[:, :, k - 1])) for k in REPORT_STEPS),
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("system", choices=list(SETTINGS))
    parser.add_argument("--retrain", action="store_true",
                        help="train PKNN and Deep Koopman again instead of loading checkpoints/baselines")
    parser.add_argument("--thetas", choices=["region", "fresh", "box"], default="region",
                        help="where the test parameters come from (see the top of this file)")
    parser.add_argument("--n-theta", type=int, default=N_THETA)
    parser.add_argument("--n-ic", type=int, default=N_IC)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=None)
    parser.add_argument("--checkpoints", default=None,
                        help="folder with our <system>.pt and <system>_region.npy (default checkpoints/)")
    parser.add_argument("--out", default=os.path.join(ROOT, "runs", "prediction_benchmark"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    name, cfg = args.system, SETTINGS[args.system]
    methods = args.methods or list(METHODS)
    if name.startswith("mujoco") and not args.retrain and args.methods is None:
        methods = ["Ours", "EDMD", "Kernel-EDMD"]
    # the benchmark used the cartpole without the actuator limit
    system = get_system(name, u_max=None) if name == "cartpole" else experiment_system(name)
    thetas, x0s, centroid = test_set(name, system, cfg["x0_scale"], args.thetas, args.n_theta, args.n_ic,
                                    args.checkpoints)

    t0 = time.time()
    predict_fns = predictors(name, system, cfg, centroid, args.retrain, args.out, methods,
                             args.checkpoints)
    errors = evaluate(system, thetas, x0s, predict_fns)
    results = results_dict(name, errors, args.thetas)

    stem = name if args.thetas == "region" else f"{name}_{args.thetas}"
    with open(os.path.join(args.out, f"{stem}.json"), "w") as f:
        json.dump(results, f, indent=1)
    np.savez(os.path.join(args.out, f"{stem}_errors.npz"), thetas=thetas, x0s=x0s,
             **{m.replace(" ", "_"): e for m, e in errors.items()})

    print(f"\n{name} ({args.thetas} parameters): worst-case error over {TRAIN_HORIZON} steps (%), "
          f"{results['methods']['Ours']['n_cases']} cases ({time.time() - t0:.0f}s)")
    print(f"{'method':16s} {'mean':>9s} {'std':>9s} {'median':>9s} {'<15%':>6s}")
    for m in methods:
        s = results["methods"][m]["worst_case"][str(TRAIN_HORIZON)]
        print(f"{m:16s} {s['mean']:9.2f} {s['std']:9.2f} {s['median']:9.2f} {s['below_threshold']:6.2f}")


if __name__ == "__main__":
    main()
