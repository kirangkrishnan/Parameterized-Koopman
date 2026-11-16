"""Closed-loop control from many initial states (extends Table III).

Table III uses one initial state per system. This script runs each method from
N_X0 initial states drawn uniformly from X0_BOX and records, per episode, whether
the system was regulated, the settling time, the closed-loop cost, the control
effort and the online computation time (for the one-shot methods the selection
time is included).

Methods
  ours_x, ours_z  one-shot selection with the decoded or the latent objective
  centroid        no selection: the mean of the verified samples is deployed
  sim_opt         the same selection problem solved on the true simulator
                  (every cost evaluation is a closed-loop simulation); this is the
                  reference for how much the surrogate loses
  nmpc, kmpc_edmd, kmpc_deep   the receding-horizon baselines of Table III

    python scripts/control_benchmark.py --system cartpole --method ours_x
    python scripts/control_benchmark.py --system mujoco_ip --method ours_x --T 60 --kick 0.5
    python scripts/control_benchmark.py --system manipulator --method sim_opt --chunk 2 --n-chunks 10

--kick s adds, at the middle of the episode, a state perturbation drawn uniformly
from s * X0_BOX (the same perturbation for every method).
Results go to runs/control_benchmark/<system>/<method>[_kick<s>]/<first>-<last>.jsonl,
one line per episode; collect_results.py summarizes them.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from scipy.integrate import solve_ivp

from control_comparison import EPISODES, MISSING, make_controller
from pkoopman.model import load_model
from pkoopman.selection import select_theta, select_theta_on_simulator
from pkoopman.systems import experiment_system

METHODS = ["ours_x", "ours_z", "centroid", "sim_opt", "nmpc", "kmpc_edmd", "kmpc_deep"]
N_X0 = 50
# Initial states are uniform in [-b, b]. The boxes contain the Table III initial
# state of each system and lie inside the box the model was trained on.
X0_BOX = {
    "vdp": [1.0, 1.0],
    "slow_manifold": [1.0, 1.0],
    "manipulator": [0.3, 0.3, 0.2, 0.2],
    "cartpole": [0.2, 0.2, 0.1, 0.2],
    "mujoco_ip": [0.1, 0.1, 0.05, 0.1],
    "mujoco_idp": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
}
SUCCESS_TOL = 0.05      # ||x_T|| below this, and no early termination
DIVERGED = 50.0         # an analytic episode is stopped once a state exceeds this
EVAL_R = 0.1            # weight on the input in the reported closed-loop cost


def initial_states(name, n):
    b = np.array(X0_BOX[name])
    return np.random.default_rng(2029).uniform(-b, b, size=(n, len(b)))


def kick_vector(name, index, size):
    b = np.array(X0_BOX[name])
    return size * np.random.default_rng(3000 + index).uniform(-b, b)


def run_episode(system, x0, steps, policy, u_max=None, fixed_feedback=False, kick=None):
    """Returns states (done+1, n), inputs (done, nu) and whether the episode
    ended early (MuJoCo termination, or divergence of an analytic plant).
    For fixed feedback laws the feedback is evaluated inside the integrator,
    otherwise the input is held over the step (as in control_comparison.py)."""
    mujoco = system.name.startswith("mujoco")
    k_kick = steps // 2 if kick is not None else -1
    x = system.reset(x0) if mujoco else np.array(x0, dtype=np.float64)
    traj, inputs, ended = [x.copy()], [], False
    for k in range(steps):
        if k == k_kick:
            x = system.reset(x + kick) if mujoco else x + kick
        if mujoco:
            u = policy(x)
            x, ended = system.step(u)
        else:
            u = np.clip(policy(x), -u_max, u_max)
            if fixed_feedback:
                rhs = lambda t, y: system.f(y, np.clip(policy(y), -u_max, u_max))
            else:
                rhs = lambda t, y: system.f(y, u)
            x = solve_ivp(rhs, [0.0, system.dt], x, max_step=system.dt / 4).y[:, -1]
            ended = not np.all(np.isfinite(x)) or np.abs(x).max() > DIVERGED
        traj.append(np.array(x, dtype=np.float64))
        inputs.append(np.atleast_1d(np.asarray(u, dtype=np.float64)))
        if ended:
            break
    return np.array(traj), np.array(inputs), bool(ended)


def episode_metrics(traj, inputs, dt, ended, tol=SUCCESS_TOL):
    norms = np.linalg.norm(traj, axis=1)
    finite = bool(np.all(np.isfinite(norms)))
    success = finite and not ended and norms[-1] < tol
    settling = None
    if success:
        outside = np.where(norms >= tol)[0]
        settling = float((outside[-1] + 1) * dt) if len(outside) else 0.0
    state_cost = float(dt * np.sum(norms[1:] ** 2)) if finite else None
    effort = float(dt * np.sum(inputs ** 2))
    return dict(success=bool(success), terminated=ended, steps_done=len(inputs),
                final_norm=float(norms[-1]) if finite else None, settling_time=settling,
                state_cost=state_cost, effort=effort,
                cost=None if state_cost is None else state_cost + EVAL_R * effort,
                max_abs_u=float(np.abs(inputs).max()) if inputs.size else 0.0)


def mpc_policy(ctrl, method, mujoco, solve_times):
    def policy(x):
        # float32 casts as in control_comparison.py (they reproduce the paper runs)
        u = np.ravel(ctrl.solve(x.astype(np.float32) if method == "kmpc_deep" else x))
        solve_times.append(ctrl.solve_time)
        if mujoco:
            return np.float32(u[0])
        return u if len(u) > 1 else float(u[0])
    return policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True, choices=list(EPISODES))
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--n-x0", type=int, default=N_X0)
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--n-chunks", type=int, default=1)
    parser.add_argument("--T", type=float, default=None, help="episode length in seconds (default as Table III)")
    parser.add_argument("--kick", type=float, default=0.0, help="mid-episode perturbation size, 0 = none")
    parser.add_argument("--tol", type=float, default=SUCCESS_TOL)
    parser.add_argument("--save-traj", action="store_true")
    parser.add_argument("--checkpoints", default=str(ROOT / "checkpoints"),
                        help="folder with our <system>.pt and <system>_region.npy")
    parser.add_argument("--out", default=str(ROOT / "runs" / "control_benchmark"))
    args = parser.parse_args()
    name, method = args.system, args.method
    if (name, method) in MISSING:
        sys.exit(f"{method} is not available for {name} (see the README)")

    mujoco = name.startswith("mujoco")
    ep = EPISODES[name]
    T = args.T or ep["T"]
    if mujoco:
        probe = experiment_system(name)
        steps = int(round(T / probe.dt))
        system = experiment_system(name, max_episode_steps=steps + 10)
    else:
        system = experiment_system(name)
        steps = int(round(T / system.dt))
    u_max = ep.get("u_max")

    x0s = initial_states(name, args.n_x0)
    idx = np.array_split(np.arange(args.n_x0), args.n_chunks)[args.chunk]
    region = np.load(Path(args.checkpoints) / f"{name}_region.npy")
    model = load_model(Path(args.checkpoints) / f"{name}.pt") if method.startswith("ours") else None
    ctrl = make_controller(name, method, system) if method in ("nmpc", "kmpc_edmd", "kmpc_deep") else None

    tag = method + (f"_kick{args.kick:g}" if args.kick else "") + (f"_T{T:g}" if args.T else "")
    out_dir = Path(args.out) / name / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{idx[0]:03d}-{idx[-1]:03d}.jsonl"
    print(f"[{name} {tag}] initial states {idx[0]}..{idx[-1]} of {args.n_x0}, "
          f"{steps} steps of {system.dt} s", flush=True)

    with open(out_file, "w") as f:
        for i in idx:
            x0 = system.reset(x0s[i]) if mujoco else x0s[i]
            kick = kick_vector(name, i, args.kick) if args.kick else None
            rec = dict(system=name, method=method, index=int(i), x0=np.asarray(x0).tolist(),
                       T=T, kick=args.kick, tol=args.tol)
            if ctrl is None:
                if method == "centroid":
                    theta, info = region.mean(axis=0), dict(t_select=0.0)
                elif method == "sim_opt":
                    theta, info = select_theta_on_simulator(region, x0, system)
                else:
                    theta, info = select_theta(model, region, x0, system, objective=method[-1])
                t0 = time.perf_counter()
                traj, inputs, ended = run_episode(system, x0, steps, lambda x: system.feedback(x, theta),
                                                  u_max, fixed_feedback=True, kick=kick)
                t_episode = time.perf_counter() - t0
                rec.update(theta=np.asarray(theta).tolist(), t_select=info["t_select"],
                           t_episode=t_episode, t_online=info["t_select"] + t_episode,
                           **{k: info[k] for k in ("success", "nit", "nfev") if k in info})
                if "success" in rec:
                    rec["slsqp_success"] = rec.pop("success")
            else:
                ctrl.reset()
                solve_times = []
                t0 = time.perf_counter()
                traj, inputs, ended = run_episode(system, x0, steps,
                                                  mpc_policy(ctrl, method, mujoco, solve_times),
                                                  u_max, kick=kick)
                t_episode = time.perf_counter() - t0
                rec.update(t_select=0.0, t_episode=t_episode, t_online=t_episode,
                           t_solve=float(np.sum(solve_times)))
            rec.update(episode_metrics(traj, inputs, system.dt, ended, args.tol))
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if args.save_traj:
                np.savez(out_dir / f"traj_{i:03d}.npz", traj=traj, inputs=inputs)
            fin = rec["final_norm"]
            print(f"  x0 {i:3d}: {'ok  ' if rec['success'] else 'FAIL'} |x_T| = "
                  f"{fin if fin is None else f'{fin:.2e}'}  t_online = {rec['t_online']:.2f} s", flush=True)
    print(f"saved {out_file}")


if __name__ == "__main__":
    main()
