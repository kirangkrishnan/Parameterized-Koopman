"""Closed-loop control comparison (Q3, Table III, Fig. 8).

Runs one method on one system from the fixed initial state of Table III and saves
the trajectory, the applied inputs and the timings to
runs/control_comparison/<system>_<method>_seed<seed>.npz (+ .json).

  python scripts/control_comparison.py --system vdp --method ours_z
  python scripts/control_comparison.py --system mujoco_ip --method nmpc --steps 100

Methods: ours_z, ours_x (one-shot selection, then fixed feedback), nmpc,
kmpc_edmd, kmpc_deep. All pipelines are deterministic; the seed only names the
output (the paper averages wall-clock times over seeds 8, 17, 42, 0, 99).

t_tot follows the rule used for Table III: Ours = wall time of the closed-loop
episode (the selection time is stored separately as t_select), NMPC = sum of
the optimizer calls, K-MPC = wall time of the whole episode.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.integrate import solve_ivp

from pkoopman.model import load_model
from pkoopman.selection import select_theta
from pkoopman.systems import get_system

# initial state, episode length in seconds, input bound of the plant
EPISODES = {
    "vdp": dict(x0=[1.0, 1.0], T=20.0, u_max=10.0),
    "slow_manifold": dict(x0=[1.0, 1.0], T=20.0, u_max=10.0),
    "manipulator": dict(x0=[0.3, -0.2, 0.0, 0.0], T=20.0, u_max=30.0),
    "cartpole": dict(x0=[0.0, 0.0, 0.1, 0.0], T=20.0, u_max=20.0),
    "mujoco_ip": dict(x0=[0.0, 0.0, 0.05, 0.0], T=20.0),
    "mujoco_idp": dict(x0=[0.0, 0.0, 0.05, 0.0, 0.05, 0.0], T=20.0),
}
METHODS = ["ours_z", "ours_x", "nmpc", "kmpc_edmd", "kmpc_deep"]
# entries of Table III that have no code here (N/E, terminated, or lost script)
MISSING = {("mujoco_ip", "kmpc_edmd"), ("mujoco_idp", "kmpc_edmd"),
           ("mujoco_idp", "kmpc_deep"), ("mujoco_idp", "nmpc")}


def run_ode(system, x0, steps, u_max, policy, feedback_inside=False):
    """Analytic plant, one solve_ivp call (RK45, max_step dt/4) per step. For the
    fixed feedback laws the feedback is evaluated inside the integrator;
    otherwise the input is held constant over the step."""
    x = np.array(x0, dtype=np.float64)
    traj, inputs = [x.copy()], []
    for _ in range(steps):
        u = np.clip(policy(x), -u_max, u_max)
        if feedback_inside:
            rhs = lambda t, y: system.f(y, np.clip(policy(y), -u_max, u_max))
        else:
            rhs = lambda t, y: system.f(y, u)
        x = solve_ivp(rhs, [0.0, system.dt], x, max_step=system.dt / 4).y[:, -1]
        traj.append(x.copy())
        inputs.append(np.atleast_1d(u).astype(float))
        if not np.all(np.isfinite(x)):
            break
    traj += [x.copy()] * (steps + 1 - len(traj))
    inputs += [np.zeros_like(inputs[0])] * (steps - len(inputs))
    return np.array(traj), np.squeeze(np.array(inputs))


def run_episode(system, x0, steps, policy, u_max=None, feedback_inside=False):
    if system.name.startswith("mujoco"):
        traj, inputs, _ = system.rollout(x0, policy, steps)
        return traj, inputs
    return run_ode(system, x0, steps, u_max, policy, feedback_inside)


def make_controller(name, method, system, refit_edmd=False):
    from baselines import koopman_mpc, nmpc
    ckpt = ROOT / "checkpoints" / "control"
    if method == "nmpc":
        if name == "mujoco_ip":
            return nmpc.InvertedPendulumNMPC(system.env)
        return nmpc.analytic_nmpc(name, system.dt)
    if method == "kmpc_edmd":
        if refit_edmd:
            s = koopman_mpc.EDMD_SETTINGS[name]
            nu = 2 if name == "manipulator" else 1
            data = koopman_mpc.collect_transitions(system, s, EPISODES[name]["u_max"], nu)
            K, Ku = koopman_mpc.fit_edmd(*data, s["ridge"])
        else:
            mats = np.load(ckpt / f"edmd_{name}.npz")
            K, Ku = mats["K"], mats["Ku"]
        return koopman_mpc.edmd_controller(name, system, K, Ku)
    model = koopman_mpc.load_deep_koopman(ckpt / f"deep_koopman_{name}.pt")
    return koopman_mpc.deep_controller(name, model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True, choices=list(EPISODES))
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--steps", type=int, default=None, help="run only the first steps")
    parser.add_argument("--refit-edmd", action="store_true",
                        help="fit the EDMD matrices from data instead of loading them")
    parser.add_argument("--out", default=str(ROOT / "runs" / "control_comparison"))
    args = parser.parse_args()
    name, method = args.system, args.method
    if (name, method) in MISSING:
        sys.exit(f"{method} is not available for {name} (see the README)")

    np.random.seed(args.seed)
    system = get_system(name)
    ep = EPISODES[name]
    steps = args.steps or int(round(ep["T"] / system.dt))
    mujoco = name.startswith("mujoco")
    x0 = system.reset(ep["x0"]) if mujoco else np.array(ep["x0"])
    meta = dict(system=name, method=method, seed=args.seed, steps=steps, dt=system.dt)
    extra = {}

    if method.startswith("ours"):
        model = load_model(ROOT / "checkpoints" / f"{name}.pt")
        region = np.load(ROOT / "checkpoints" / f"{name}_region.npy")
        theta, info = select_theta(model, region, x0, system, objective=method[-1])
        print(f"{name} {method}: theta = {np.round(theta, 4)} (SLSQP success {info['success']}, "
              f"{info['nit']} iterations, {info['t_select']:.2f} s)")
        t0 = time.perf_counter()
        traj, inputs = run_episode(system, x0, steps, lambda x: system.feedback(x, theta),
                                   ep.get("u_max"), feedback_inside=True)
        t_episode = time.perf_counter() - t0
        meta.update(t_select=info["t_select"], slsqp_success=info["success"], t_tot=t_episode)
        extra["theta"] = theta
        t_solve = 0.0
    else:
        ctrl = make_controller(name, method, system, args.refit_edmd)
        solve_times = []

        def policy(x):
            # the float32 casts reproduce what the paper runs fed to the
            # deep encoder and to the MuJoCo IP environment
            u = np.ravel(ctrl.solve(x.astype(np.float32) if method == "kmpc_deep" else x))
            solve_times.append(ctrl.solve_time)
            if mujoco:
                return np.float32(u[0])
            return u if len(u) > 1 else float(u[0])

        t0 = time.perf_counter()
        traj, inputs = run_episode(system, x0, steps, policy, ep.get("u_max"))
        t_episode = time.perf_counter() - t0
        t_solve = float(np.sum(solve_times))
        meta["t_tot"] = t_solve if method == "nmpc" else t_episode

    meta.update(t_episode=t_episode, t_solve=t_solve,
                stabilized=bool(np.all(np.isfinite(traj[-1]))),
                final_norm=float(np.linalg.norm(traj[-1])))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{name}_{method}_seed{args.seed}"
    np.savez(stem.with_suffix(".npz"), traj=traj, inputs=inputs, **extra)
    stem.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"{name} {method} seed {args.seed}: t_tot = {meta['t_tot']:.3f} s, "
          f"|x_T| = {meta['final_norm']:.2e}, saved {stem}.npz")


if __name__ == "__main__":
    main()
