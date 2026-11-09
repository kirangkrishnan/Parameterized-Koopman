"""Koopman MPC baselines with control (Table III): z+ = A z + B u, x = decoder(z).

K-MPC (EDMD): polynomial (and trig) dictionary fitted by ridge regression on
one-step transitions under random inputs; the decoder reads the linear terms.
K-MPC (Deep): MLP encoder/decoder with a learned (A, B), trained once per system
on random-input trajectories (models in checkpoints/control/).

Both are used in a receding-horizon loop: at each step the measured state is
lifted, a quadratic cost on the decoded prediction is minimized over the input
sequence with IPOPT, and the first input is applied.
"""
import time
from itertools import combinations_with_replacement, count

import casadi as ca
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp

from baselines.nmpc import ipopt_options

_ids = count()

# horizon, state weight Q, input weight R, input bound (same for EDMD and deep)
MPC_SETTINGS = {
    "vdp": dict(horizon=15, Q=np.eye(2), R=0.1, u_max=10.0),
    "slow_manifold": dict(horizon=15, Q=np.eye(2), R=0.1, u_max=10.0),
    "cartpole": dict(horizon=15, Q=np.diag([1.0, 1.0, 10.0, 1.0]), R=0.1, u_max=20.0),
    "manipulator": dict(horizon=15, Q=np.diag([15.0, 15.0, 1.0, 1.0]), R=0.1, u_max=30.0),
    "mujoco_ip": dict(horizon=15, Q=np.diag([1.0, 0.1, 10.0, 0.1]), R=0.01, u_max=3.0),
}

# EDMD dictionary and data: monomial degree, angle coordinates that get sin/cos
# terms, number of transitions, box for the sampled states, ridge parameter,
# and the bound on the next state used to discard samples
EDMD_SETTINGS = {
    "vdp": dict(degree=3, angles=(), n_samples=10000, x_scale=0.6, ridge=1e-4, keep=(slice(None), 20.0)),
    "slow_manifold": dict(degree=1, angles=(), n_samples=10000, x_scale=0.6, ridge=1e-4, keep=(slice(None), 20.0)),
    "cartpole": dict(degree=1, angles=(), n_samples=10000, x_scale=0.3, ridge=1e-4, keep=(slice(None), 20.0)),
    "manipulator": dict(degree=1, angles=(0, 1), n_samples=12000, x_scale=0.3, ridge=1e-3, keep=(slice(0, 2), 3.0)),
}


class LiftedMPC:
    """min sum_k x_k' Q x_k + u_k' R u_k + terminal_scale x_H' Q x_H with
    z_{k+1} = A z_k + B u_k, x_k = decoder(z_k), |u| <= u_max."""

    def __init__(self, A, B, decoder, nx, Q, R, u_max, horizon, terminal_scale=10.0,
                 max_iter=60, tol=1e-6, acceptable_tol=1e-3):
        A, B = np.asarray(A, np.float64), np.asarray(B, np.float64)
        self.nz, self.nu, self.H = A.shape[0], B.shape[1], horizon
        A, B = ca.DM(A), ca.DM(B)
        Q, R = ca.DM(np.asarray(Q, float)), ca.DM(R * np.eye(self.nu))
        U = ca.SX.sym("U", horizon * self.nu)
        z0 = ca.SX.sym("z0", self.nz)
        z, J = z0, 0
        for k in range(horizon):
            uk = U[k * self.nu:(k + 1) * self.nu]
            z = A @ z + B @ uk
            x = decoder(z)
            J += ca.mtimes([x.T, Q, x])
            J += ca.mtimes([uk.T, R, uk])
        x = decoder(z)
        J += float(terminal_scale) * ca.mtimes([x.T, Q, x])
        self.solver = ca.nlpsol(f"kmpc_{next(_ids)}", "ipopt", {"x": U, "p": z0, "f": J},
                                ipopt_options(max_iter, tol, acceptable_tol))
        self.u_max = u_max
        self.lbx = np.full(horizon * self.nu, -float(u_max))
        self.ubx = np.full(horizon * self.nu, float(u_max))
        self.reset()

    def reset(self):
        self.u_warm = np.zeros((self.H, self.nu))
        self.lam_x = None
        self.solve_time = 0.0

    def solve(self, z0):
        previous = self.u_warm.copy()
        args = dict(x0=previous.reshape(-1), p=np.asarray(z0, np.float64), lbx=self.lbx, ubx=self.ubx)
        if self.lam_x is not None:
            args["lam_x0"] = self.lam_x
        t0 = time.perf_counter()
        sol = self.solver(**args)
        self.solve_time = time.perf_counter() - t0
        U = np.asarray(sol["x"], float).reshape(self.H, self.nu)
        if self.solver.stats()["success"] and np.all(np.isfinite(U)):
            self.lam_x = np.asarray(sol["lam_x"], float).reshape(-1)
        else:
            U, self.lam_x = previous, None
        self.u_warm[:-1] = U[1:]
        self.u_warm[-1] = 0.0
        return np.clip(U[0], -self.u_max, self.u_max)


class KoopmanMPC:
    """Receding-horizon controller: lift the measured state, solve, return u_0."""

    def __init__(self, lift, A, B, decoder, nx, settings):
        self.lift = lift
        s = settings
        self.mpc = LiftedMPC(A, B, decoder, nx, s["Q"], s["R"], s["u_max"], s["horizon"])

    def reset(self):
        self.mpc.reset()

    @property
    def solve_time(self):
        return self.mpc.solve_time

    def solve(self, x):
        u = self.mpc.solve(self.lift(x))
        return np.clip(u, -self.mpc.u_max, self.mpc.u_max)


# EDMD

class Dictionary:
    """Monomials of degree 1..degree (state coordinates first), then sin(x_i) and
    cos(x_i) - 1 for the listed angle coordinates. No constant term."""

    def __init__(self, n, degree, angles=()):
        self.n, self.angles = n, tuple(angles)
        self.powers = []
        for d in range(1, degree + 1):
            for combo in combinations_with_replacement(range(n), d):
                idx = [0] * n
                for i in combo:
                    idx[i] += 1
                self.powers.append(np.array(idx))
        self.dim = len(self.powers) + 2 * len(self.angles)

    def __call__(self, x):
        psi = [np.prod(x ** p) for p in self.powers]
        psi += [np.sin(x[i]) for i in self.angles] + [np.cos(x[i]) - 1.0 for i in self.angles]
        return np.array(psi)


def collect_transitions(system, settings, u_max, nu=1, seed=0):
    """One-step transitions of the open-loop plant from random states and random
    constant inputs (integrated like the closed-loop runs, RK45, max_step dt/4)."""
    s = settings
    # same stream as np.random.seed(seed) in the paper runs, so the saved matrices are reproduced
    rng = np.random.RandomState(seed)
    lift = Dictionary(system.n, s["degree"], s["angles"])
    idx, bound = s["keep"]
    Psi, Psi_next, U = [], [], []
    for _ in range(5 * s["n_samples"]):
        if len(Psi) >= s["n_samples"]:
            break
        x = rng.uniform(-s["x_scale"], s["x_scale"], size=system.n)
        u = float(rng.uniform(-u_max, u_max)) if nu == 1 else rng.uniform(-u_max, u_max, nu)
        f = lambda t, y: system.f(y, np.clip(u, -u_max, u_max))
        x_next = solve_ivp(f, [0, system.dt], x, max_step=system.dt / 4).y[:, -1]
        if not np.all(np.isfinite(x_next)) or np.any(np.abs(x_next[idx]) > bound):
            continue
        Psi.append(lift(x))
        Psi_next.append(lift(x_next))
        U.append(np.atleast_1d(u))
    return np.array(Psi), np.array(Psi_next), np.array(U)


def fit_edmd(Psi, Psi_next, U, ridge):
    """Ridge least squares for Psi_next ~ K Psi + Ku U. Returns K, Ku."""
    Phi = np.hstack([Psi, U])
    W = np.linalg.solve(Phi.T @ Phi + ridge * np.eye(Phi.shape[1]), Phi.T @ Psi_next).T
    d = Psi.shape[1]
    return W[:, :d], W[:, d:]


def edmd_controller(name, system, K, Ku):
    s = EDMD_SETTINGS[name]
    lift = Dictionary(system.n, s["degree"], s["angles"])
    n = system.n
    return KoopmanMPC(lift, K, Ku, lambda z: z[:n], n, MPC_SETTINGS[name])


# Deep Koopman with control

class DeepKoopmanControl(nn.Module):
    """phi, psi: MLPs with two tanh layers of 64 units; z+ = A z + B u."""

    def __init__(self, n, nu, dz):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
                                     nn.Linear(64, dz))
        self.decoder = nn.Sequential(nn.Linear(dz, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
                                     nn.Linear(64, n))
        self.A = nn.Parameter(torch.zeros(dz, dz))
        self.B = nn.Parameter(torch.zeros(dz, nu))


def load_deep_koopman(path):
    state = torch.load(path, map_location="cpu")
    dz, nu = state["B"].shape
    model = DeepKoopmanControl(state["encoder.0.weight"].shape[1], nu, dz)
    model.load_state_dict(state)
    return model.eval()


def casadi_mlp(seq):
    """CasADi version of a torch Sequential of Linear and Tanh layers."""
    layers = [(ca.DM(m.weight.detach().numpy().astype(np.float64)),
               ca.DM(m.bias.detach().numpy().astype(np.float64)))
              for m in seq if isinstance(m, nn.Linear)]

    def f(z):
        for i, (W, b) in enumerate(layers):
            z = W @ z + b
            if i < len(layers) - 1:
                z = ca.tanh(z)
        return z
    return f


def deep_controller(name, model):
    def lift(x):
        # the encoder runs in float32 on the float32-cast state
        with torch.no_grad():
            return model.encoder(torch.tensor(x, dtype=torch.float32)[None])[0].numpy()
    n = model.decoder[-1].out_features
    return KoopmanMPC(lift, model.A.detach().numpy(), model.B.detach().numpy(),
                      casadi_mlp(model.decoder), n, MPC_SETTINGS[name])
