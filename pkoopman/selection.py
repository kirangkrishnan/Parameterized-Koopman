"""One-shot selection of the feedback parameter (Algorithm 2 in the paper).

For the observed x0 we minimize a surrogate cost of the Koopman rollout over theta,
subject to theta lying in the convex hull of the verified samples and in the box.
The cost is evaluated in float64 numpy and SLSQP uses its default finite-difference
gradient.
"""
import time

import numpy as np
import torch
from scipy.optimize import LinearConstraint, minimize
from scipy.spatial import ConvexHull

# Per-system settings that were used for Table III.
#   horizon       selection horizon H_sel (model steps)
#   q_z, q_x      weight on sum ||z_k||^2 (Ours-Z) or sum ||x_hat_k||^2 (Ours-X)
#   x_weights     optional diagonal weights inside the decoded-state sum
#   r_theta       extra weight on ||theta||^2
#   ftol, maxiter SLSQP options (maxiter 100 is the scipy default)
#   action        MuJoCo only: penalties on the feedback input along the predicted
#                 trajectory (weight on sum u^2, input limit for the saturation
#                 penalty, weights on the gains squared)
SETTINGS = {
    "vdp":           dict(horizon=12, q_z=50.0, q_x=50.0, r_theta=0.0, ftol=1e-4, maxiter=100),
    "slow_manifold": dict(horizon=12, q_z=1.0, q_x=1.0, r_theta=0.01, ftol=1e-4, maxiter=100),
    "manipulator":   dict(horizon=400, q_z=15.0, q_x=15.0, r_theta=0.1, ftol=1e-4, maxiter=100),
    "cartpole":      dict(horizon=200, q_z=15.0, q_x=15.0, r_theta=0.0, ftol=1e-5, maxiter=100),
    "mujoco_ip":     dict(horizon=200, q_z=1000.0, q_x=15.0, x_weights=(1.0, 1.0, 1.0, 1.0),
                          r_theta=0.0, ftol=1e-4, maxiter=300,
                          action=dict(r_u=10.0, u_limit=3.0, gain_weights=(0.1, 0.1, 0.01, 0.5))),
    "mujoco_idp":    dict(horizon=12, q_z=15.0, q_x=15.0, r_theta=0.0, ftol=1e-4, maxiter=300,
                          action=dict(r_u=10.0, u_limit=1.0,
                                      gain_weights=(0.1, 0.1, 0.05, 0.5, 0.05, 0.5))),
}


class NumpyModel:
    """Copy of a PolicyKoopman for evaluation inside the optimizer. The float32
    weights are kept and every product is done in float64."""

    def __init__(self, model):
        def linear_layers(seq):
            return [(m.weight.detach().cpu().numpy(), m.bias.detach().cpu().numpy())
                    for m in seq if isinstance(m, torch.nn.Linear)]
        self.enc = linear_layers(model.encoder)
        self.dec = linear_layers(model.decoder)
        self.op = linear_layers(model.operator)
        self.theta_scale = model.theta_scale.cpu().numpy()
        self.dz = model.dz

    @staticmethod
    def _mlp(layers, h, act):
        for i, (W, b) in enumerate(layers):
            h = h @ W.T + b
            if i < len(layers) - 1:
                h = act(h)
        return h

    def encode(self, x, theta):
        th = theta / self.theta_scale
        return (self._mlp(self.enc, np.concatenate([x, th]), np.tanh)
                - self._mlp(self.enc, np.concatenate([np.zeros_like(x), th]), np.tanh))

    def decode(self, Z, theta):
        T = np.tile(theta / self.theta_scale, (Z.shape[0], 1))
        return (self._mlp(self.dec, np.concatenate([Z, T], axis=1), np.tanh)
                - self._mlp(self.dec, np.concatenate([np.zeros_like(Z), T], axis=1), np.tanh))

    def K(self, theta):
        out = self._mlp(self.op, theta / self.theta_scale, lambda h: np.maximum(0, h))
        return out[:self.dz * self.dz].reshape(self.dz, self.dz)

    def latent_rollout(self, x0, theta, horizon):
        z = self.encode(x0, theta)
        K = self.K(theta)
        Z = np.empty((horizon, self.dz))
        for k in range(horizon):
            z = K @ z
            Z[k] = z
        return Z


def hull_constraint(region):
    """A theta <= b from the facets of the convex hull of the verified samples."""
    hull = ConvexHull(region)
    A, b = hull.equations[:, :-1], hull.equations[:, -1]
    return LinearConstraint(A, np.full(len(b), -np.inf), -b)


def state_cost(X, s):
    """Weighted sum of squared states along a trajectory X (H, n)."""
    if "x_weights" in s:
        # diagonal weighting, summed coordinate by coordinate
        return s["q_x"] * sum(w * np.sum(X[:, i] ** 2) for i, w in enumerate(s["x_weights"]))
    return s["q_x"] * np.sum(X ** 2)


def add_extra_terms(cost, X, system, x0, theta, s):
    """Terms shared by every objective: gain size and, for MuJoCo, the input penalties."""
    cost = cost + s["r_theta"] * np.sum(theta ** 2)
    if "action" in s:
        for term in action_penalties(X, system, x0, theta, s["action"]):
            cost = cost + term
    return cost


def surrogate_cost(net, system, x0, theta, objective, s):
    Z = net.latent_rollout(x0, theta, s["horizon"])
    X = net.decode(Z, theta) if objective == "x" or "action" in s else None
    cost = s["q_z"] * np.sum(Z ** 2) if objective == "z" else state_cost(X, s)
    return add_extra_terms(cost, X, system, x0, theta, s)


# cost assigned to a candidate whose true closed loop fails (diverges or terminates)
FAILED_ROLLOUT_COST = 1e6


def simulator_cost(system, x0, theta, s):
    """The decoded objective evaluated on the true closed loop instead of the model."""
    traj = system.simulate(x0, theta, s["horizon"])
    if traj is None or not np.all(np.isfinite(traj)):
        return FAILED_ROLLOUT_COST
    X = traj[1:]
    return add_extra_terms(state_cost(X, s), X, system, x0, theta, s)


def action_penalties(X, system, x0, theta, a):
    """Input effort, saturation, first-input and gain-size terms (MuJoCo systems)."""
    gains = system.to_gains(theta)
    u = system.raw_feedback(X, gains)
    over = np.maximum(np.abs(u) - a["u_limit"], 0.0)
    u0 = system.raw_feedback(x0, gains)
    gain_cost = sum(w * g ** 2 for w, g in zip(a["gain_weights"], gains))
    return [a["r_u"] * np.sum(u ** 2), 1000.0 * np.sum(over ** 2), 20.0 * u0 ** 2, gain_cost]


def _minimize(cost, region, system, s):
    theta0 = np.mean(region, axis=0)
    bounds = list(zip(system.theta_low, system.theta_high))
    t0 = time.perf_counter()
    res = minimize(cost, theta0, method="SLSQP", bounds=bounds,
                   constraints=[hull_constraint(region)],
                   options={"ftol": s["ftol"], "maxiter": s["maxiter"]})
    t_select = time.perf_counter() - t0
    theta = res.x if res.success else theta0
    return theta, dict(success=bool(res.success), nit=int(res.nit), nfev=int(res.nfev),
                       cost=float(res.fun), t_select=t_select)


def select_theta(model, region, x0, system, objective="x", settings=None):
    """Solve the selection problem once. Returns (theta, info). If SLSQP reports
    failure the mean of the verified samples is returned, as in the paper runs."""
    s = settings or SETTINGS[system.name]
    net = NumpyModel(model)
    x0 = np.asarray(x0, dtype=np.float64)
    return _minimize(lambda th: surrogate_cost(net, system, x0, th, objective, s), region, system, s)


def select_theta_on_simulator(region, x0, system, settings=None):
    """Reference for the surrogate: the same problem (decoded objective, horizon,
    hull constraint, optimizer settings) with every cost evaluation done by
    simulating the true closed loop. This is the theta^opt of the paper restricted
    to C, up to the local nature of SLSQP."""
    s = settings or SETTINGS[system.name]
    x0 = np.asarray(x0, dtype=np.float64)
    return _minimize(lambda th: simulator_cost(system, x0, th, s), region, system, s)
