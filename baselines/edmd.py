"""EDMD with a monomial dictionary and Kernel-EDMD with Gaussian RBF features.

Both are fitted by ridge regression on trajectories at one fixed theta (the centroid
of the verified region) and ignore theta when predicting. Column vectors,
psi_{k+1} = K psi_k.
"""
from itertools import combinations_with_replacement

import numpy as np


def ridge_operator(P_now, P_next, ridge):
    """K minimizing ||P_next - P_now K^T||^2 + ridge ||K||^2."""
    A = P_now.T @ P_now + ridge * np.eye(P_now.shape[1])
    return np.linalg.solve(A, P_now.T @ P_next).T


def transitions(trajs):
    now = np.vstack([tr[:-1] for tr in trajs])
    nxt = np.vstack([tr[1:] for tr in trajs])
    return now, nxt


class EDMD:
    """Dictionary: all monomials of degree 1..degree, then sin(x_i), cos(x_i) - 1
    for every index in trig. The first n monomials are x itself, which is the
    state readout."""

    def __init__(self, n, degree=3, trig=(), ridge=1e-3):
        self.n, self.trig, self.ridge = n, tuple(trig), ridge
        exps = []
        for d in range(1, degree + 1):
            for combo in combinations_with_replacement(range(n), d):
                e = np.zeros(n, dtype=int)
                for i in combo:
                    e[i] += 1
                exps.append(e)
        self.exponents = np.array(exps)

    def lift(self, x):
        x = np.asarray(x, dtype=float)
        feats = [np.prod(x[..., None, :] ** self.exponents, axis=-1)]
        for i in self.trig:
            feats += [np.sin(x[..., i:i + 1]), np.cos(x[..., i:i + 1]) - 1.0]
        return np.concatenate(feats, axis=-1)

    def fit(self, trajs):
        now, nxt = transitions(trajs)
        self.K = ridge_operator(self.lift(now), self.lift(nxt), self.ridge)
        return self

    def predict(self, x0, steps):
        psi = self.lift(x0)
        preds = []
        for _ in range(steps):
            psi = self.K @ psi
            preds.append(psi[:self.n])
        return np.array(preds)


class KernelEDMD:
    """Gaussian RBF features exp(-gamma ||x - c_j||^2) on centers drawn from the
    training states, with a linear readout x = C psi fitted by ridge regression.
    gamma defaults to 1 / median squared distance between training states."""

    def __init__(self, n_centers=80, gamma=None, ridge=1e-3, seed=0):
        self.n_centers, self.gamma, self.ridge, self.seed = n_centers, gamma, ridge, seed

    def lift(self, x):
        d2 = np.sum((self.centers - x[..., None, :]) ** 2, axis=-1)
        return np.exp(-self.gamma * d2)

    def fit(self, trajs):
        now, nxt = transitions(trajs)
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(now), min(self.n_centers, len(now)), replace=False)
        self.centers = now[idx]
        if self.gamma is None:
            d2 = np.sum((now[:, None, :] - now[None, :, :]) ** 2, axis=2)
            self.gamma = 1.0 / max(np.median(d2[d2 > 0]), 1e-8)

        P_now, P_next = self.lift(now), self.lift(nxt)
        self.K = ridge_operator(P_now, P_next, self.ridge)
        A = P_now.T @ P_now + self.ridge * np.eye(P_now.shape[1])
        self.C = now.T @ P_now @ np.linalg.inv(A)
        return self

    def predict(self, x0, steps):
        psi = self.lift(x0)
        preds = []
        for _ in range(steps):
            psi = self.K @ psi
            preds.append(self.C @ psi)
        return np.array(preds)
