import numpy as np
from scipy.integrate import solve_ivp

from .base import ODESystem, grid

M1, M2, L1, L2, G = 1.0, 1.0, 1.0, 1.0, 9.81
TAU_MAX = 30.0


class Manipulator(ODESystem):
    """Two-link planar arm with point masses, angles measured from upright.
    Decentralized PD feedback tau_i = -(kp_i q_i + kd_i qd_i), saturated at 30 Nm.
    theta = [kp1, kd1, kp2, kd2]."""
    name = "manipulator"
    n, p = 4, 4
    dt = 0.05
    theta_low = np.zeros(4)
    theta_high = np.array([40.0, 20.0, 20.0, 20.0])
    theta_scale = theta_high          # the model sees theta / theta_scale
    x0_low = -np.array([0.35, 0.35, 1.0, 1.0])
    x0_high = np.array([0.35, 0.35, 1.0, 1.0])
    state_limit = 3.0

    def f(self, x, tau):
        q1, q2, qd1, qd2 = x
        d = q1 - q2
        M = np.array([[(M1 + M2) * L1 ** 2, M2 * L1 * L2 * np.cos(d)],
                      [M2 * L1 * L2 * np.cos(d), M2 * L2 ** 2]])
        b = np.array([M2 * L1 * L2 * np.sin(d) * qd2 ** 2 - (M1 + M2) * G * L1 * np.sin(q1),
                      -M2 * L1 * L2 * np.sin(d) * qd1 ** 2 - M2 * G * L2 * np.sin(q2)])
        qdd = np.linalg.solve(M, np.asarray(tau) - b)
        return [qd1, qd2, qdd[0], qdd[1]]

    def feedback(self, x, theta):
        kp1, kd1, kp2, kd2 = theta
        tau1 = -(kp1 * x[0] + kd1 * x[2])
        tau2 = -(kp2 * x[1] + kd2 * x[3])
        return np.clip([tau1, tau2], -TAU_MAX, TAU_MAX)

    def test_initial_conditions(self):
        return grid(self.x0_low, self.x0_high, [3, 3, 2, 2])

    def converges(self, theta, T=6.0, tol=0.15):
        # with too little gain the arm settles at a second, bounded equilibrium
        # away from upright, so "stays bounded" is not enough
        x0 = np.array([0.35, 0.35, 1.0, 1.0])
        sol = solve_ivp(self.closed_loop, (0.0, T), x0, args=(theta,))
        return sol.success and np.abs(sol.y[:, -1]).max() < tol

    def is_valid(self, theta, trajectories):
        if any(tr is None or np.abs(tr).max() > self.state_limit for tr in trajectories):
            return False
        return self.converges(theta)
