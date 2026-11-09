import numpy as np

from .base import ODESystem, grid

M_CART, M_POLE, L, G = 1.0, 0.1, 0.5, 9.81
U_MAX = 20.0

# theta in [-1, 1]^4 is mapped linearly onto this box of physical gains
GAIN_LOW = np.array([-5.0, -20.0, -110.0, -35.0])
GAIN_HIGH = np.array([5.0, 2.0, -20.0, -3.0])


def to_gains(theta):
    return GAIN_LOW + 0.5 * (np.asarray(theta) + 1.0) * (GAIN_HIGH - GAIN_LOW)


class Cartpole(ODESystem):
    """Cart-pole with the pole angle measured from upright, state [p, pd, phi, phid].
    u = -(k1 p + k2 pd + k3 sin(phi) + k4 phid), k = to_gains(theta), |u| <= 20."""
    name = "cartpole"
    n, p = 4, 4
    dt = 0.05
    theta_low, theta_high = -np.ones(4), np.ones(4)
    x0_low = -np.array([0.5, 0.5, 0.3, 0.8])
    x0_high = np.array([0.5, 0.5, 0.3, 0.8])

    def __init__(self, u_max=U_MAX):
        self.u_max = u_max

    def f(self, x, u):
        s, c = np.sin(x[2]), np.cos(x[2])
        total = M_CART + M_POLE
        tmp = (u + M_POLE * L * x[3] ** 2 * s) / total
        phidd = (G * s - c * tmp) / (L * (4.0 / 3.0 - M_POLE * c ** 2 / total))
        pdd = tmp - M_POLE * L * phidd * c / total
        return [x[1], pdd, x[3], phidd]

    def feedback(self, x, theta):
        k = to_gains(theta)
        u = -(k[0] * x[0] + k[1] * x[1] + k[2] * np.sin(x[2]) + k[3] * x[3])
        return u if self.u_max is None else np.clip(u, -self.u_max, self.u_max)

    def test_initial_conditions(self):
        return grid(self.x0_low, self.x0_high, [3, 3, 3, 3])

    def in_bounds(self, traj):
        return traj is not None and np.abs(traj[:, 0]).max() <= 10.0 and np.abs(traj[:, 2]).max() <= np.deg2rad(20)
