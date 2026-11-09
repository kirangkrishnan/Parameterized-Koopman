import numpy as np

from .base import ODESystem, grid


class VanDerPol(ODESystem):
    """Controlled Van der Pol oscillator, x1' = x2, x2' = (1 - x1^2) x2 - x1 + u,
    with u = -(th1 x1 + th2 x2 + th3 x1^2 x2 + th4)."""
    name = "vdp"
    n, p = 2, 4
    dt = 0.05
    theta_low = np.array([0.0, 0.0, -1.0, -5.0])
    theta_high = np.array([5.0, 5.0, 1.0, 5.0])
    x0_low, x0_high = -np.ones(2), np.ones(2)

    def f(self, x, u):
        return [x[1], (1 - x[0] ** 2) * x[1] - x[0] + u]

    def feedback(self, x, theta):
        return -(theta[0] * x[0] + theta[1] * x[1] + theta[2] * x[0] ** 2 * x[1] + theta[3])

    def test_initial_conditions(self):
        return grid([-1, -1], [1, 1], [5, 5])
