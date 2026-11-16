import numpy as np

from .base import ODESystem, grid

MU, LAM = -0.1, -1.0


class SlowManifold(ODESystem):
    """Polynomial system with the attracting manifold x2 = x1^2,
    x1' = mu x1 + u, x2' = lam (x2 - x1^2), with the same feedback family as VdP
    except for the last term: u = -(th1 x1 + th2 x2 + th3 x1^3 + th4 x1^2 x2)."""
    name = "slow_manifold"
    n, p = 2, 4
    dt = 0.05
    theta_low = np.array([0.0, 0.0, -1.0, -1.0])
    theta_high = np.array([5.0, 5.0, 1.0, 1.0])
    x0_low, x0_high = -2 * np.ones(2), 2 * np.ones(2)
    method = "Radau"
    solver_options = dict(max_step=0.05, rtol=1e-3, atol=1e-5)

    def f(self, x, u):
        return [MU * x[0] + u, LAM * (x[1] - x[0] ** 2)]

    def feedback(self, x, theta):
        return -(theta[0] * x[0] + theta[1] * x[1] + theta[2] * x[0] ** 3 + theta[3] * x[0] ** 2 * x[1])

    def test_initial_conditions(self):
        return grid([-2, -2], [2, 2], [3, 3])

    def is_valid(self, theta, trajectories):
        return all(tr is not None and np.abs(tr).max() <= 15.0 for tr in trajectories)
