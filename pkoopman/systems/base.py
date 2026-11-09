import numpy as np
from scipy.integrate import solve_ivp


class ODESystem:
    """A plant x' = f(x, u) closed with a parameterized feedback u = k(x, theta).

    Subclasses set the class attributes and implement f() and feedback().
    theta always means the vector the model sees (for the cartpole this is a
    normalized vector that gets mapped to physical gains inside feedback()).
    """
    name = ""
    n = p = 0
    dt = 0.05
    theta_low = theta_high = None      # box Theta for the feedback parameter
    x0_low = x0_high = None            # box for training initial conditions
    method = "RK45"
    solver_options = {}

    def f(self, x, u):
        raise NotImplementedError

    def feedback(self, x, theta):
        raise NotImplementedError

    def closed_loop(self, t, x, theta):
        return self.f(x, self.feedback(x, theta))

    def simulate(self, x0, theta, steps, **options):
        """Integrate the closed loop and return the states at k*dt, shape (steps+1, n).
        Returns None if the integrator fails."""
        t_eval = np.linspace(0.0, steps * self.dt, steps + 1)
        opts = dict(self.solver_options, **options)
        sol = solve_ivp(self.closed_loop, (0.0, steps * self.dt), np.asarray(x0, float),
                        t_eval=t_eval, method=self.method, args=(np.asarray(theta, float),), **opts)
        if not sol.success or sol.y.shape[1] != steps + 1:
            return None
        return sol.y.T

    def sample_theta(self, rng, size):
        return rng.uniform(self.theta_low, self.theta_high, size=(size, self.p))

    def sample_x0(self, rng, size):
        return rng.uniform(self.x0_low, self.x0_high, size=(size, self.n))

    def test_initial_conditions(self):
        """Grid of initial conditions used when the region is built."""
        raise NotImplementedError

    def is_valid(self, theta, trajectories):
        """Extra check used during region construction (true closed loop is sane)."""
        return all(tr is not None for tr in trajectories)


def grid(lows, highs, counts):
    axes = [np.linspace(lo, hi, c) for lo, hi, c in zip(lows, highs, counts)]
    return np.array(np.meshgrid(*axes, indexing="ij")).reshape(len(axes), -1).T
