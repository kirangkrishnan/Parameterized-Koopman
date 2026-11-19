"""Gymnasium MuJoCo InvertedPendulum-v5 and InvertedDoublePendulum-v5 as
parameterized closed-loop systems.

There is no closed-form f here; every transition is one env.step(), so dt is the
environment step (frame_skip * model timestep): 0.04 s for the pendulum and
0.05 s for the double pendulum. theta is a normalized vector in [-1, 1]^p that
is mapped linearly onto a box of physical gains, as for the analytic cartpole.
"""
import numpy as np
import gymnasium as gym

# Two gain boxes for the inverted pendulum. The model was trained, and its region
# built, with the "training" box. The control runs of Table 3 used the wider
# "control" box, which is the default so that those runs are reproduced.
IP_GAIN_BOXES = {
    "training": (np.array([-1.0, -3.0, -25.0, -4.0]), np.array([1.0, 1.0, -5.0, -0.5])),
    "control": (np.array([-2.0, -5.0, -40.0, -10.0]), np.array([2.0, 2.0, -5.0, -1.0])),
}
IP_GAIN_LOW, IP_GAIN_HIGH = IP_GAIN_BOXES["control"]

# Double pendulum: a box around a scaled LQR gain
IDP_GAIN_LOW = np.array([0.07674178187183751, 0.16996379747877746, 1.187511819561223,
                         0.6836618963795779, 4.841099575184763, 0.7238104554244518])
IDP_GAIN_HIGH = np.array([0.11511267280775626, 0.25494569621816615, 1.7812677293418346,
                          1.0254928445693667, 7.261649362777143, 1.0857156831366774])


class MujocoSystem:
    """Common part. Subclasses define the state <-> (qpos, qvel, obs) maps and
    the feature vector of the linear feedback u = -gains . features(x)."""
    name = ""
    env_id = ""
    n = p = 0
    gain_low = gain_high = None
    x0_low = x0_high = None

    def __init__(self, max_episode_steps=None):
        # Gymnasium stops these environments after 1000 steps by default; pass a
        # larger value for longer episodes
        kwargs = {} if max_episode_steps is None else dict(max_episode_steps=max_episode_steps)
        self.env = gym.make(self.env_id, **kwargs)
        self.env.reset(seed=0)
        self.dt = float(self.env.unwrapped.dt)
        self.u_max = float(self.env.action_space.high[0])
        self.theta_low, self.theta_high = -np.ones(self.p), np.ones(self.p)

    def to_gains(self, theta):
        return self.gain_low + 0.5 * (np.asarray(theta) + 1.0) * (self.gain_high - self.gain_low)

    def raw_feedback(self, x, gains):
        """Unclipped u = -(g1 f1(x) + ... + gp fp(x)); x may be (n,) or (H, n)."""
        x = np.asarray(x)
        feats = self.features(x)
        u = gains[0] * feats[0]
        for g, f in zip(gains[1:], feats[1:]):
            u = u + g * f
        return -u

    def reset(self, x0):
        """Put the simulator in state x0 and return the measured state."""
        self.env.reset()
        qpos, qvel = self.to_qpos_qvel(np.asarray(x0, dtype=np.float64))
        self.env.unwrapped.set_state(qpos, qvel)
        return self.from_obs(self.env.unwrapped._get_obs())

    def step(self, u):
        """Apply action u for one env step. Returns (next state, episode ended)."""
        obs, _, terminated, truncated, _ = self.env.step(np.atleast_1d(u))
        return self.from_obs(obs), terminated or truncated

    def rollout(self, x0, policy, steps):
        """Run policy(x) -> u from x0. After termination the last state is repeated
        and the input is set to zero. Returns states (steps+1, n), inputs (steps,)
        and the number of steps done."""
        x = self.reset(x0)
        traj, inputs = [x.copy()], []
        done = steps
        for k in range(steps):
            u = policy(x)
            x, ended = self.step(u)
            traj.append(x.copy())
            inputs.append(float(np.ravel(u)[0]))
            if ended:
                done = k + 1
                break
        traj += [x.copy()] * (steps + 1 - len(traj))
        inputs += [0.0] * (steps - len(inputs))
        return np.array(traj), np.array(inputs), done

    def simulate(self, x0, theta, steps):
        """Closed loop under fixed theta, shape (steps+1, n), or None if the
        environment terminated the episode."""
        traj, _, done = self.rollout(x0, lambda x: self.feedback(x, theta), steps)
        return traj if done == steps else None

    def sample_theta(self, rng, size):
        return rng.uniform(self.theta_low, self.theta_high, size=(size, self.p))

    def sample_x0(self, rng, size):
        return rng.uniform(self.x0_low, self.x0_high, size=(size, self.n))

    def is_valid(self, theta, trajectories):
        return all(tr is not None for tr in trajectories)


class InvertedPendulum(MujocoSystem):
    """State [p, pd, phi, phid] (MuJoCo observation [p, phi, pd, phid] reordered),
    u = -(k1 p + k2 pd + k3 sin(phi) + k4 phid) clipped to [-3, 3]."""
    name = "mujoco_ip"
    env_id = "InvertedPendulum-v5"
    n, p = 4, 4
    gain_low, gain_high = IP_GAIN_LOW, IP_GAIN_HIGH
    x0_low = -np.array([0.5, 0.5, 0.2, 0.8])
    x0_high = np.array([0.5, 0.5, 0.2, 0.8])

    def __init__(self, gain_box="control", **kwargs):
        super().__init__(**kwargs)
        self.gain_box = gain_box
        self.gain_low, self.gain_high = IP_GAIN_BOXES[gain_box]

    def features(self, x):
        return [x[..., 0], x[..., 1], np.sin(x[..., 2]), x[..., 3]]

    def test_initial_conditions(self):
        # the 16 corners of a small box around upright
        c = np.array([0.2, 0.2, 0.08, 0.3])
        signs = np.array(np.meshgrid(*[[-1, 1]] * 4, indexing="ij")).reshape(4, -1).T
        return signs * c

    def feedback(self, x, theta):
        return float(np.clip(self.raw_feedback(x, self.to_gains(theta)), -self.u_max, self.u_max))

    def from_obs(self, obs):
        obs = np.asarray(obs, dtype=np.float64)
        return np.array([obs[0], obs[2], obs[1], obs[3]])

    def to_qpos_qvel(self, x):
        return np.array([x[0], x[2]]), np.array([x[1], x[3]])


class InvertedDoublePendulum(MujocoSystem):
    """State [p, pd, th1, th1d, th2, th2d], angles recovered with arctan2 from the
    sin/cos entries of the observation. u = -gains . [p, pd, sin th1, th1d, sin th2, th2d],
    cast to float32 and clipped to [-1, 1]."""
    name = "mujoco_idp"
    env_id = "InvertedDoublePendulum-v5"
    n, p = 6, 6
    gain_low, gain_high = IDP_GAIN_LOW, IDP_GAIN_HIGH
    x0_low = -np.array([0.3, 0.3, 0.1, 0.3, 0.1, 0.3])
    x0_high = np.array([0.3, 0.3, 0.1, 0.3, 0.1, 0.3])

    def features(self, x):
        return [x[..., 0], x[..., 1], np.sin(x[..., 2]), x[..., 3], np.sin(x[..., 4]), x[..., 5]]

    def test_initial_conditions(self):
        # 1500 random states in a small box around upright, fixed seed
        bounds = np.array([0.15, 0.15, 0.06, 0.2, 0.06, 0.2])
        return np.random.default_rng(42).uniform(-bounds, bounds, size=(1500, 6))

    def feedback(self, x, theta):
        u = np.float32(self.raw_feedback(x, self.to_gains(theta)))
        return np.clip(u, np.float32(-self.u_max), np.float32(self.u_max))

    def from_obs(self, obs):
        obs = np.asarray(obs, dtype=np.float64)
        th1 = np.arctan2(obs[1], obs[3])
        th2 = np.arctan2(obs[2], obs[4])
        return np.array([obs[0], obs[5], th1, obs[6], th2, obs[7]])

    def to_qpos_qvel(self, x):
        return np.array([x[0], x[2], x[4]]), np.array([x[1], x[3], x[5]])


ENVS = {"mujoco_ip": InvertedPendulum, "mujoco_idp": InvertedDoublePendulum}


def get(name, **kwargs):
    return ENVS[name](**kwargs)
