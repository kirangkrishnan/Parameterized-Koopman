"""Nonlinear MPC baselines (Table III).

Analytic systems: single shooting on the true model, RK4 discretization, IPOPT,
warm started from the shifted previous solution. MuJoCo inverted pendulum: a
cart-pole model whose parameters are read from the MuJoCo model, with input-rate
and pole-angle constraints. No NMPC is provided for the double pendulum: the
script behind that Table III entry has been lost.
"""
import time
from itertools import count

import casadi as ca
import numpy as np
import scipy.linalg

from pkoopman.systems.manipulator import M1, M2, L1, L2, G as G_ARM, TAU_MAX
from pkoopman.systems.slow_manifold import MU, LAM
from pkoopman.systems.cartpole import M_CART, M_POLE, L as L_POLE, G as G_CP

_ids = count()


def ipopt_options(max_iter, tol, acceptable_tol):
    return {"print_time": False, "error_on_fail": False,
            "ipopt.print_level": 0, "ipopt.sb": "yes",
            "ipopt.max_iter": int(max_iter), "ipopt.tol": float(tol),
            "ipopt.acceptable_tol": float(acceptable_tol), "ipopt.acceptable_iter": 3,
            "ipopt.warm_start_init_point": "yes", "ipopt.mu_init": 1e-3,
            "ipopt.bound_push": 1e-8, "ipopt.bound_frac": 1e-8}


def vdp_dynamics(x, u):
    return ca.vertcat(x[1], (1.0 - x[0] ** 2) * x[1] - x[0] + u[0])


def slow_manifold_dynamics(x, u):
    return ca.vertcat(MU * x[0] + u[0], LAM * (x[1] - x[0] ** 2))


def cartpole_dynamics(x, u):
    s, c = ca.sin(x[2]), ca.cos(x[2])
    total = M_CART + M_POLE
    tmp = (u[0] + M_POLE * L_POLE * (x[3] ** 2) * s) / total
    phidd = (G_CP * s - c * tmp) / (L_POLE * (4.0 / 3.0 - M_POLE * c ** 2 / total))
    pdd = tmp - (M_POLE * L_POLE * phidd * c) / total
    return ca.vertcat(x[1], pdd, x[3], phidd)


def manipulator_dynamics(x, u):
    d = x[0] - x[1]
    M12 = M2 * L1 * L2 * ca.cos(d)
    M = ca.vertcat(ca.horzcat((M1 + M2) * L1 ** 2, M12), ca.horzcat(M12, M2 * L2 ** 2))
    b = ca.vertcat(M2 * L1 * L2 * ca.sin(d) * x[3] ** 2 - (M1 + M2) * G_ARM * L1 * ca.sin(x[0]),
                   -M2 * L1 * L2 * ca.sin(d) * x[2] ** 2 - M2 * G_ARM * L2 * ca.sin(x[1]))
    qdd = ca.solve(M, ca.vertcat(u[0], u[1]) - b)
    return ca.vertcat(x[2], x[3], qdd[0], qdd[1])


# dynamics, input dimension, input bound, state weight Q (R = 0.1, H = 20 for all)
ANALYTIC = {
    "vdp": (vdp_dynamics, 1, 10.0, np.eye(2)),
    "slow_manifold": (slow_manifold_dynamics, 1, 10.0, np.eye(2)),
    "cartpole": (cartpole_dynamics, 1, 20.0, np.eye(4)),
    "manipulator": (manipulator_dynamics, 2, TAU_MAX, np.diag([15.0, 15.0, 1.0, 1.0])),
}


class NMPC:
    """min sum_k x_k' Q x_k + u_k' R u_k + terminal_scale x_H' Q x_H, |u| <= u_max,
    over the input sequence, with x_{k+1} = RK4(x_k, u_k)."""

    def __init__(self, dynamics, nx, nu, dt, Q, R, u_max, horizon=20, terminal_scale=10.0,
                 max_iter=60, tol=1e-6, acceptable_tol=1e-3):
        self.nx, self.nu, self.H = nx, nu, horizon
        Q, R = ca.DM(np.asarray(Q, float)), ca.DM(R * np.eye(nu))

        def rk4(x, u):
            k1 = dynamics(x, u)
            k2 = dynamics(x + 0.5 * dt * k1, u)
            k3 = dynamics(x + 0.5 * dt * k2, u)
            k4 = dynamics(x + dt * k3, u)
            return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        U = ca.SX.sym("U", horizon * nu)
        x0 = ca.SX.sym("x0", nx)
        x, J = x0, 0
        for k in range(horizon):
            uk = U[k * nu:(k + 1) * nu]
            x = rk4(x, uk)
            J += ca.mtimes([x.T, Q, x])
            J += ca.mtimes([uk.T, R, uk])
        J += float(terminal_scale) * ca.mtimes([x.T, Q, x])
        self.solver = ca.nlpsol(f"nmpc_{next(_ids)}", "ipopt", {"x": U, "p": x0, "f": J},
                                ipopt_options(max_iter, tol, acceptable_tol))
        self.u_max = u_max
        self.lbx = np.full(horizon * nu, -float(u_max))
        self.ubx = np.full(horizon * nu, float(u_max))
        self.reset()

    def reset(self):
        self.u_warm = np.zeros((self.H, self.nu))
        self.lam_x = None
        self.solve_time = 0.0

    def solve(self, x):
        """Returns the first input of the optimal sequence (shape (nu,)). If IPOPT
        fails the shifted previous sequence is used instead."""
        previous = self.u_warm.copy()
        args = dict(x0=previous.reshape(-1), p=np.asarray(x, float), lbx=self.lbx, ubx=self.ubx)
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


def analytic_nmpc(name, dt):
    dynamics, nu, u_max, Q = ANALYTIC[name]
    return NMPC(dynamics, Q.shape[0], nu, dt, Q, 0.1, u_max)


class InvertedPendulumNMPC:
    """NMPC for MuJoCo InvertedPendulum-v5 in the state order [p, pd, phi, phid].

    Prediction model: rigid cart-pole with damping, parameters taken from the
    MuJoCo model, integrated with frame_skip RK4 substeps per env step. Cost on
    state, input and input rate plus a quartic pole-angle term; constraints
    |u_k - u_{k-1}| <= 1 and |phi_k| <= 0.19. At every step the IPOPT solution
    is compared with an LQR rollout and the shifted previous plan, and the
    cheapest of the three is applied.
    """
    Q = np.diag([60.0, 4.0, 700.0, 12.0])
    QF_SCALE = 25.0
    R = 0.08
    R_DU = 0.60
    DU_MAX = 1.00
    PHI_MAX = 0.19
    PHI_QUARTIC = 1.0e5

    def __init__(self, env, horizon=20, u_max=3.0):
        import mujoco
        model = env.unwrapped.model
        dt = float(env.unwrapped.dt)
        substeps = max(1, int(env.unwrapped.frame_skip))
        idx = lambda kind, name: int(mujoco.mj_name2id(model, kind, name))
        cart, pole = idx(mujoco.mjtObj.mjOBJ_BODY, "cart"), idx(mujoco.mjtObj.mjOBJ_BODY, "pole")
        slider = int(model.jnt_dofadr[idx(mujoco.mjtObj.mjOBJ_JOINT, "slider")])
        hinge = int(model.jnt_dofadr[idx(mujoco.mjtObj.mjOBJ_JOINT, "hinge")])
        M, m = float(model.body_mass[cart]), float(model.body_mass[pole])
        l = float(np.linalg.norm(np.asarray(model.body_ipos[pole], dtype=np.float64)))
        I_com = float(model.body_inertia[pole, 1])
        b_x, b_phi = float(model.dof_damping[slider]), float(model.dof_damping[hinge])
        gear = float(abs(model.actuator_gear[idx(mujoco.mjtObj.mjOBJ_ACTUATOR, "slide"), 0]))
        g = float(abs(model.opt.gravity[2]))

        def f(x, u):
            s, c = ca.sin(x[2]), ca.cos(x[2])
            J = I_com + m * l * l
            coupling = m * l * c
            rhs_x = gear * u - b_x * x[1] + m * l * s * x[3] * x[3]
            rhs_phi = m * g * l * s - b_phi * x[3]
            det = (M + m) * J - coupling * coupling
            return ca.vertcat(x[1], (J * rhs_x - coupling * rhs_phi) / det,
                              x[3], (-coupling * rhs_x + (M + m) * rhs_phi) / det)

        def step(x, u):
            h = dt / substeps
            for _ in range(substeps):
                k1 = f(x, u)
                k2 = f(x + 0.5 * h * k1, u)
                k3 = f(x + 0.5 * h * k2, u)
                k4 = f(x + h * k3, u)
                x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            return x

        xs, us = ca.MX.sym("x", 4), ca.MX.sym("u", 1)
        xn = step(xs, us[0])
        self.model_step = ca.Function("ip_step", [xs, us], [xn])
        A, B = ca.Function("ip_jac", [xs, us], [ca.jacobian(xn, xs), ca.jacobian(xn, us)])(
            np.zeros(4), np.zeros(1))
        A, B = np.asarray(A, float), np.asarray(B, float)
        Rm = np.array([[self.R]])
        P = scipy.linalg.solve_discrete_are(A, B, self.Q, Rm)
        self.K_lqr = np.linalg.solve(B.T @ P @ B + Rm, B.T @ P @ A).reshape(1, 4)

        # parameters: current state and the previously applied input
        U, par = ca.MX.sym("U", horizon), ca.MX.sym("P", 5)
        Q, QF = ca.DM(self.Q), ca.DM(self.QF_SCALE * self.Q)
        x, J, g_list = par[0:4], 0, []
        for k in range(horizon):
            du = U[k] - (par[4] if k == 0 else U[k - 1])
            J += ca.mtimes([x.T, Q, x])
            J += self.R * U[k] * U[k]
            J += self.R_DU * du * du
            J += self.PHI_QUARTIC * x[2] ** 4
            g_list.append(du)
            x = step(x, U[k])
            g_list.append(x[2])
        J += ca.mtimes([x.T, QF, x])
        J += self.PHI_QUARTIC * x[2] ** 4
        opts = {"expand": True, "print_time": False, "error_on_fail": False,
                "ipopt.print_level": 0, "ipopt.sb": "yes", "ipopt.max_iter": 45,
                "ipopt.tol": 1e-5, "ipopt.acceptable_tol": 1e-4, "ipopt.acceptable_iter": 4,
                "ipopt.mu_strategy": "adaptive", "ipopt.warm_start_init_point": "yes",
                "ipopt.warm_start_bound_push": 1e-8, "ipopt.warm_start_mult_bound_push": 1e-8,
                "ipopt.warm_start_slack_bound_push": 1e-8}
        nlp = {"x": U, "p": par, "f": J, "g": ca.vertcat(*g_list)}
        self.solver = ca.nlpsol(f"ip_nmpc_{next(_ids)}", "ipopt", nlp, opts)
        self.cost = ca.Function("ip_cost", [U, par], [J])

        self.H = horizon
        self.u_low = max(-u_max, float(env.action_space.low[0]))
        self.u_high = min(u_max, float(env.action_space.high[0]))
        self.lbx, self.ubx = np.full(horizon, self.u_low), np.full(horizon, self.u_high)
        self.lbg = np.tile([-self.DU_MAX, -self.PHI_MAX], horizon)
        self.ubg = np.tile([self.DU_MAX, self.PHI_MAX], horizon)
        self.reset()

    def reset(self):
        self.warm = None
        self.u_prev = 0.0
        self.lam_x = self.lam_g = None
        self.solve_time = 0.0

    def lqr_plan(self, x):
        plan, u_prev = np.empty(self.H), self.u_prev
        for k in range(self.H):
            lo, hi = max(self.u_low, u_prev - self.DU_MAX), min(self.u_high, u_prev + self.DU_MAX)
            plan[k] = np.clip(-float((self.K_lqr @ x)[0]), lo, hi)
            x = np.asarray(self.model_step(x, np.array([plan[k]])), float).reshape(4)
            u_prev = plan[k]
        return plan

    def plan_cost(self, plan, par):
        return float(np.asarray(self.cost(plan, par)).reshape(-1)[0])

    def solve(self, x):
        x = np.asarray(x, float)
        par = np.concatenate([x, [self.u_prev]])
        lqr = self.lqr_plan(x)
        guess = lqr.copy() if self.warm is None else np.clip(self.warm, self.lbx, self.ubx)
        args = dict(x0=guess, p=par, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg)
        if self.lam_x is not None and self.lam_g is not None:
            args["lam_x0"], args["lam_g0"] = self.lam_x, self.lam_g
        t0 = time.perf_counter()
        sol = self.solver(**args)
        self.solve_time = time.perf_counter() - t0

        candidates = [(lqr, self.plan_cost(lqr, par)), (guess, self.plan_cost(guess, par))]
        opt = np.asarray(sol["x"], float).reshape(self.H)
        if np.all(np.isfinite(opt)):
            opt = np.clip(opt, self.lbx, self.ubx)
            candidates.append((opt, self.plan_cost(opt, par)))
        if self.solver.stats()["success"]:
            self.lam_x = np.asarray(sol["lam_x"], float).reshape(-1)
            self.lam_g = np.asarray(sol["lam_g"], float).reshape(-1)

        plan = min(candidates, key=lambda c: c[1])[0]
        u0 = float(np.clip(plan[0], self.u_low, self.u_high))
        self.warm = np.append(plan[1:], plan[-1])
        self.u_prev = u0
        return u0
