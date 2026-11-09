"""Construction of the verified parameter region C (Algorithm 1 in the paper).

1. Sample parameters from the full box and keep the ones where the model passes
   the prediction test on a grid of initial conditions.
2. Build the convex hull of the survivors, sample new candidates inside it and
   test them again. If some fail, the survivors define a smaller hull and we
   repeat. Stop when every candidate inside the hull passes.
3. Draw the final parameter samples from the converged hull.
"""
import numpy as np
import torch
from scipy.spatial import Delaunay

from .metrics import worst_case_error


def model_errors(model, system, thetas, x0s, steps):
    """Worst-case prediction error of the model for every theta, taken over all x0s.
    Also returns whether the true closed loop passed system.is_valid()."""
    device = next(model.parameters()).device
    errors = np.full(len(thetas), np.inf)
    valid = np.zeros(len(thetas), dtype=bool)
    x0_t = torch.tensor(np.asarray(x0s), dtype=torch.float32, device=device)

    for i, theta in enumerate(thetas):
        trajs = [system.simulate(x0, theta, steps) for x0 in x0s]
        valid[i] = system.is_valid(theta, trajs)
        if any(tr is None for tr in trajs):
            continue

        th = torch.tensor(theta, dtype=torch.float32, device=device).expand(len(x0s), -1)
        with torch.no_grad():
            pred = model.predict(x0_t, th, steps).cpu().numpy()
        errors[i] = worst_case_error(pred, np.stack(trajs)[:, 1:]).max()
    return errors, valid


def passing(model, system, thetas, x0s, steps, threshold, require_valid):
    errors, valid = model_errors(model, system, thetas, x0s, steps)
    ok = errors < threshold
    if require_valid:
        ok &= valid
    return thetas[ok]


def sample_in_hull(hull, points, n, rng):
    lo, hi = points.min(axis=0), points.max(axis=0)
    out = []
    while len(out) < n:
        cand = rng.uniform(lo, hi, size=(4 * n, points.shape[1]))
        out.extend(cand[hull.find_simplex(cand) >= 0])
    return np.array(out[:n])


def find_region(model, system, rng, n_explore=3000, n_test=1500, n_final=4000,
                steps=12, threshold=15.0, max_iter=100, require_valid=False):
    x0s = system.test_initial_conditions()

    explored = system.sample_theta(rng, n_explore)
    anchors = passing(model, system, explored, x0s, steps, threshold, require_valid)
    print(f"exploration: {len(anchors)}/{n_explore} parameters pass")
    if len(anchors) <= system.p + 1:
        raise RuntimeError("too few passing parameters to build a hull")

    history = [anchors]
    for it in range(1, max_iter + 1):
        hull = Delaunay(anchors)
        cand = sample_in_hull(hull, anchors, n_test, rng)
        survivors = passing(model, system, cand, x0s, steps, threshold, require_valid)
        rate = len(survivors) / n_test
        print(f"iteration {it}: {len(survivors)}/{n_test} pass ({100 * rate:.1f}%)")
        anchors = survivors
        history.append(anchors)
        if rate >= 1.0:
            break

    final = sample_in_hull(Delaunay(anchors), anchors, n_final, rng)
    return dict(final=final, anchors=anchors, explored=explored, history=history)
