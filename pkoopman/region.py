"""Construction of the verified parameter region C (Algorithm 1 in the paper).

1. Sample parameters from the full box and keep the ones where the model passes
   the prediction test on a grid of initial conditions.
2. Build the convex hull of the survivors, sample new candidates inside it and
   test them again. If some fail, the survivors define a smaller hull and we
   repeat. Stop when every candidate inside the hull passes.
3. Draw the final parameter samples from the converged hull.

Steps 1-2 use the scout model. After the final model has been trained on
parameters from that hull, verify_region() repeats the test with the final
model on fresh samples and shrinks the hull again if too many fail (Phase 3).
"""
import multiprocessing as mp
import os

import numpy as np
import torch
from scipy.spatial import Delaunay

from .metrics import worst_case_error

# set before the worker pool is forked, so that every worker has its own copy
_JOB = {}


def _simulate_one(theta):
    system, x0s, steps = _JOB["system"], _JOB["x0s"], _JOB["steps"]
    trajs = [system.simulate(x0, theta, steps) for x0 in x0s]
    return trajs, bool(system.is_valid(theta, trajs))


def default_workers():
    return int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))


def simulate_all(system, thetas, x0s, steps, workers=None):
    """True closed loops for every theta from every x0, spread over worker
    processes. Returns a list of trajectory lists and an array of is_valid flags."""
    workers = workers or default_workers()
    _JOB.update(system=system, x0s=x0s, steps=steps)
    if workers <= 1 or len(thetas) < 2 * workers:
        out = [_simulate_one(th) for th in thetas]
    else:
        with mp.get_context("fork").Pool(workers) as pool:
            out = pool.map(_simulate_one, list(thetas), chunksize=max(1, len(thetas) // (8 * workers)))
    return [o[0] for o in out], np.array([o[1] for o in out], dtype=bool)


def model_errors(model, system, thetas, x0s, steps, workers=None):
    """Worst-case prediction error of the model for every theta, taken over all x0s.
    Also returns whether the true closed loop passed system.is_valid()."""
    device = next(model.parameters()).device
    errors = np.full(len(thetas), np.inf)
    x0_t = torch.tensor(np.asarray(x0s), dtype=torch.float32, device=device)
    all_trajs, valid = simulate_all(system, thetas, x0s, steps, workers)

    for i, (theta, trajs) in enumerate(zip(thetas, all_trajs)):
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


def verify_region(model, system, region, rng, n_samples=1500, steps=12, threshold=15.0,
                  acceptance=0.95, max_iter=30, require_valid=False, n_keep=None):
    """Phase 3: test the final model on n_samples fresh parameters from the hull
    of `region`. If fewer than `acceptance` of them pass, the passing ones span a
    smaller hull and the test is repeated there. Returns the accepted region
    (samples from the last hull) and the history, or accepted=False if the hull
    collapses or max_iter is reached.

    The accepted hull is the one the last batch was drawn from, so the rate that
    stopped the loop was measured on it; a separate draw with a new seed
    (scripts/region_check.py) gives the reported estimate."""
    x0s = system.test_initial_conditions()
    anchors, history = np.asarray(region), []
    for it in range(max_iter + 1):
        hull = Delaunay(anchors)
        cand = sample_in_hull(hull, anchors, n_samples, rng)
        errors, valid = model_errors(model, system, cand, x0s, steps)
        ok = errors < threshold
        if require_valid:
            ok &= valid
        rate = float(ok.mean())
        history.append(dict(iteration=it, pass_rate=rate, n_vertices=int(len(anchors)),
                            median_error=float(np.median(errors))))
        print(f"phase 3, iteration {it}: {ok.sum()}/{n_samples} pass ({100 * rate:.1f}%), "
              f"median error {np.median(errors):.1f}%", flush=True)
        if rate >= acceptance:
            final = sample_in_hull(hull, anchors, n_keep or len(region), rng)
            return dict(accepted=True, final=final, anchors=anchors, history=history)
        if ok.sum() <= system.p + 1:
            break
        anchors = cand[ok]
    return dict(accepted=False, final=None, anchors=anchors, history=history)
