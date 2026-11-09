import numpy as np


def worst_case_error(x_pred, x_true, floor=0.1):
    """Maximum normalized error over the horizon, in percent.

    x_pred, x_true: arrays of shape (..., H, n) holding x_1..x_H.
    The denominator is floored at 0.1 so that states close to the origin
    do not blow up the relative error.
    """
    err = np.linalg.norm(x_pred - x_true, axis=-1)
    scale = np.maximum(np.linalg.norm(x_true, axis=-1), floor)
    return 100.0 * np.max(err / scale, axis=-1)


def summarize(errors, threshold=15.0):
    errors = np.asarray(errors)
    return dict(mean=float(errors.mean()), median=float(np.median(errors)),
                std=float(errors.std()), max=float(errors.max()),
                below_threshold=float(np.mean(errors < threshold)))
