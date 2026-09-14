"""Observation-only smoothed geometric median from a trainable Gram matrix.

Equal initial client mass; no labels, RTC trust weights or persistent state.
The smoothed objective is mean(sqrt(||x_i-z||^2 + epsilon^2)).
Numerical convergence does not establish Byzantine trustworthiness.
"""
import numpy as np

MAX_ITERATIONS = 200
RELATIVE_TOLERANCE = 1e-7
RELATIVE_SMOOTHING = 1e-6
RELATIVE_MIN_NORM = 1e-6


def geometric_reference(gram, *, omit=None, max_iterations=MAX_ITERATIONS):
    gram = np.asarray(gram, dtype=np.float64)
    if (gram.ndim != 2 or gram.shape[0] != gram.shape[1] or not np.isfinite(gram).all()
            or gram.shape[0] < 2 or type(max_iterations) is not int or max_iterations < 1):
        raise ValueError('invalid geometric reference Gram matrix or iterations')
    n = len(gram)
    if omit is not None and (type(omit) is not int or not 0 <= omit < n):
        raise ValueError('invalid omitted client')
    mask = np.ones(n, dtype=bool)
    if omit is not None:
        mask[omit] = False
    scale_sq = float(np.mean(np.maximum(np.diag(gram)[mask], 0)))
    alpha = mask.astype(float) / mask.sum()
    if scale_sq <= 1e-24:
        return alpha, {'converged': False, 'iterations': 0, 'relative_step': None, 'scale': scale_sq ** .5}
    g = (gram + gram.T) / (2 * scale_sq)
    diag = np.diag(g)
    converged = False
    for iteration in range(1, max_iterations + 1):
        distances_sq = np.maximum(diag - 2 * (g @ alpha) + alpha @ g @ alpha, 0)
        beta = mask / np.sqrt(distances_sq + RELATIVE_SMOOTHING ** 2)
        beta /= beta.sum()
        change = beta - alpha
        relative_step = float(np.sqrt(max(0, change @ g @ change)))
        alpha = beta
        if relative_step <= RELATIVE_TOLERANCE:
            converged = True
            break
    return alpha, {'converged': converged, 'iterations': iteration,
                   'relative_step': relative_step, 'scale': scale_sq ** .5}
