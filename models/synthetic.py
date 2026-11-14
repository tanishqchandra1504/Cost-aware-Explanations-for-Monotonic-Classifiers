"""Synthetic monotonic classifiers on [0, 1]^n.

All three families are monotonic by construction (non-negative weights,
max of monotone functions, OR of ANDs of upward thresholds), which we also
check empirically in the tests.

  linear  : binary, w . x >= b, w_i >= 0
  maxlin  : 4 ordered classes from max of k sparse non-negative linear scores
  dnf     : binary, OR of terms, term = AND_{i in T} (x_i >= t_i)

Thresholds are set from quantiles of the score under uniform inputs so the
classes are roughly balanced.
"""
import numpy as np

from mono.core import MonoClassifier


def _quantile_thresholds(score_fn, n, n_classes, rng, n_samples=4000):
    s = score_fn(rng.uniform(size=(n_samples, n)))
    return np.quantile(s, np.arange(1, n_classes) / n_classes)


def _classes(score, thr):
    return np.sum(score[:, None] >= thr[None, :], axis=1).astype(int)


def make_linear(n, rng):
    w = rng.exponential(1.0, n)
    score = lambda X: np.atleast_2d(X) @ w
    thr = _quantile_thresholds(score, n, 2, rng)
    return MonoClassifier(lambda X: _classes(score(X), thr), np.zeros(n), np.ones(n), "linear")


def make_maxlin(n, rng, k=3, n_classes=4):
    W = np.zeros((k, n))
    for j in range(k):
        S = rng.choice(n, size=max(2, n // 2), replace=False)
        W[j, S] = rng.exponential(1.0, len(S))
        W[j] /= W[j].sum()
    score = lambda X: np.max(np.atleast_2d(X) @ W.T, axis=1)
    thr = _quantile_thresholds(score, n, n_classes, rng)
    return MonoClassifier(lambda X: _classes(score(X), thr), np.zeros(n), np.ones(n), "maxlin")


def make_dnf(n, rng, term_size=3, n_terms=None):
    n_terms = n_terms or max(3, n // 2)
    t = rng.uniform(0.3, 0.7, n)
    terms = [rng.choice(n, size=term_size, replace=False) for _ in range(n_terms)]

    def predict(X):
        B = np.atleast_2d(X) >= t[None, :]
        out = np.zeros(B.shape[0], dtype=bool)
        for T in terms:
            out |= np.all(B[:, T], axis=1)
        return out.astype(int)

    return MonoClassifier(predict, np.zeros(n), np.ones(n), "dnf")


FAMILIES = {"linear": make_linear, "maxlin": make_maxlin, "dnf": make_dnf}


def make_costs(n, rng, sigma=1.0):
    """Log-normal costs normalised to mean 1. sigma = 0 gives equal costs."""
    if sigma == 0:
        return np.ones(n)
    c = rng.lognormal(0.0, sigma, n)
    return c / c.mean()


def make_instance(family, n, seed, sigma=1.0):
    """One experimental instance = (classifier, point v, feature costs),
    fully determined by the seed."""
    rng = np.random.default_rng(seed)
    clf = FAMILIES[family](n, rng)
    v = rng.uniform(size=n)
    costs = make_costs(n, rng, sigma)
    return clf, v, costs
