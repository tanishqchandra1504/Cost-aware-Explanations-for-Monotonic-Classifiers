"""Checks used to verify returned explanations.

These do not reuse the code in explain.py. Two kinds of evidence are used:
  * concrete counterexample points (sound for any classifier), used for
    "the prediction can change" claims;
  * the lower/upper bound test (needs monotonicity) plus random sampling in
    the box, used for "the prediction cannot change" claims.
Calls made here are not counted.
"""
import numpy as np


def _box(clf, v, fixed):
    vl, vu = clf.lo.copy(), clf.hi.copy()
    idx = list(fixed)
    vl[idx] = v[idx]
    vu[idx] = v[idx]
    return vl, vu


def can_change(clf, v, fixed):
    """True if some point that agrees with v on `fixed` gets another class.
    Returns the result of the bound test (witness is vl or vu)."""
    c = clf.predict_uncounted(v)
    vl, vu = _box(clf, v, fixed)
    return clf.predict_uncounted(vl) != c or clf.predict_uncounted(vu) != c


def no_change_sampled(clf, v, fixed, rng, n_samples=256):
    """Sampling-based falsification of 'prediction cannot change'."""
    c = clf.predict_uncounted(v)
    X = rng.uniform(clf.lo, clf.hi, size=(n_samples, clf.n))
    idx = list(fixed)
    X[:, idx] = v[idx]
    return bool(np.all(clf.predict_uncounted(X) == c))


def check_axp(clf, v, axp, rng, n_samples=256):
    """Returns (sufficient, minimal)."""
    v = np.asarray(v, dtype=float)
    axp = list(axp)
    sufficient = (not can_change(clf, v, axp)) and no_change_sampled(clf, v, axp, rng, n_samples)
    minimal = all(can_change(clf, v, [j for j in axp if j != i]) for i in axp)
    return sufficient, minimal


def check_cxp(clf, v, cxp, rng, n_samples=256):
    """Returns (changes_prediction, minimal)."""
    v = np.asarray(v, dtype=float)
    cxp = set(cxp)
    rest = [j for j in range(clf.n) if j not in cxp]
    changes = can_change(clf, v, rest)
    minimal = True
    for i in cxp:
        fixed = rest + [i]
        if can_change(clf, v, fixed) or not no_change_sampled(clf, v, fixed, rng, n_samples):
            minimal = False
            break
    return changes, minimal


def check(kind, clf, v, expl, rng, n_samples=256):
    f = check_axp if kind == "axp" else check_cxp
    a, b = f(clf, v, expl, rng, n_samples)
    return bool(a and b)


def empirical_monotonicity(clf, rng, n_pairs=5000):
    """Fraction of random comparable pairs (a <= b) with kappa(a) <= kappa(b)."""
    A = rng.uniform(clf.lo, clf.hi, size=(n_pairs, clf.n))
    B = A + rng.uniform(0, 1, size=A.shape) * (clf.hi - A) * (rng.uniform(size=A.shape) < 0.5)
    return float(np.mean(clf.predict_uncounted(A) <= clf.predict_uncounted(B)))
