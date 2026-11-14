"""Exact minimum-cost AXp / CXp, used as the reference ("oracle") in experiments.

1. brute_force_*: try all 2^n subsets (only for small n, used in tests).
2. min_cost_*: implicit hitting-set loop. By the AXp/CXp duality, a
   minimum-cost AXp is a minimum-cost hitting set of all CXps (and vice versa).
   We only generate the CXps (AXps) that are needed:
     - solve weighted MaxSAT for a min-cost hitting set H of the sets found so far
     - if H is an AXp (CXp) we are done: it is optimal
     - otherwise extract a new CXp (AXp) that H misses and repeat
   Weights are scaled to integers (x 1e6) for RC2; the reported cost is
   recomputed with the original float costs.
"""
import itertools

import numpy as np
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF

from .core import expl_cost
from .explain import find_axp, find_cxp

SCALE = 10**6


def _min_hitting_set(sets, costs):
    w = WCNF()
    for S in sets:
        w.append([i + 1 for i in S])
    for i, c in enumerate(costs):
        w.append([-(i + 1)], weight=max(1, int(round(c * SCALE))))
    with RC2(w) as rc2:
        model = rc2.compute()
    return [i for i in range(len(costs)) if model[i] > 0]


def _box_pred(clf, v, fixed):
    vl, vu = clf.lo.copy(), clf.hi.copy()
    vl[fixed], vu[fixed] = v[fixed], v[fixed]
    return clf(vl), clf(vu)


def min_cost_axp(clf, v, costs, max_iter=100000):
    v = np.asarray(v, dtype=float)
    cxps = []
    for _ in range(max_iter):
        H = _min_hitting_set(cxps, costs)
        cl, cu = _box_pred(clf, v, H)
        if cl == cu:
            return H, len(cxps)
        cxps.append(find_cxp(clf, v, seed=H))
    raise RuntimeError("min_cost_axp did not converge")


def min_cost_cxp(clf, v, costs, max_iter=100000):
    v = np.asarray(v, dtype=float)
    if clf(clf.lo) == clf(clf.hi):
        return None, 0
    axps = []
    for _ in range(max_iter):
        Y = _min_hitting_set(axps, costs)
        fixed = [i for i in range(clf.n) if i not in Y]
        cl, cu = _box_pred(clf, v, fixed)
        if cl != cu:
            return Y, len(axps)
        axps.append(find_axp(clf, v, seed=Y))
    raise RuntimeError("min_cost_cxp did not converge")


def min_cost(kind, clf, v, costs):
    return (min_cost_axp if kind == "axp" else min_cost_cxp)(clf, v, costs)


# ---- brute force (small n only) -------------------------------------------

def _all_subsets(n):
    for r in range(n + 1):
        for S in itertools.combinations(range(n), r):
            yield list(S)


def brute_force_min_axp(clf, v, costs):
    v = np.asarray(v, dtype=float)
    c = clf.predict_uncounted(v)
    best = None
    for S in _all_subsets(clf.n):
        vl, vu = clf.lo.copy(), clf.hi.copy()
        vl[S], vu[S] = v[S], v[S]
        if clf.predict_uncounted(vl) == c and clf.predict_uncounted(vu) == c:
            if best is None or expl_cost(S, costs) < expl_cost(best, costs) - 1e-12:
                best = S
    return best


def brute_force_min_cxp(clf, v, costs):
    v = np.asarray(v, dtype=float)
    best = None
    for Y in _all_subsets(clf.n):
        fixed = [i for i in range(clf.n) if i not in Y]
        vl, vu = clf.lo.copy(), clf.hi.copy()
        vl[fixed], vu[fixed] = v[fixed], v[fixed]
        if clf.predict_uncounted(vl) != clf.predict_uncounted(vu):
            if best is None or expl_cost(Y, costs) < expl_cost(best, costs) - 1e-12:
                best = Y
    return best
