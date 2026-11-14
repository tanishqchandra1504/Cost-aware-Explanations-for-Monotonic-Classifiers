"""findAXp and findCXp (Algorithms 1 and 2 of Marques-Silva et al., ICML 2021).

The only change from the paper is that the order in which features are
analysed is a parameter (the paper / XMono always use 1..N).

Features are 0-indexed here. `seed` is the set S from the paper, used when the
algorithms are called from the enumeration procedure:
  - findAXp: S = features that are already known to be free
  - findCXp: S = features that are already known to be fixed
"""
import numpy as np


def find_axp(clf, v, order=None, seed=(), trace=None):
    """Return one AXp (sorted list of feature indices).

    Invariant: kappa(vL) == kappa(vU), so every point in the box [vL, vU] gets
    the same prediction as v (Proposition 1 in the paper).
    """
    v = np.asarray(v, dtype=float)
    n = len(v)
    order = range(n) if order is None else order
    vl, vu = v.copy(), v.copy()
    free = set(seed)
    for i in free:
        vl[i], vu[i] = clf.lo[i], clf.hi[i]
    picked = []
    for i in order:
        if i in free:
            continue
        # try to let feature i take any value (FreeAttr)
        vl[i], vu[i] = clf.lo[i], clf.hi[i]
        cl, cu = clf(vl), clf(vu)
        if trace is not None:
            trace.append((i, vl.copy(), vu.copy(), cl, cu))
        if cl != cu:
            # invariant broken -> feature must stay fixed (FixAttr)
            vl[i] = vu[i] = v[i]
            picked.append(i)
        else:
            free.add(i)
    return sorted(picked)


def find_cxp(clf, v, order=None, seed=(), trace=None):
    """Return one CXp (sorted list of feature indices), or None if the
    prediction cannot change at all (kappa(lo) == kappa(hi)).

    Invariant: kappa(vL) != kappa(vU), i.e. letting the features in P and C
    take any value can still change the prediction.
    """
    v = np.asarray(v, dtype=float)
    n = len(v)
    order = range(n) if order is None else order
    vl, vu = clf.lo.copy(), clf.hi.copy()
    fixed = set(seed)
    for i in fixed:
        vl[i] = vu[i] = v[i]
    if not fixed and clf(vl) == clf(vu):
        return None
    picked = []
    for i in order:
        if i in fixed:
            continue
        # try to fix feature i to its value in v (FixAttr)
        vl[i] = vu[i] = v[i]
        cl, cu = clf(vl), clf(vu)
        if trace is not None:
            trace.append((i, vl.copy(), vu.copy(), cl, cu))
        if cl == cu:
            # prediction can no longer change -> feature must stay free
            vl[i], vu[i] = clf.lo[i], clf.hi[i]
            picked.append(i)
        else:
            fixed.add(i)
    return sorted(picked)


def explain(kind, clf, v, order=None, seed=()):
    if kind == "axp":
        return find_axp(clf, v, order, seed)
    if kind == "cxp":
        return find_cxp(clf, v, order, seed)
    raise ValueError(kind)
