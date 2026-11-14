"""Multi-start selection: run findAXp/findCXp with K orders, keep the cheapest.

Every run returns a valid subset-minimal explanation, so the selected one is
valid too. Each run costs exactly 2 * n classifier calls, so two multi-start
variants with the same K use exactly the same number of calls.
"""
import numpy as np

from .core import expl_cost
from .explain import explain
from .orderings import cost_desc, perturbed_desc, random_order


def multistart(kind, clf, v, costs, K, rng, guided=True, tau=0.5):
    """guided=True : run 1 = cost-descending order, runs 2..K = perturbed
                     cost-descending orders (variant D).
       guided=False: K random orders (variant D', same budget)."""
    best, best_key = None, None
    for k in range(K):
        if guided:
            order = cost_desc(costs) if k == 0 else perturbed_desc(costs, rng, tau)
        else:
            order = random_order(costs, rng)
        e = explain(kind, clf, v, order)
        if e is None:
            return None
        key = (round(expl_cost(e, costs), 9), len(e))
        if best is None or key < best_key:
            best, best_key = e, key
    return best
