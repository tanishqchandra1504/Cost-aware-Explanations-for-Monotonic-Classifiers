"""Feature orderings.

Why descending cost: in both findAXp and findCXp a feature that is analysed
early is more likely to be dropped from the explanation (later features have
to make up for it). So trying expensive features first tends to keep them out
of the explanation.
"""
import numpy as np


def index_order(costs, rng=None):
    return list(range(len(costs)))


def random_order(costs, rng):
    return list(rng.permutation(len(costs)))


def cost_desc(costs, rng=None):
    c = np.asarray(costs)
    # lexsort uses the last key as primary: sort by -cost, ties by index
    return list(np.lexsort((np.arange(len(c)), -c)))


def cost_asc(costs, rng=None):
    c = np.asarray(costs)
    return list(np.lexsort((np.arange(len(c)), c)))


def perturbed_desc(costs, rng, tau=0.5):
    """Descending order of c_i * exp(tau * e_i), e_i ~ N(0, 1)."""
    c = np.asarray(costs, dtype=float)
    key = c * np.exp(tau * rng.standard_normal(len(c)))
    return list(np.lexsort((np.arange(len(c)), -key)))


ORDERINGS = {
    "index": index_order,
    "random": random_order,
    "cost_desc": cost_desc,
    "cost_asc": cost_asc,
}
