import numpy as np
import pytest

from mono.core import MonoClassifier, expl_cost
from mono.enumerate import enumerate_xps
from mono.exact import (brute_force_min_axp, brute_force_min_cxp,
                        min_cost_axp, min_cost_cxp)
from mono.explain import find_axp, find_cxp
from mono.multistart import multistart
from mono.orderings import cost_asc, cost_desc, random_order
from mono.verify import check, empirical_monotonicity
from models.synthetic import FAMILIES, make_instance
from models.xmono_models import GRADES, student_grades

V0 = np.array([10, 10, 5, 0.0])   # student s1 from the paper


# ---- paper, Tables 1-3 -----------------------------------------------------

def test_table1_axp():
    clf = student_grades()
    trace = []
    axp = find_axp(clf, V0, trace=trace)
    assert axp == [0, 1]                       # {1, 2} in the paper
    preds = [(GRADES[cl], GRADES[cu]) for (_, _, _, cl, cu) in trace]
    assert preds == [("C", "A"), ("E", "A"), ("A", "A"), ("A", "A")]
    assert clf.calls == 8                      # 2 calls per feature


def test_table2_cxp():
    clf = student_grades()
    trace = []
    cxp = find_cxp(clf, V0, trace=trace)
    assert cxp == [1]                          # {2} in the paper
    preds = [(GRADES[cl], GRADES[cu]) for (_, _, _, cl, cu) in trace]
    assert preds == [("E", "A"), ("A", "A"), ("E", "A"), ("E", "A")]


def test_table3_enumeration():
    axps, cxps, complete, n_sat = enumerate_xps(student_grades(), V0)
    assert complete
    assert sorted(axps) == [[0, 1]]
    assert sorted(cxps) == [[0], [1]]
    assert n_sat == len(axps) + len(cxps) + 1  # as stated in the paper


# ---- toy examples used in the README to motivate descending order ---------

def _boolclf(f, n):
    return MonoClassifier(lambda X: f(np.atleast_2d(X)).astype(int), np.zeros(n), np.ones(n))


def test_order_matters_axp():
    clf = _boolclf(lambda X: (X[:, 0] + X[:, 1]) >= 1, 2)       # x1 OR x2
    v, c = np.ones(2), np.array([10.0, 1.0])
    assert find_axp(clf, v, cost_desc(c)) == [1]                # cost 1
    assert find_axp(clf, v, cost_asc(c)) == [0]                 # cost 10


def test_order_matters_cxp():
    clf = _boolclf(lambda X: (X[:, 0] + X[:, 1]) >= 2, 2)       # x1 AND x2
    v, c = np.ones(2), np.array([10.0, 1.0])
    assert find_cxp(clf, v, cost_desc(c)) == [1]
    assert find_cxp(clf, v, cost_asc(c)) == [0]


def test_greedy_not_optimal():
    # x1 OR (x2 AND x3), costs (5, 3, 3): descending gives {2,3} (6), optimum {1} (5)
    clf = _boolclf(lambda X: (X[:, 0] >= 1) | ((X[:, 1] >= 1) & (X[:, 2] >= 1)), 3)
    v, c = np.ones(3), np.array([5.0, 3.0, 3.0])
    assert find_axp(clf, v, cost_desc(c)) == [1, 2]
    assert min_cost_axp(clf, v, c)[0] == [0]
    # a multi-start run that happens to try feature 2 or 3 first can fix it
    best = multistart("axp", clf, v, c, K=16, rng=np.random.default_rng(0), tau=2.0)
    assert best == [0]


# ---- exact solver and validity on random synthetic instances --------------

@pytest.mark.parametrize("family", list(FAMILIES))
def test_exact_matches_brute_force(family):
    for seed in range(15):
        clf, v, c = make_instance(family, 8, seed)
        bf = brute_force_min_axp(clf, v, c)
        ex, _ = min_cost_axp(clf, v, c)
        assert abs(expl_cost(bf, c) - expl_cost(ex, c)) < 1e-6
        bf = brute_force_min_cxp(clf, v, c)
        ex, _ = min_cost_cxp(clf, v, c)
        if bf is None:
            assert ex is None
        else:
            assert abs(expl_cost(bf, c) - expl_cost(ex, c)) < 1e-6


@pytest.mark.parametrize("family", list(FAMILIES))
def test_enumeration_contains_optimum(family):
    for seed in range(10):
        clf, v, c = make_instance(family, 8, seed)
        axps, cxps, complete, _ = enumerate_xps(clf, v)
        assert complete
        best = min(expl_cost(a, c) for a in axps)
        assert abs(best - expl_cost(min_cost_axp(clf, v, c)[0], c)) < 1e-6
        assert len({tuple(a) for a in axps}) == len(axps)   # no repeats


@pytest.mark.parametrize("family", list(FAMILIES))
def test_models_are_monotonic(family):
    rng = np.random.default_rng(1)
    for seed in range(5):
        clf, _, _ = make_instance(family, 12, seed)
        assert empirical_monotonicity(clf, rng) == 1.0


@pytest.mark.parametrize("family", list(FAMILIES))
def test_every_order_gives_valid_explanations(family):
    rng = np.random.default_rng(2)
    for seed in range(10):
        clf, v, c = make_instance(family, 10, seed)
        for order in (None, cost_desc(c), cost_asc(c), random_order(c, rng)):
            assert check("axp", clf, v, find_axp(clf, v, order), rng)
            cx = find_cxp(clf, v, order)
            if cx is not None:
                assert check("cxp", clf, v, cx, rng)
