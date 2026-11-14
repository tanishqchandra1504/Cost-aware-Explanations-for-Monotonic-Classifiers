"""Enumeration of all AXps and CXps (Algorithm 3 of the paper), using PySAT.

Variable u_{i+1} = True means feature i is free. Blocking clauses:
  AXp P -> (OR_{i in P} u_i)      (some feature of P must be free)
  CXp P -> (OR_{i in P} -u_i)     (some feature of P must be fixed)
"""
import numpy as np
from pysat.solvers import Solver

from .explain import find_axp, find_cxp


def enumerate_xps(clf, v, max_xps=None, solver_name="g4"):
    """Returns (axps, cxps, complete, n_sat_calls)."""
    v = np.asarray(v, dtype=float)
    n = clf.n
    axps, cxps = [], []
    complete = True
    n_sat = 0
    with Solver(name=solver_name) as s:
        # make sure all variables exist in the solver
        for i in range(1, n + 1):
            s.add_clause([i, -i])
        while True:
            n_sat += 1
            if not s.solve():
                break
            if max_xps is not None and len(axps) + len(cxps) >= max_xps:
                complete = False
                break
            model = s.get_model()
            free = [i for i in range(n) if model[i] > 0]
            fixed = [i for i in range(n) if model[i] < 0]
            vl, vu = v.copy(), v.copy()
            vl[free], vu[free] = clf.lo[free], clf.hi[free]
            if clf(vl) == clf(vu):
                P = find_axp(clf, v, seed=free)
                axps.append(P)
                if not P:          # empty AXp: prediction is constant
                    break
                s.add_clause([i + 1 for i in P])
            else:
                P = find_cxp(clf, v, seed=fixed)
                cxps.append(P)
                s.add_clause([-(i + 1) for i in P])
    return axps, cxps, complete, n_sat
