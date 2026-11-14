"""Main experiments (E1, E2, E5). Results are written to results/<exp>.csv.

Usage:
    python -m experiments.run_experiments --exp all
    python -m experiments.run_experiments --exp e1 --seeds 100

Every instance (classifier, point, costs) is generated from its seed, so the
same seed gives the same instance across variants and across experiments
(only the costs change in the sigma sweep).
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from mono.core import expl_cost
from mono.enumerate import enumerate_xps

from mono.explain import explain
from mono.multistart import multistart
from mono.orderings import cost_asc, cost_desc, index_order, random_order
from mono.verify import check
from models.synthetic import make_costs, make_instance
from models.xmono_models import load_benchmark

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
KINDS = ("axp", "cxp")
ENUM_CAP = 2000          # stop enumerating after this many explanations


def run_variant(kind, clf, v, costs, variant, rng, K=8, tau=0.5):
    clf.reset_calls()
    if variant == "A_index":
        e = explain(kind, clf, v, index_order(costs))
    elif variant == "B_random":
        e = explain(kind, clf, v, random_order(costs, rng))
    elif variant == "C_costdesc":
        e = explain(kind, clf, v, cost_desc(costs))
    elif variant == "Abl_costasc":
        e = explain(kind, clf, v, cost_asc(costs))
    elif variant == "D_ms_guided":
        e = multistart(kind, clf, v, costs, K, rng, guided=True, tau=tau)
    elif variant == "Dp_ms_random":
        e = multistart(kind, clf, v, costs, K, rng, guided=False)
    else:
        raise ValueError(variant)
    return e, clf.calls


def evaluate(clf, v, costs, variants, meta, verify=True):
    """Run all variants on one instance, for AXps and CXps. Returns rows.

    The exact reference is the cheapest explanation in the complete list
    produced by Algorithm 3 (enumeration). We first used the MaxSAT hitting-set
    solver in mono/exact.py for this, but RC2 was far too slow on some linear
    instances with n = 20 (minutes vs. 0.1 s for enumeration). If enumeration
    hits ENUM_CAP the optimum is unknown and the row gets opt_cost = NaN.
    """
    rows = []
    clf.reset_calls()
    t0 = time.perf_counter()
    axps, cxps, complete, _ = enumerate_xps(clf, v, max_xps=ENUM_CAP)
    enum_time, enum_calls = time.perf_counter() - t0, clf.calls
    for kind in KINDS:
        xps = axps if kind == "axp" else cxps
        if not xps:              # no CXp: prediction is constant everywhere
            continue
        opt = min(xps, key=lambda e: (expl_cost(e, costs), len(e)))
        opt_cost = expl_cost(opt, costs) if complete else float("nan")
        opt_time, opt_calls = enum_time, enum_calls
        for j, (variant, K, tau) in enumerate(variants):
            rng = np.random.default_rng([meta.get("seed", 0), j, 7])
            t0 = time.perf_counter()
            e, calls = run_variant(kind, clf, v, costs, variant, rng, K, tau)
            rt = time.perf_counter() - t0
            cost = expl_cost(e, costs)
            rows.append(dict(
                meta, kind=kind, variant=variant, K=K, tau=tau,
                cost=cost, size=len(e), calls=calls, runtime=rt,
                valid=check(kind, clf, v, e, rng, 128) if verify else None,
                opt_cost=opt_cost, opt_size=len(opt), opt_calls=opt_calls, opt_time=opt_time,
                is_opt=bool(cost <= opt_cost + 1e-9) if complete else None,
                gap=((cost / opt_cost - 1.0) if opt_cost > 0 else 0.0) if complete else float("nan"),
                n_expl=len(xps), enum_complete=complete,
                cost_cv=float(np.std(costs) / np.mean(costs))))
    return rows


BASE_VARIANTS = [("A_index", 1, None), ("B_random", 1, None), ("C_costdesc", 1, None),
                 ("Abl_costasc", 1, None), ("D_ms_guided", 8, 0.5), ("Dp_ms_random", 8, None)]


# ---- task definitions --------------------------------------------------------

def task_synthetic(args):
    family, n, seed, sigma, variants = args
    clf, v, costs = make_instance(family, n, seed, sigma)
    meta = dict(family=family, n=n, seed=seed, sigma=sigma)
    return evaluate(clf, v, costs, variants, meta)


def task_benchmark(args):
    name, idx, cost_seed = args
    clf, insts, _ = load_benchmark(name)
    iid, v, _ = insts[idx]
    costs = make_costs(clf.n, np.random.default_rng([cost_seed, idx]), 1.0)
    meta = dict(family=name, n=clf.n, seed=cost_seed * 1000 + idx, sigma=1.0, inst=iid)
    return evaluate(clf, v, costs, BASE_VARIANTS, meta)


def build_tasks(exp, n_seeds):
    fams = ["linear", "maxlin", "dnf"]
    if exp == "e1":
        return task_synthetic, [(f, n, s, 1.0, BASE_VARIANTS)
                                for f in fams for n in (8, 12, 16, 20) for s in range(n_seeds)]
    if exp == "e2_budget":
        var = [("C_costdesc", 1, None)]
        for K in (2, 4, 8, 16, 32):
            var += [("D_ms_guided", K, 0.5), ("Dp_ms_random", K, None)]
        var += [("Dp_ms_random", 1, None)]
        return task_synthetic, [(f, 16, s, 1.0, var) for f in fams for s in range(n_seeds)]
    if exp == "e2_sigma":
        var = [("A_index", 1, None), ("B_random", 1, None), ("C_costdesc", 1, None),
               ("D_ms_guided", 8, 0.5), ("Dp_ms_random", 8, None)]
        return task_synthetic, [(f, 16, s, sg, var) for f in fams
                                for sg in (0.0, 0.25, 0.5, 1.0, 2.0) for s in range(n_seeds)]
    if exp == "e2_tau":
        var = [("D_ms_guided", 8, t) for t in (0.1, 0.25, 0.5, 1.0, 2.0)]
        return task_synthetic, [(f, 16, s, 1.0, var) for f in fams for s in range(n_seeds)]
    if exp == "e5":
        tasks = []
        for name, n_inst in (("StudentGrades", 100), ("BankruptcyRisk", 39)):
            tasks += [(name, i, cs) for i in range(n_inst) for cs in range(5)]
        return task_benchmark, tasks
    raise ValueError(exp)


def run(exp, n_seeds, workers):
    fn, tasks = build_tasks(exp, n_seeds)
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(fn, tasks, chunksize=4):
            rows.extend(r)
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, f"{exp}.csv"), index=False, lineterminator="\n")
    print(f"{exp}: {len(tasks)} instances, {len(df)} rows, {time.time() - t0:.1f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="all")
    p.add_argument("--seeds", type=int, default=100)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = p.parse_args()
    exps = ["e1", "e2_budget", "e2_sigma", "e2_tau", "e5"] if a.exp == "all" else [a.exp]
    for e in exps:
        run(e, a.seeds, a.workers)
