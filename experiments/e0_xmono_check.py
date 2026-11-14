"""E0: check our implementation against the original XMono outputs.

For every StudentGrades / BankruptcyRisk instance that XMono was run on, we
enumerate all AXps and CXps and compare the sets with XMono's log. Since both
enumerations are complete, the sets must be identical (order may differ).
"""
import json
import os

import numpy as np

from mono.enumerate import enumerate_xps
from mono.verify import check
from models.xmono_models import GRADES, load_benchmark

OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def run():
    rng = np.random.default_rng(0)
    summary = {}
    for name in ["StudentGrades", "BankruptcyRisk"]:
        clf, insts, logs = load_benchmark(name)
        rows = []
        for iid, v, label in insts:
            pred = clf.predict_uncounted(v)
            pred_lbl = GRADES[pred] if name == "StudentGrades" else str(pred)
            clf.reset_calls()
            axps, cxps, complete, n_sat = enumerate_xps(clf, v)
            x_ax, x_cx = logs[iid]
            valid = all(check("axp", clf, v, a, rng) for a in axps) and \
                all(check("cxp", clf, v, c, rng) for c in cxps)
            out_of_domain = bool(np.any(v < clf.lo) or np.any(v > clf.hi))
            rows.append(dict(
                inst=iid, label_match=(pred_lbl == label),
                axp_match=sorted(axps) == sorted(x_ax),
                cxp_match=sorted(cxps) == sorted(x_cx),
                n_axp=len(axps), n_cxp=len(cxps), sat_calls=n_sat,
                clf_calls=clf.calls, valid=valid, out_of_domain=out_of_domain))
        s = {k: int(sum(r[k] for r in rows)) for k in
             ["label_match", "axp_match", "cxp_match", "valid", "out_of_domain"]}
        s["instances"] = len(rows)
        s["avg_n_axp"] = float(np.mean([r["n_axp"] for r in rows]))
        s["avg_n_cxp"] = float(np.mean([r["n_cxp"] for r in rows]))
        s["mismatches"] = [r["inst"] for r in rows if not (r["axp_match"] and r["cxp_match"])]
        summary[name] = s
        print(name, json.dumps(s, indent=1))
    os.makedirs(OUT, exist_ok=True)
    json.dump(summary, open(os.path.join(OUT, "e0_xmono_check.json"), "w", newline="\n"), indent=1)


if __name__ == "__main__":
    run()
