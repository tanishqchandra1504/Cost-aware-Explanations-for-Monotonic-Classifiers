"""Python ports of the two hard-coded classifiers in XMono (scripts/HardCoded.pm)
plus parsers for XMono's spec / instance / log files (data/xmono/).

XMono: https://github.com/jpmarquessilva/xmono (GPL-2.0). Only the data files
are copied; the classifiers below are re-written from the Perl source.
"""
import os
import re

import numpy as np

from mono.core import MonoClassifier

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "xmono")


def read_spec(path):
    toks = open(path).read().split()
    n = int(toks[0])
    names, lo, hi = [], [], []
    for k in range(n):
        names.append(toks[1 + 3 * k])
        lo.append(float(toks[2 + 3 * k]))
        hi.append(float(toks[3 + 3 * k]))
    m = int(toks[1 + 3 * n])
    classes = toks[2 + 3 * n: 2 + 3 * n + m]   # listed from lowest to highest
    return names, np.array(lo), np.array(hi), classes


def read_instance(path):
    toks = open(path).read().split()
    n = int(toks[0])
    return np.array([float(t) for t in toks[1:1 + n]]), toks[1 + n]


def read_log(path):
    """Sets of AXps / CXps reported by XMono (1-indexed -> 0-indexed)."""
    axps, cxps = [], []
    for line in open(path):
        m = re.match(r"^(AXp|CXp):(.*)$", line.strip())
        if m:
            s = sorted(int(t) - 1 for t in m.group(2).split())
            (axps if m.group(1) == "AXp" else cxps).append(s)
    return axps, cxps


# ---- Student grades (running example of the paper) -------------------------

def student_grades_predict(X):
    X = np.atleast_2d(X)
    Q, E, H, R = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    S = np.maximum(0.3 * Q + 0.6 * E + 0.1 * H, R)
    # F=0, E=1, D=2, C=3, B=4, A=5
    return (S >= 2).astype(int) + (S >= 4) + (S >= 5) + (S >= 7) + (S >= 9)


GRADES = ["F", "E", "D", "C", "B", "A"]


def student_grades():
    return MonoClassifier(student_grades_predict, [0] * 4, [10] * 4, "StudentGrades")


# ---- Bankruptcy risk decision tree (rdt_classifier in HardCoded.pm) ---------

def bankruptcy_predict(X):
    X = np.atleast_2d(X)
    x4, x5, x6, x7, x8 = X[:, 4], X[:, 5], X[:, 6], X[:, 7], X[:, 8]
    # ite(x6<=3, ite(x5<=1, ite(x4,1,ite(x7,1,2)), 2), ite(x8<=3, 2, 3))
    inner = np.where(x4 != 0, 1, np.where(x7 != 0, 1, 2))
    left = np.where(x5 <= 1, inner, 2)
    right = np.where(x8 <= 3, 2, 3)
    return np.where(x6 <= 3, left, right).astype(int)


def bankruptcy_risk():
    _, lo, hi, _ = read_spec(os.path.join(DATA, "spec", "BankruptcyRisk", "BR.txt"))
    return MonoClassifier(bankruptcy_predict, lo, hi, "BankruptcyRisk")


def load_benchmark(name):
    """Returns (classifier, list of (inst_id, v, label), dict id -> (axps, cxps))."""
    clf = student_grades() if name == "StudentGrades" else bankruptcy_risk()
    inst_dir = os.path.join(DATA, "inst", name)
    log_dir = os.path.join(DATA, "logs", "xmono", name)
    insts, logs = [], {}
    for f in sorted(os.listdir(log_dir)):
        iid = f.replace(".out", "")
        v, label = read_instance(os.path.join(inst_dir, iid + ".txt"))
        insts.append((iid, v, label))
        logs[iid] = read_log(os.path.join(log_dir, f))
    return clf, insts, logs
