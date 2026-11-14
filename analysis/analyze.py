"""Builds the result tables (results/summary.md) and figures (figures/*.png)
from the CSV files written by experiments/run_experiments.py.

    python -m analysis.analyze
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")

VARIANTS = ["A_index", "B_random", "C_costdesc", "Abl_costasc", "D_ms_guided", "Dp_ms_random"]
LABEL = {"A_index": "A: index order (paper)", "B_random": "B: random order",
         "C_costdesc": "C: cost-descending", "Abl_costasc": "Ablation: cost-ascending",
         "D_ms_guided": "D: guided multi-start", "Dp_ms_random": "D': random multi-start"}
# fixed colour per variant (categorical palette, fixed order)
COLOR = {"C_costdesc": "#2a78d6", "D_ms_guided": "#eb6834", "A_index": "#1baf7a",
         "B_random": "#eda100", "Dp_ms_random": "#e87ba4", "Abl_costasc": "#008300"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
KIND = {"axp": "AXp", "cxp": "CXp"}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})

out = []          # markdown lines


def md_table(df, floatfmt="{:.3f}"):
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = [str(idx)] + [floatfmt.format(x) if isinstance(x, (float, np.floating)) else str(x)
                              for x in row.values]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def add_reduction(d, ref="A_index"):
    """Paired relative cost reduction vs. the reference variant (same instance)."""
    key = ["family", "n", "seed", "sigma", "kind"]
    base = d[d.variant == ref][key + ["cost"]].rename(columns={"cost": "ref_cost"})
    d = d.merge(base, on=key)
    d["red_vs_A"] = np.where(d.ref_cost > 0, 1 - d.cost / d.ref_cost, 0.0)
    return d


def agg_red(g):
    """1 - (total cost of X) / (total cost of A) over a group of instances."""
    return 1 - g.cost.sum() / g.ref_cost.sum()


def paired_test(d, a, b, kind):
    x = d[(d.kind == kind) & (d.variant == a)].sort_values(["family", "n", "seed", "sigma"]).cost.values
    y = d[(d.kind == kind) & (d.variant == b)].sort_values(["family", "n", "seed", "sigma"]).cost.values
    diff = x - y
    win, tie, loss = int(np.sum(diff < -1e-9)), int(np.sum(np.abs(diff) <= 1e-9)), int(np.sum(diff > 1e-9))
    p = wilcoxon(x, y).pvalue if (win + loss) > 0 else 1.0
    return win, tie, loss, p


def style(ax, ylabel=None, xlabel=None):
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)


# ---------------------------------------------------------------------------
def e0():
    s = json.load(open(os.path.join(RES, "e0_xmono_check.json")))
    out.append("## E0: agreement with the original XMono outputs\n")
    df = pd.DataFrame(s).T[["instances", "axp_match", "cxp_match", "valid", "avg_n_axp", "avg_n_cxp"]]
    df.index.name = "benchmark"
    out.append(md_table(df, "{:.2f}") + "\n")


def e1():
    d = pd.read_csv(os.path.join(RES, "e1.csv"))
    d = add_reduction(d)
    comp = d[d.enum_complete]
    out.append("## E1: main comparison (synthetic, sigma = 1, n in {8,12,16,20}, 100 seeds each)\n")
    out.append(f"Instances: {d.seed.nunique() * 12}. Rows with complete enumeration "
               f"(exact optimum known): {d.enum_complete.mean():.1%}. "
               f"Explanations passing verification: {d.valid.mean():.1%}.\n")
    for kind in ("axp", "cxp"):
        g = comp[comp.kind == kind].groupby("variant").agg(
            pct_optimal=("is_opt", "mean"), median_gap=("gap", "median"),
            mean_gap=("gap", "mean"),
            mean_size=("size", "mean"), calls=("calls", "mean")).loc[VARIANTS]
        g["cost_red_vs_A"] = comp[comp.kind == kind].groupby("variant").apply(agg_red, include_groups=False).loc[VARIANTS]
        g["pct_optimal"] *= 100
        g.index = [LABEL[v] for v in g.index]
        g.index.name = KIND[kind]
        out.append(f"### {KIND[kind]}s\n")
        out.append(md_table(g) + "\n")
        # per family / n (% optimal)
        p = comp[comp.kind == kind].pivot_table(index=["family", "n"], columns="variant",
                                                values="is_opt", aggfunc="mean")[VARIANTS] * 100
        p.index = [f"{f}, n={n}" for f, n in p.index]
        p.index.name = "% optimal"
        p.columns = [c.split("_")[0] for c in p.columns]
        out.append(md_table(p, "{:.0f}") + "\n")

    out.append("### Paired tests (cost, Wilcoxon signed-rank, all E1 instances)\n")
    out.append("| kind | comparison | X cheaper | tie | X more expensive | p-value |\n|---|---|---|---|---|---|")
    for kind in ("axp", "cxp"):
        for a, b in [("C_costdesc", "A_index"), ("C_costdesc", "B_random"),
                     ("D_ms_guided", "Dp_ms_random"), ("D_ms_guided", "C_costdesc"),
                     ("Abl_costasc", "A_index")]:
            w, t, l, p = paired_test(d, a, b, kind)
            out.append(f"| {KIND[kind]} | X={a} vs {b} | {w} | {t} | {l} | {p:.1e} |")
    out.append("")

    # Figure 1: % optimal per variant
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, kind in zip(axes, ("axp", "cxp")):
        vals = comp[comp.kind == kind].groupby("variant").is_opt.mean()[VARIANTS] * 100
        ax.bar(range(len(VARIANTS)), vals.values, width=0.7,
               color=[COLOR[v] for v in VARIANTS], edgecolor="white", linewidth=1)
        for i, x in enumerate(vals.values):
            ax.text(i, x + 1.5, f"{x:.0f}", ha="center", color=INK, fontsize=8)
        ax.set_xticks(range(len(VARIANTS)))
        ax.set_xticklabels(["A", "B", "C", "Abl", "D", "D'"])
        ax.set_title(f"{KIND[kind]}s", fontsize=9.5, color=INK)
        ax.set_ylim(0, 108)
        style(ax, "% of instances at exact minimum cost" if kind == "axp" else None)
    fig.suptitle("E1: how often each variant returns a minimum-cost explanation (K = 8 for multi-start)",
                 fontsize=9.5, color=INK, y=1.04)
    fig.text(0.5, -0.06, "A = index order (paper), B = random, C = cost-descending, Abl = cost-ascending, "
             "D = guided multi-start, D' = random multi-start", ha="center", fontsize=7.5, color=INK2)
    fig.savefig(os.path.join(FIG, "fig1_e1_pct_optimal.png"))
    plt.close(fig)

    # Figure 4 (E4): reduction of C vs A by number of explanations
    bins = [0, 1, 3, 10, 50, np.inf]
    names = ["1", "2-3", "4-10", "11-50", ">50"]
    comp = comp.copy()
    comp["nbin"] = pd.cut(comp.n_expl, bins=bins, labels=names)
    out.append("## E4: does cost guidance help more when there are many explanations?\n")
    cc = comp[comp.variant == "C_costdesc"]
    tab = pd.DataFrame({f"{m}_{KIND[k]}": cc[cc.kind == k].groupby("nbin", observed=False).apply(
        agg_red if m == "cost_red" else len, include_groups=False)
        for k in ("axp", "cxp") for m in ("cost_red", "instances")})
    tab.index.name = "#explanations"
    out.append("Cost reduction of C (cost-descending) vs A (index order), 1 - sum(cost_C)/sum(cost_A), "
               "binned by the number of AXps (resp. CXps) of the instance.\n")
    out.append(md_table(tab, "{:.3f}") + "\n")

    cv = d[(d.variant == "C_costdesc") & d.enum_complete].copy()
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    x = np.arange(len(names))
    for j, kind in enumerate(("axp", "cxp")):
        m = cv[cv.kind == kind].assign(nbin=pd.cut(cv[cv.kind == kind].n_expl, bins=bins, labels=names)) \
            .groupby("nbin", observed=False).red_vs_A.mean().values * 100
        col = "#2a78d6" if kind == "axp" else "#eb6834"
        ax.plot(x, m, marker="o", markersize=5, linewidth=2, color=col, label=f"{KIND[kind]}s")
        ax.text(x[-1] + 0.1, m[-1], KIND[kind] + "s", color=INK, va="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(-0.3, len(names) - 0.4)
    style(ax, "cost reduction of C vs A (%)", "number of explanations of the instance")
    ax.set_title("E4: benefit of cost-descending order grows with the number of explanations",
                 fontsize=9, color=INK)
    fig.savefig(os.path.join(FIG, "fig4_e4_gain_vs_nexpl.png"))
    plt.close(fig)


def e2_budget():
    d = pd.read_csv(os.path.join(RES, "e2_budget.csv"))
    d = d[d.enum_complete]
    out.append("## E2a: multi-start budget K (n = 16, sigma = 1, 3 families x 100 seeds)\n")
    rows = []
    for kind in ("axp", "cxp"):
        for v, lab in (("D_ms_guided", "guided"), ("Dp_ms_random", "random")):
            for K, g in d[(d.kind == kind) & (d.variant == v)].groupby("K"):
                rows.append(dict(kind=KIND[kind], method=lab, K=K, pct_optimal=100 * g.is_opt.mean(),
                                 median_gap=g.gap.median(), calls=g.calls.mean()))
        c = d[(d.kind == kind) & (d.variant == "C_costdesc")]
        rows.append(dict(kind=KIND[kind], method="guided", K=1, pct_optimal=100 * c.is_opt.mean(),
                         median_gap=c.gap.median(), calls=c.calls.mean()))
    t = pd.DataFrame(rows).sort_values(["kind", "method", "K"])
    out.append(md_table(t.set_index("kind"), "{:.2f}") + "\n")
    out.append("(guided K=1 is variant C; random K=1 is a single random order.)\n")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, kind in zip(axes, ("AXp", "CXp")):
        for lab, col, v in (("guided (D)", "#eb6834", "guided"), ("random (D')", "#e87ba4", "random")):
            s = t[(t.kind == kind) & (t.method == v)]
            ax.plot(s.K, s.pct_optimal, marker="o", markersize=5, linewidth=2, color=col, label=lab)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32])
        ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
        ax.set_title(f"{kind}s", fontsize=9.5, color=INK)
        ax.set_ylim(0, 105)
        style(ax, "% at exact minimum cost" if kind == "AXp" else None, "number of runs K (calls = 2nK)")
        ax.legend(frameon=False, loc="lower right")
    fig.suptitle("E2a: guided vs random multi-start at equal classifier-call budget", fontsize=9.5, color=INK, y=1.04)
    fig.savefig(os.path.join(FIG, "fig2_e2_budget.png"))
    plt.close(fig)


def e2_sigma():
    d = add_reduction(pd.read_csv(os.path.join(RES, "e2_sigma.csv")))
    out.append("## E2b: cost heterogeneity sigma (n = 16, K = 8)\n")
    out.append("Cost reduction vs A (index order), 1 - sum(cost_X)/sum(cost_A). sigma = 0 means equal costs.\n")
    t = d[d.variant != "A_index"].groupby(["kind", "sigma", "variant"]).apply(
        agg_red, include_groups=False).unstack("variant")
    t = t[["B_random", "C_costdesc", "D_ms_guided", "Dp_ms_random"]]
    t.index = [f"{KIND[k]}, sigma={s}" for k, s in t.index]
    t.index.name = "setting"
    out.append(md_table(t, "{:.3f}") + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, kind in zip(axes, ("axp", "cxp")):
        for v in ("C_costdesc", "D_ms_guided", "Dp_ms_random", "B_random"):
            s = d[(d.kind == kind) & (d.variant == v)].groupby("sigma").apply(agg_red, include_groups=False) * 100
            ax.plot(s.index, s.values, marker="o", markersize=5, linewidth=2, color=COLOR[v], label=LABEL[v])
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_title(f"{KIND[kind]}s", fontsize=9.5, color=INK)
        style(ax, "cost reduction vs A (%)" if kind == "axp" else None, "cost spread sigma (log-normal)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7.5, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.17))
    fig.suptitle("E2b: cost guidance helps more as feature costs become more unequal", fontsize=9.5, color=INK, y=1.04)
    fig.savefig(os.path.join(FIG, "fig3_e2_sigma.png"))
    plt.close(fig)


def e2_tau():
    d = pd.read_csv(os.path.join(RES, "e2_tau.csv"))
    d = d[d.enum_complete]
    out.append("## E2c: perturbation strength tau (guided multi-start, K = 8, n = 16)\n")
    t = d.pivot_table(index="tau", columns="kind", values=["is_opt", "gap"],
                      aggfunc={"is_opt": "mean", "gap": "median"})
    t.columns = [f"{'pct_optimal' if a == 'is_opt' else 'median_gap'}_{KIND[b]}" for a, b in t.columns]
    for c in t.columns:
        if c.startswith("pct"):
            t[c] *= 100
    t.index.name = "tau"
    out.append(md_table(t, "{:.3f}") + "\n")


def e5():
    d = add_reduction(pd.read_csv(os.path.join(RES, "e5.csv")).assign(sigma=1.0))
    d = d[d.enum_complete]
    out.append("## E5: classifiers from the paper's benchmarks (XMono), random costs (sigma = 1, 5 cost draws)\n")
    for fam in ("StudentGrades", "BankruptcyRisk"):
        f = d[d.family == fam]
        g = f.pivot_table(index="variant", columns="kind", values="is_opt", aggfunc="mean").loc[VARIANTS]
        g.columns = [f"pct_optimal_{KIND[k]}" for k in g.columns]
        for k in ("axp", "cxp"):
            g[f"cost_red_vs_A_{KIND[k]}"] = f[f.kind == k].groupby("variant").apply(
                agg_red, include_groups=False).loc[VARIANTS]
        for c in g.columns:
            if c.startswith("pct"):
                g[c] *= 100
        g.index = [LABEL[v] for v in g.index]
        g.index.name = fam
        n_ax = d[(d.family == fam) & (d.kind == "axp") & (d.variant == "A_index")].n_expl.mean()
        n_cx = d[(d.family == fam) & (d.kind == "cxp") & (d.variant == "A_index")].n_expl.mean()
        out.append(f"Average number of AXps / CXps per instance: {n_ax:.2f} / {n_cx:.2f}\n")
        out.append(md_table(g, "{:.3f}") + "\n")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    out.append("# Results summary (generated by analysis/analyze.py)\n")
    e0(); e1(); e2_budget(); e2_sigma(); e2_tau(); e5()
    open(os.path.join(RES, "summary.md"), "w", encoding="utf-8", newline="\n").write("\n".join(out))
    print("\n".join(out))
