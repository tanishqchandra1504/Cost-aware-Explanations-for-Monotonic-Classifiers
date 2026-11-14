# Project explanation: problem, hypothesis, tests, results, next steps

These are study notes for the team. They explain what the project does, how we
tested it, and what the numbers in `results/summary.md` mean. The numbers come
from `python run_all.py` with the default `--seeds 100`.

---

## 1. The problem

**Monotonic classifiers.** A classifier is monotonic if increasing an input
feature can never lower the predicted class. Credit scoring is an example: a
higher income should never make your risk class worse. Many real models are
built this way on purpose.

**Formal explanations.** The paper explains one prediction κ(v) = c in two
ways:

- **AXp (abductive, "why?")**: a smallest-possible set of features such that,
  if those features keep their current values, the prediction stays c
  *whatever the other features do*. Example: "you got grade A because
  Quiz = 10 and Exam = 10; homework and project don't matter."
- **CXp (contrastive, "why not?")**: a smallest-possible set of features which,
  if allowed to change, *can* change the prediction. Example: "changing just
  the Exam mark could change your grade."

Both are *subset-minimal*: you can't remove any feature and still have a valid
explanation. They are *formal*: guaranteed true, unlike LIME or Anchor.

**The paper's algorithm.** findAXp starts with all features fixed at their
values in v. It goes through the features **one at a time, in index order
1, 2, …, N**. For each feature it asks: "if I let this feature take any value,
can the prediction change?" It answers with just two classifier calls, one on
the lowest corner of the box (vL) and one on the highest (vU). This works only
because the model is monotonic: if those two corners get the same class, every
point between them does too. If the prediction can't change, the feature is
dropped from the explanation; otherwise it stays. findCXp is the mirror image.
The cost is 2N classifier calls, which is very fast.

**The gap we target.** An instance often has **several different AXps or
CXps**, and which one you get depends **only on the order** the features are
processed in. The paper always uses index order, which is arbitrary. But
features aren't equally "cheap" to put in an explanation. Some are sensitive
(age, ethnicity), hard to measure, expensive to verify, or confusing to users.
If several valid explanations exist, we should return one built from cheap
features. The paper never considers this.

---

## 2. Research question and hypothesis

- **Question:** can we pick the feature order using feature costs, plus a few
  extra runs (multi-start), so the returned explanation is cheaper than with
  the paper's order or a random order, while staying formally valid?
- **Hypothesis:** cost guidance helps most when (a) an instance has **several**
  explanations and (b) feature costs **differ a lot**. With only one
  explanation, or equal costs, it should make no difference.

The hypothesis is testable in both directions. We should see **zero** gain in
the "1 explanation" and "equal costs" cases, and a gain that **grows** with the
number of explanations and with cost spread. If gains appeared in the zero
cases, something would be wrong with the reasoning or the code.

---

## 3. Our extension

**C – cost-descending order.** Process the most expensive feature first. Why
first? Features processed early are the easiest to drop. Once a feature is
dropped (made free), there's less slack left, so later features are more
likely to be forced to stay. Trying expensive features first therefore pushes
them *out* of the explanation.

Toy proof from our tests: `x1 OR x2` with costs (10, 1). Order (1, 2) returns
{2} (cost 1); order (2, 1) returns {1} (cost 10).

**D – guided multi-start.** C is greedy and can be wrong. With
`x1 OR (x2 AND x3)` and costs (5, 3, 3), descending order drops x1 first and is
forced to keep {2, 3} (cost 6), but {1} (cost 5) is cheaper. So D runs K
orders: the plain descending one, plus K−1 orders with a little random noise on
the costs (each feature sorted by c_i · exp(τ · e_i), e_i ~ N(0, 1)). It keeps
the cheapest result.

**Validity is free.** Every run of the original algorithm returns a valid,
subset-minimal explanation whatever the order. So C and D can't produce
invalid explanations; we only change *which* valid one is returned. We still
check every output independently (`mono/verify.py`).

---

## 4. How we tested it

### Setup

- **Instances:** each one is (classifier, input point v, feature costs),
  generated from a seed. Every variant runs on *exactly the same* instances, so
  comparisons are paired.
- **Synthetic classifiers** (monotonic by construction, also checked in tests):
  - linear threshold models (tend to have *many* explanations)
  - max of linear models (4 ordered classes)
  - monotone DNF rules
- **Real benchmark classifiers:** the two from XMono, the paper's
  student-grades model and the BankruptcyRisk decision tree.
- **Costs:** log-normal, normalised to mean 1. Its spread σ controls how
  unequal costs are (σ = 0 means all equal).

### Variants (baselines and ours)

| ID  | What | Role |
|-----|------|------|
| A   | index order 1..N | the original paper's method |
| B   | one random order | "does any order beat A?" |
| C   | cost-descending | our main proposal |
| Abl | cost-ascending | ablation: costs used the *wrong* way |
| D   | guided multi-start, K = 8 | our second proposal |
| D′  | random multi-start, K = 8 | fair baseline for D: same number of calls, no cost information |

### Exact reference ("what's the best possible?")

To say how close a variant gets to optimal, we need the true cheapest
explanation. We get it with the paper's own Algorithm 3: it lists *every* AXp
and CXp using a SAT solver, and we take the cheapest. This also tells us *how
many* explanations each instance has, which the hypothesis needs. The listing
is capped at 2,000 explanations; 1.8% of instances hit the cap and are
excluded from the optimality numbers.

### Metrics

- **% optimal:** how often the variant returns a cheapest explanation.
- **Cost reduction vs A:** 1 − (total cost of variant) / (total cost of A).
  Positive means cheaper than the paper.
- **Gap:** cost / optimum − 1, reported as mean and median.
- **Size:** number of features in the explanation.
- **Calls:** number of classifier calls, our measure of effort.
- **Valid:** share of explanations passing the independent checker.
- **Wilcoxon signed-rank test:** a paired statistical test on costs, because
  the same instances are used for every variant.

### Experiments and what each one answers

| Exp | Question it answers |
|-----|---------------------|
| E0  | Is our reimplementation correct? (Without this, nothing else counts.) |
| E1  | Does C/D beat the baselines overall? |
| E2  | How sensitive is it to K (budget), σ (cost spread) and τ (noise)? This is the course's required sensitivity analysis. |
| E3  | Ablation: does the gain really come from the ordering? |
| E4  | The hypothesis directly: does the gain grow with the number of explanations? |
| E5  | Does it hold on the paper's real benchmark classifiers? |

---

## 5. Results, one by one

### E0 – correctness

- Unit tests reproduce Tables 1–3 of the paper exactly, including the class at
  every step (C/A, E/A, A/A…).
- On all 139 XMono instances (100 StudentGrades, 39 BankruptcyRisk), our full
  lists of AXps and CXps are **identical** to the authors' saved outputs.
- Our 1.03 AXps per instance on BankruptcyRisk matches the paper's Table 4b.

**Meaning:** our Python code does what the paper's code does, so our baseline
A *is* the paper's method.

### E1 – main comparison (1,200 instances)

| | AXp % optimal | AXp cost ↓ vs A | CXp % optimal | CXp cost ↓ vs A |
|---|---|---|---|---|
| A (paper)        | 48     | 0       | 25     | 0       |
| B random         | 49     | −0.6%   | 24     | −1.7%   |
| **C cost-desc**  | **89** | **18%** | **91** | **69%** |
| Abl cost-asc     | 34     | −14%    | 3      | −115%   |
| **D guided MS**  | **97** | **19%** | **99** | **70%** |
| D′ random MS     | 80     | 16%     | 79     | 67%     |

How to read it:

- **A ≈ B.** The paper's index order is effectively a random order with
  respect to cost; it finds the cheapest explanation only about half the time
  (AXp) or a quarter of the time (CXp).
- **C nearly doubles the AXp optimal rate and nearly quadruples the CXp rate,
  with the same number of calls.** The "calls" column (27.8 vs 27.8) shows it
  costs nothing extra. CXp calls are 2 higher because findCXp first checks
  that the prediction can change at all.
- **D adds a few more points, but uses 8× the calls.**
- **D beats D′ at the same budget.** So the cost information matters, not just
  re-running.
- **Why CXps gain more than AXps:** CXps here are usually 1–2 features (mean
  size 1.36). So the question is basically "which single feature?", and cheap
  versus expensive single features differ a lot under log-normal costs. AXps
  are larger (about 6 features) and share many features, so there's less room
  to swap.
- **Mean gap vs median gap:** the CXp mean gap for A is 4.69 (469%) while the
  median is 1.53. A few instances where A picks a very expensive single
  feature blow up the mean. That's why we use median, % optimal and the
  total-cost ratio instead.
- **All 100% valid.**
- **Statistics:** C is cheaper than A on 606 instances, ties on 557 and is
  worse on 37 (AXp, p ≈ 10⁻⁹⁵). For CXp it's 886 / 272 / 42 (p ≈ 10⁻¹⁴⁰).
  Ties happen mostly when there's only one explanation. C is worse than A only
  rarely: that's the greedy failure case, where A gets lucky.

### E2 – sensitivity

**Budget K (Fig 2).**

- One guided run (K = 1, 32 calls) is optimal 83% of the time for AXps. Random
  multi-start needs K = 16 (512 calls) to reach 85%.
- For CXps, guided K = 1 gets 91%, while random needs K = 32 to reach 94%.
- **Meaning:** cost guidance is worth about 16× the classifier calls. Guided
  multi-start saturates near 100% by K = 8–16, so K = 8 is a reasonable
  default.

**Cost spread σ (Fig 3).**

- σ = 0 (equal costs): C gives **exactly 0%** gain, as predicted. With equal
  costs, descending order with index tie-breaking *is* index order.
- Gains grow steadily with σ: AXp from 5% to 34%, CXp from 25% to 93%.
- **Meaning:** this confirms part (b) of the hypothesis.
- Side finding: at σ = 0, multi-start still gives 9% (AXp) and 27% (CXp). With
  equal costs, cost = size, so multi-start becomes a search for *smaller*
  explanations, which is a useful bonus.

**Noise τ.**

- % optimal is 88 at τ = 0.1, 94 at 0.5, 96 at τ = 1, and 92 at τ = 2.
- **Meaning:** too little noise repeats almost the same order; too much noise
  throws away the cost information. Moderate noise is best. The results aren't
  very sensitive to τ in the 0.5–1 range.

### E3 – ablation

- Cost-*ascending* is clearly **worse than the paper's order**: AXp 34% vs 48%
  optimal; CXp only 3% vs 25%, with 115% more cost.
- **Meaning:** the direction of the ordering causes the effect. If ordering
  didn't matter, ascending would look like A. Getting systematically worse
  when costs are used the wrong way is strong evidence that our reasoning
  about "processed early = dropped" is right.
- D′ (re-running without costs) only catches up with C at large K. So the
  improvement isn't just "running more times".

### E4 – the hypothesis directly (Fig 4)

| # explanations | AXp cost ↓ (C vs A) | CXp cost ↓ |
|---|---|---|
| 1     | 0%  | 0%  |
| 2–3   | 12% | 54% |
| 4–10  | 21% | 67% |
| 11–50 | 28% | 78% |
| >50   | 32% | 75% |

- One explanation gives **exactly zero** gain. That must be so, since every
  order returns the only explanation, and it confirms nothing is broken.
- For AXps the gain grows steadily with the number of explanations. For CXps
  it rises then flattens around 75–78%: it's already near the maximum possible
  saving.
- **Meaning:** this confirms part (a) of the hypothesis. Note that the CXp
  "1 explanation" bin has only 6 instances, so don't lean on it.

### E5 – the paper's real classifiers

- **StudentGrades** has 1.23 AXps per instance; **BankruptcyRisk** has 1.03.
  Almost every instance has just *one* AXp, so there's almost nothing to gain
  on AXps: 87 → 99% and 99 → 100% optimal.
- **CXps** still have about 2–3 options, and C goes from about 52% to 98–100%
  optimal, with 43–46% lower cost.
- **Meaning:** the method behaves on real models exactly as E4 predicts: no
  gain where there's no choice, a large gain where there is.

---

## 6. Verdict on the hypothesis

**Supported.**

1. Cost-guided ordering makes explanations substantially cheaper without
   losing validity (E1, 100% verified).
2. The gain is zero exactly where the hypothesis says it should be: one
   explanation, or equal costs.
3. The gain grows with both the number of explanations (E4) and the cost
   spread (E2).
4. The ablation shows the ordering is the cause (E3).

Exactly as the proposal stated, we do **not** guarantee the global minimum. C
misses it about 10% of the time, and D about 1–3% of the time.

---

## 7. Limitations and threats to validity

These go in the report's discussion; they also count for marks.

- **Synthetic costs:** they model "burden", not real-world recourse or
  actionability.
- **Mostly synthetic classifiers.** The linear family is hardest: C is optimal
  on only about 60% of instances there when n ≥ 16. We didn't test trained
  monotonic models such as gradient boosting with monotonic constraints,
  because of time.
- **Exact optimum missing for 1.8% of instances** (the capped ones). These are
  the hardest instances, so the linear-family optimality numbers are slightly
  optimistic.
- **The MaxSAT exact solver was too slow** (minutes per instance), so we
  switched to full enumeration. That's an honest negative result worth a
  sentence.
- **Cheaper doesn't mean smaller.** D′ returns slightly *smaller* explanations
  than D (AXp size 5.96 vs 6.05) but more *expensive* ones. Cost and size are
  different goals.
- **BankruptcyRisk data quirk:** 14 instances have value 5 while XMono's spec
  says the domain is [1, 4]. We kept XMono's spec so E0 compares like with
  like. Also, 3 instances where the tree's prediction differs from the dataset
  label are fine: we explain the model's prediction, not the true label.
- **Runtime is negligible** because these classifiers are cheap, so we compare
  classifier calls rather than seconds.
- **Small oddity:** at σ = 0, a random order gave CXps about 5% cheaper
  (smaller) than index order. With equal costs there's no reason to expect
  this, so it's probably noise on 300 instances; don't over-interpret it.

---

## 8. What's still left

1. **The report (4–6 pages):**

   | Section | Length | Contents |
   |---|---|---|
   | Intro and paper summary | 0.75 p | |
   | Our extension | 0.75 p | the two toy examples |
   | Setup | 0.5 p | variants table, metrics, exact reference |
   | Results | 2 p | E0 in two lines; E1 table plus Fig 1; Figs 2 and 3 for E2; E3 in a paragraph; Fig 4 for E4; E5 in two lines |
   | Limitations | 0.5 p | from section 7 above |
   | AI disclosure and repo link | short | |

   Most of the wording can come from `README.md`.
2. **Complete the "Use of AI tools" section** in the README and the report:
   say who did what and how you checked it. It's required, and it's likely to
   come up in the viva.
3. **GitHub:** create a public repo and push the project. Leave out the three
   course PDFs, and put the repo link in the report.
4. **Viva prep, for both of you:**
   - Run `find_axp` by hand on (10, 10, 5, 0) using Table 1.
   - Explain why descending order works and when it fails (the
     x1 OR (x2 AND x3) example).
   - Explain why the bound test needs monotonicity.
   - Explain what the Wilcoxon test checks.
   - Explain why D′ is the fair baseline for D.
5. **Optional, if time is left:** one trained monotonic model (scikit-learn
   `HistGradientBoostingClassifier` with `monotonic_cst`) as E5b. It would
   answer the "only synthetic models" limitation, but it isn't required to
   submit.
