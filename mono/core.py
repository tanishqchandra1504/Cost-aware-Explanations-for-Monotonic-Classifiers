"""Basic objects shared by the rest of the code."""
import numpy as np


class MonoClassifier:
    """Black-box wrapper around a monotonic classifier.

    predict_batch maps an (m, n) array to m class indices (ints). Classes are
    assumed to be ordered, i.e. a larger index means a "higher" class.
    lo / hi give the domain bounds of each feature.

    Every call through __call__ is counted, since the number of classifier
    calls is our main measure of effort.
    """

    def __init__(self, predict_batch, lo, hi, name="clf"):
        self.predict_batch = predict_batch
        self.lo = np.asarray(lo, dtype=float)
        self.hi = np.asarray(hi, dtype=float)
        self.n = len(self.lo)
        self.name = name
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return int(self.predict_batch(np.asarray(x, dtype=float)[None, :])[0])

    def predict_uncounted(self, X):
        """Used by the verifier / tests, not counted."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            return int(self.predict_batch(X[None, :])[0])
        return self.predict_batch(X)

    def reset_calls(self):
        self.calls = 0


def expl_cost(expl, costs):
    return float(np.sum(np.asarray(costs)[list(expl)])) if len(expl) else 0.0
