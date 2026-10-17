"""Compara dos CSV de submission. ExtraTrees con n_jobs>1 suma en orden no determinista (~1e-16), por eso se compara con tolerancia y no por hash."""
import sys, numpy as np, pandas as pd
a, b = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
assert (a.ID.values == b.ID.values).all() and (a.TargetF1.values == b.TargetF1.values).all()
d = np.abs(a.TargetRAUC.values - b.TargetRAUC.values).max(); assert d < 1e-9, d
print("OK maxdiff", d)
