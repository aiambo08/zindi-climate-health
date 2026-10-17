"""Diferencias emparejadas por fold (mismos splits fijos) de modelos/ensembles existentes frente a s04~ (aprox. 1 semilla)."""
import numpy as np, pandas as pd
from common import load, score, ROOT, T
tr, _ = load(); y = tr[T].values; sp = pd.read_csv(ROOT/"data/splits.csv"); od = pd.read_csv(ROOT/"reports/diag/oof_diag.csv")
assert (od.ID.values == tr.ID.values).all() and (sp.ID.values == tr.ID.values).all()
C = {"et30_mf1|dt": ["et30_mf1|dt"], "lgbm_slow|full": ["lgbm_slow|full"], "lr_spline_age|dt": ["lr_spline_age|dt"],
     "et30+lr_spline|dt": ["et30_mf1|dt", "lr_spline_age|dt"], "lgbm+lr_spline|dt": ["lgbm_slow|dt", "lr_spline_age|dt"],
     "s02~|full": ["et30_mf1|full", "lgbm_slow|full"], "s03~|full": ["et30_mf1|full", "lgbm_slow|full", "lr_spline_age|full"],
     "s04~|dt": ["et30_mf1|dt", "lgbm_slow|dt", "lr_spline_age|dt"]}
rows = []
for s in ("loc2", "dist2"):
    P = {k: od[[f"{c}|{s}" for c in v]].mean(1).values for k, v in C.items()}
    for k, p in P.items():
        d = []; fs = []
        for f in sorted(sp[s].unique()):
            va = sp[s].values == f; a = score(y[va], p[va])["score"]; d.append(a - score(y[va], P["s04~|dt"][va])["score"]); fs.append(a)
        rows.append(dict(scheme=s, combo=k, score=round(score(y, p)["score"], 4), d_vs_s04=round(score(y, p)["score"] - score(y, P["s04~|dt"])["score"], 4),
                         fold_d_mean=round(np.mean(d), 4), fold_d_min=round(min(d), 4), fold_d_max=round(max(d), 4),
                         folds_better=int(sum(x > 0 for x in d)), n_folds=len(d), worst_fold=round(min(fs), 4),
                         per_fold=" ".join(f"{x:.4f}" for x in fs)))
pd.DataFrame(rows).to_csv(ROOT/"reports/diag/paired_vs_s04.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
