"""Híbridos de s03 fijados antes de evaluar (sobre la caché OOF de p2.py, 3 semillas):
  H1 = media 1/3 de et30_mf1|full, lgbm_slow|full, lr_spline_age|dt  (s03 sin clima en la LR, cuyo clima absoluto colapsa en Mbale City)
  H2 = media 50/50 de s03 y s04
Comparación emparejada por fold frente a s03 y frente a s04 en loc2, dist2 y loc2_r43/r44/r45. Umbral 0,5, sin pesos de clase.
  python hybrid.py -> reports/hybrid/"""
import numpy as np, pandas as pd
from p2 import FOLDS, y, dist, CACHE, SEEDS, ROOT
from common import score
OUT = ROOT/"reports/hybrid"; OUT.mkdir(parents=True, exist_ok=True)
C = {s: dict(np.load(CACHE/f"p2_s{s}.npz")) for s in SEEDS}
def comp(cfgs, sch): return np.mean([np.mean([C[s][f"out|{c}|{sch}"] for s in SEEDS], 0) for c in cfgs], 0)
F3 = ["et30_mf1|full", "lgbm_slow|full", "lr_spline_age|full"]; D3 = ["et30_mf1|dt", "lgbm_slow|dt", "lr_spline_age|dt"]
P = {"s03": {}, "s04": {}, "H1": {}, "H2": {}}
for sch in FOLDS:
    P["s03"][sch] = comp(F3, sch); P["s04"][sch] = comp(D3, sch)
    P["H1"][sch] = comp(["et30_mf1|full", "lgbm_slow|full", "lr_spline_age|dt"], sch)
    P["H2"][sch] = (P["s03"][sch] + P["s04"][sch]) / 2
    # por semilla (variación del estimador)
    for s in SEEDS:
        one = lambda cf: np.mean([C[s][f"out|{c}|{sch}"] for c in cf], 0)
        P.setdefault(f"s03@{s}", {})[sch] = one(F3); P.setdefault(f"s04@{s}", {})[sch] = one(D3)
        P.setdefault(f"H1@{s}", {})[sch] = one(["et30_mf1|full", "lgbm_slow|full", "lr_spline_age|dt"])
        P.setdefault(f"H2@{s}", {})[sch] = (one(F3) + one(D3)) / 2
PAIRS = [(h, r) for h in ("H1", "H2") for r in ("s03", "s04")] + [(f"{h}@{s}", f"{r}@{s}") for h in ("H1", "H2") for r in ("s03", "s04") for s in SEEDS]
G, Pf, E = [], [], []
for n, r in PAIRS:
    for sch in FOLDS:
        p, q = P[n][sch], P[r][sch]; g, h = score(y, p), score(y, q)
        G.append(dict(cfg=n, ref=r, scheme=sch, f1=round(g["f1"], 4), auc=round(g["auc"], 4), score=round(g["score"], 4), ref_score=round(h["score"], 4),
                      d_f1=round(g["f1"]-h["f1"], 4), d_auc=round(g["auc"]-h["auc"], 4), d_score=round(g["score"]-h["score"], 4), pred_pos=round((p >= .5).mean(), 3)))
        e1, e2 = (p >= .5) != y, (q >= .5) != y
        E.append(dict(cfg=n, ref=r, scheme=sch, labels_changed=int(((p >= .5) != (q >= .5)).sum()), wrong_only_cfg=int((e1 & ~e2).sum()), wrong_only_ref=int((e2 & ~e1).sum())))
        for f in np.unique(FOLDS[sch]):
            va = FOLDS[sch] == f; a, b = score(y[va], q[va]), score(y[va], p[va])
            Pf.append(dict(cfg=n, ref=r, scheme=sch, fold=int(f), group=dist[va][0] if sch == "dist2" else "", n=int(va.sum()),
                           score=round(b["score"], 4), d_score=round(b["score"]-a["score"], 4)))
G, Pf, E = pd.DataFrame(G), pd.DataFrame(Pf), pd.DataFrame(E)
S = Pf.groupby(["cfg", "ref", "scheme"], sort=False).agg(better=("d_score", lambda x: int((x > 0).sum())), k=("d_score", "size"), worst=("score", "min")).reset_index()
W = G.pivot_table(index=["cfg", "ref"], columns="scheme", values="d_score", sort=False)[list(FOLDS)].reset_index()
for d, nm in ((G, "global"), (Pf, "paired_folds"), (S, "paired_summary"), (E, "errors"), (W, "delta_wide")): d.to_csv(OUT/f"{nm}.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print(W.to_string(index=False)); print(G[~G.cfg.str.contains("@") & G.scheme.isin(["loc2", "dist2"])].to_string(index=False))
print(S[~S.cfg.str.contains("@") & S.scheme.isin(["loc2", "dist2"])].to_string(index=False)); print(E[~E.cfg.str.contains("@") & E.scheme.isin(["loc2", "dist2"])].to_string(index=False))
print(Pf[~Pf.cfg.str.contains("@") & (Pf.scheme == "dist2")].to_string(index=False))
