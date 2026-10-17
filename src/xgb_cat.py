"""Bloque manual XGBoost / CatBoost regularizado. Uso:
  python xgb_cat.py            -> cribado semilla 42, splits fijos loc2/dist2
  python xgb_cat.py CFG [CFG]  -> confirmación de CFG ("modelo|bloques") con semillas 42/43/44 y particiones loc2 alternativas
Iteraciones fijas, sin early stopping, sin pesos de clase, umbral 0,5."""
import sys, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from common import load, features, score, ROOT, T, norm_location, district
from models import make

OUT = ROOT/"reports/xgb_cat"; OUT.mkdir(parents=True, exist_ok=True)
tr, _ = load(); y = tr[T].values; sp = pd.read_csv(ROOT/"data/splits.csv"); assert (sp.ID.values == tr.ID.values).all()
B = {"dt": ("demo", "time"), "full": ("demo", "time", "orig", "clim")}
X = {k: features(tr, b) for k, b in B.items()}
dist = tr.location.map(district).values
FOLDS = {"loc2": sp.loc2.values, "dist2": sp.dist2.values}
for r in (43, 44, 45):
    f = np.zeros(len(y), int)
    for k, (_, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=r).split(X["dt"], y, tr.location.map(norm_location))): f[va] = k
    FOLDS[f"loc2_r{r}"] = f
NEW = [f"{m}|{b}" for m in ("xgb_d2", "xgb_d4", "cat_d3_l10") for b in ("dt", "full")]
REF = ["et30_mf1|dt", "lgbm_slow|full", "cat_reg|dt", "cat_reg|full"]   # mejor individual (loc2+dist2) y referencias previas
S04 = ["et30_mf1|dt", "lgbm_slow|dt", "lr_spline_age|dt"]

def oof(cfg, seed, sch):
    m, b = cfg.split("|"); o = np.zeros(len(y)); fo = FOLDS[sch]
    for f in np.unique(fo):
        va = fo == f; mm = make(m, seed).fit(X[b][~va], y[~va]); assert list(mm.classes_) == [0, 1]; o[va] = mm.predict_proba(X[b][va])[:, 1]
    return o
def train_metrics(cfg, seed):
    m, b = cfg.split("|"); mm = make(m, seed).fit(X[b], y); return score(y, mm.predict_proba(X[b])[:, 1])

def compare(P, refs, tag):
    glob, paired, err = [], [], []
    for n, d in P.items():
        for sch, p in d.items():
            g = score(y, p); row = dict(cfg=n, scheme=sch, **{k: round(v, 4) for k, v in g.items()}, pred_pos=round((p >= .5).mean(), 3))
            for r in refs:
                if r in P and sch in P[r]: row[f"d_vs_{r}"] = round(g["score"] - score(y, P[r][sch])["score"], 4)
            if n in TRAIN: row.update({f"train_{k}": round(v, 4) for k, v in TRAIN[n].items()})
            glob.append(row)
            for r in refs:
                if r not in P or sch not in P[r] or not sch in ("loc2", "dist2") and not sch.startswith("loc2_"): continue
                q = P[r][sch]
                for f in np.unique(FOLDS[sch]):
                    va = FOLDS[sch] == f; a, b = score(y[va], q[va]), score(y[va], p[va])
                    paired.append(dict(cfg=n, ref=r, scheme=sch, fold=int(f), group=dist[va][0] if sch == "dist2" else "", n=int(va.sum()),
                                       score=round(b["score"], 4), ref_score=round(a["score"], 4), d_score=round(b["score"]-a["score"], 4),
                                       d_f1=round(b["f1"]-a["f1"], 4), d_auc=round(b["auc"]-a["auc"], 4)))
                # complementariedad de errores (descriptiva, sin combinar)
                e1, e2 = (p >= .5) != y, (q >= .5) != y
                err.append(dict(cfg=n, ref=r, scheme=sch, corr=round(np.corrcoef(p, q)[0, 1], 4), labels_changed=int(((p >= .5) != (q >= .5)).sum()),
                                wrong_only_cfg=int((e1 & ~e2).sum()), wrong_only_ref=int((e2 & ~e1).sum()), wrong_both=int((e1 & e2).sum())))
    G = pd.DataFrame(glob); Pd = pd.DataFrame(paired); E = pd.DataFrame(err)
    G.to_csv(OUT/f"{tag}_global.csv", index=False); Pd.to_csv(OUT/f"{tag}_paired_folds.csv", index=False); E.to_csv(OUT/f"{tag}_errors.csv", index=False)
    s = Pd.groupby(["cfg", "ref", "scheme"]).agg(mean_d=("d_score", "mean"), min_d=("d_score", "min"), max_d=("d_score", "max"),
                                                  better=("d_score", lambda x: int((x > 0).sum())), k=("d_score", "size"), worst=("score", "min")).round(4).reset_index()
    s.to_csv(OUT/f"{tag}_paired_summary.csv", index=False)
    return G, s, E

pd.set_option("display.width", 250)
if len(sys.argv) == 1:
    SCH = ("loc2", "dist2"); TRAIN = {c: train_metrics(c, 42) for c in NEW + REF}
    P = {c: {s: oof(c, 42, s) for s in SCH} for c in NEW + REF}
    P["s04"] = {s: np.mean([np.mean([oof(c, sd, s) for sd in (42, 43, 44)], 0) for c in S04], 0) for s in SCH}   # s04 exacto, 3 semillas
    G, s, E = compare(P, ["et30_mf1|dt", "s04"], "screen")
else:
    cfgs = sys.argv[1:]; TRAIN = {}
    P = {}
    for c in cfgs + ["et30_mf1|dt"]:
        for sd in (42, 43, 44): P[f"{c}@{sd}"] = {s: oof(c, sd, s) for s in FOLDS}
    P["s04"] = {s: np.mean([np.mean([oof(c, sd, s) for sd in (42, 43, 44)], 0) for c in S04], 0) for s in FOLDS}
    G, s, E = compare(P, ["et30_mf1|dt@42", "s04"], "confirm")
print(G.to_string(index=False)); print(s.to_string(index=False)); print(E[E.scheme.isin(["loc2", "dist2"])].to_string(index=False))
