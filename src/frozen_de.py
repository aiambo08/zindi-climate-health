"""(1) Evaluación exacta de los candidatos congelados s02/s03/s04 (mismos modelos, bloques y semillas 42,43,44 que
reports/submissions.jsonl) en loc2 (principal) y dist2 (robustez); (2) bloque D (ET leaf 10 vs 30) y E (ET + clima reducido vs sin clima).
Splits fijos de data/splits.csv; variación de partición aparte (StratifiedGroupKFold loc2, random_state 43-45)."""
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from common import load, features, score, ROOT, RAW, T, SEED, norm_location, district
from models import make
AGE_BINS, AGE_LAB = [-1, 0.999, 4, 14, 29, 49, 64, 200], ["<1","1-4","5-14","15-29","30-49","50-64","65+"]  # = temporal.py
PER_BINS, PER_LAB = [0, 2009, 2011, 2013, 2015, 2017, 2019, 3000], ["<=09","10-11","12-13","14-15","16-17","18-19","20+"]

OUT = ROOT/"reports/frozen_de"; OUT.mkdir(parents=True, exist_ok=True)
tr, te = load(); y = tr[T].values; sp = pd.read_csv(ROOT/"data/splits.csv"); assert (sp.ID.values == tr.ID.values).all()
FULL, DT, DTC = ("demo","time","orig","clim"), ("demo","time"), ("demo","time","clim_red")
SEEDS = (42, 43, 44)
CAND = {"s02": (["et30_mf1","lgbm_slow"], FULL), "s03": (["et30_mf1","lgbm_slow","lr_spline_age"], FULL),
        "s04": (["et30_mf1","lgbm_slow","lr_spline_age"], DT)}
FILES = {"s02": "s02_et30mf1_lgbmslow.csv", "s03": "s03_et30mf1_lgbmslow_lrspline.csv", "s04": "s04_dt_et30_lgbm_lrspline.csv"}
Xs = {b: features(tr, b) for b in (FULL, DT, DTC)}; Xt = {b: features(te, b) for b in (FULL, DT, DTC)}
age_g = pd.cut(tr.age, AGE_BINS, labels=AGE_LAB); per_g = pd.cut(pd.to_datetime(tr.deathdate).dt.year, PER_BINS, labels=PER_LAB)
dist = tr.location.map(district).values
alt = {}
for r in (43, 44, 45):
    f = np.zeros(len(y), int)
    for k, (_, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=r).split(Xs[DT], y, tr.location.map(norm_location))): f[va] = k
    alt[f"loc2_r{r}"] = f
FOLDS = {"loc2": sp.loc2.values, "dist2": sp.dist2.values, **alt}

cache = {}
def oof(m, b, s, sch):
    k = (m, b, s, sch)
    if k not in cache:
        o = np.zeros(len(y)); fo = FOLDS[sch]
        for f in np.unique(fo):
            va = fo == f; mm = make(m, s).fit(Xs[b][~va], y[~va]); assert list(mm.classes_) == [0, 1]; o[va] = mm.predict_proba(Xs[b][va])[:, 1]
        cache[k] = o
    return cache[k]
def ens(models, b, seeds, sch): return np.mean([np.mean([oof(m, b, s, sch) for s in seeds], 0) for m in models], 0)

def sc(mask, p):
    yy, pp = y[mask], p[mask]; two = len(set(yy)) == 2; f1 = f1_score(yy, pp >= .5, zero_division=0); auc = roc_auc_score(yy, pp) if two else np.nan
    return dict(n=int(mask.sum()), prev=round(yy.mean(), 3), mean_p=round(pp.mean(), 3), pred_pos=round((pp >= .5).mean(), 3),
                f1=round(f1, 4), auc=round(auc, 4), score=round(.6*f1 + .4*auc, 4) if two else np.nan)

def report(tag, P, ref):
    """P: {nombre: {esquema: oof}}; ref: nombre de referencia para diferencias emparejadas y etiquetas cambiadas."""
    glob, paired, slices, prob = [], [], [], []
    for n, d in P.items():
        for sch, p in d.items():
            g = score(y, p); r = P[ref][sch]
            glob.append(dict(cfg=n, scheme=sch, f1=round(g["f1"], 4), auc=round(g["auc"], 4), score=round(g["score"], 4),
                             d_vs_ref=round(g["score"] - score(y, r)["score"], 4), pred_pos=round((p >= .5).mean(), 3),
                             labels_changed_vs_ref=int(((p >= .5) != (r >= .5)).sum())))
            if sch in ("loc2", "dist2"):
                for f in np.unique(FOLDS[sch]):
                    va = FOLDS[sch] == f; a, b = score(y[va], r[va]), score(y[va], p[va])
                    paired.append(dict(cfg=n, scheme=sch, fold=int(f), group=dist[va][0] if sch == "dist2" else "", n=int(va.sum()),
                                       score=round(b["score"], 4), d_score=round(b["score"]-a["score"], 4), d_f1=round(b["f1"]-a["f1"], 4), d_auc=round(b["auc"]-a["auc"], 4)))
                prob.append(dict(cfg=n, scheme=sch, **{f"q{int(q*100):02d}": round(np.quantile(p, q), 3) for q in (0, .05, .25, .5, .75, .95, 1)},
                                 frac_04_06=round(((p > .4) & (p < .6)).mean(), 3)))
            if sch == "loc2":
                for var, gg, labs in [("edad", age_g, AGE_LAB), ("periodo", per_g, PER_LAB)]:
                    for lv in labs: slices.append(dict(cfg=n, var=var, level=lv, **sc((gg == lv).values, p)))
    for nm, rows in [("global", glob), ("paired_folds", paired), ("age_period", slices), ("prob_dist", prob)]:
        pd.DataFrame(rows).to_csv(OUT/f"{tag}_{nm}.csv", index=False)
    return pd.DataFrame(glob)

# ---------- (1) congelados exactos ----------
fz = OUT/"frozen_repro.txt"; lines = []
for c, (ms, b) in CAND.items():   # reproducción del CSV congelado desde los datos originales
    p = np.mean([np.mean([make(m, s).fit(Xs[b], y).predict_proba(Xt[b])[:, 1] for s in SEEDS], 0) for m in ms], 0)
    f = pd.read_csv(ROOT/"submissions/frozen"/FILES[c]); assert (f.ID.values == te.ID.values).all()
    lines.append(f"{c}: maxdiff TargetRAUC={np.abs(f.TargetRAUC.values - p).max():.2e}, TargetF1 idénticos={(f.TargetF1.values == (p >= .5)).all()}")
open(fz, "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
P = {c: {sch: ens(ms, b, SEEDS, sch) for sch in FOLDS} for c, (ms, b) in CAND.items()}
for c, (ms, b) in CAND.items():   # variación de semilla del estimador: ensemble con una sola semilla (mismos splits)
    for s in SEEDS: P[f"{c}@seed{s}"] = {sch: ens(ms, b, (s,), sch) for sch in ("loc2", "dist2")}
for c, (ms, b) in CAND.items():   # componentes individuales exactos (3 semillas)
    for m in ms: P[f"{c}:{m}"] = {sch: ens([m], b, SEEDS, sch) for sch in ("loc2", "dist2")}
print(report("frozen", P, "s04").query("scheme in ['loc2','dist2'] or cfg in ['s02','s03','s04']").to_string(index=False))

# ---------- (2) D y E: cribado semilla 42; semillas 43/44 y particiones alternativas por separado ----------
Q = {}
for n, (m, b) in {"ref:et30|dt": ("et30_mf1", DT), "D:et10|dt": ("et10_mf1", DT), "E:et30|dt+clim_red": ("et30_mf1", DTC)}.items():
    Q[n] = {sch: oof(m, b, 42, sch) for sch in FOLDS}
    for s in (43, 44): Q[f"{n}@seed{s}"] = {sch: oof(m, b, s, sch) for sch in ("loc2", "dist2")}
print(report("de", Q, "ref:et30|dt").to_string(index=False))
