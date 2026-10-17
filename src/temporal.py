"""Bloque temporal A/B/C (una semilla; LR lbfgs es determinista). Validación principal loc2, robustez dist2.
Todas las transformaciones (splines, escalado) se ajustan dentro de cada fold vía Pipeline.
Grupos de edad y periodos predefinidos (los mismos de diag2.py, fijados antes de este bloque)."""
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from common import load, features, score, ROOT, T, SEED, norm_location, district

OUT = ROOT/"reports/temporal"; OUT.mkdir(parents=True, exist_ok=True)
AGE_BINS, AGE_LAB = [-1, 0.999, 4, 14, 29, 49, 64, 200], ["<1","1-4","5-14","15-29","30-49","50-64","65+"]
PER_BINS, PER_LAB = [0, 2009, 2011, 2013, 2015, 2017, 2019, 3000], ["<=09","10-11","12-13","14-15","16-17","18-19","20+"]

class AgeSplineXYear(BaseEstimator, TransformerMixin):
    """Base spline de edad (mismos nodos que A) multiplicada por el año estandarizado; ajustado en el fold."""
    def __init__(self, n_knots=6): self.n_knots = n_knots
    def fit(self, X, y=None):
        X = np.asarray(X, float); self.s_ = SplineTransformer(n_knots=self.n_knots, degree=3).fit(X[:, :1])
        self.mu_, self.sd_ = X[:, 1].mean(), X[:, 1].std(); return self
    def transform(self, X):
        X = np.asarray(X, float); return self.s_.transform(X[:, :1]) * ((X[:, 1:2] - self.mu_) / self.sd_)

def cfg(name):
    tf = [("age", SplineTransformer(n_knots=6, degree=3), ["age"])]
    if name in ("B", "C"): tf.append(("year", SplineTransformer(n_knots=4, degree=3), ["year"]))
    if name == "C": tf.append(("age_x_year", AgeSplineXYear(6), ["age", "year"]))
    return make_pipeline(ColumnTransformer(tf, remainder=StandardScaler()), LogisticRegression(C=1.0, max_iter=3000))

DESC = {"A": "spline edad (6 nodos, grado 3 → 8 bases, sin escalar) + year lineal estandarizado + male, rural, doy_sin, doy_cos estandarizados; C=1 (L2)",
        "B": "A + year como spline (4 nodos, grado 3 → 6 bases) en lugar de lineal; C=1",
        "C": "B + 8 términos edad_spline × year_estandarizado; C=1"}

tr, _ = load(); y = tr[T].values; sp = pd.read_csv(ROOT/"data/splits.csv"); assert (sp.ID.values == tr.ID.values).all()
X = features(tr, ("demo", "time")); yr = pd.to_datetime(tr.deathdate).dt.year
age_g = pd.cut(tr.age, AGE_BINS, labels=AGE_LAB); per_g = pd.cut(yr, PER_BINS, labels=PER_LAB)
dist = tr.location.map(district).values

def sc(m, p):
    yy, pp = y[m], p[m]; two = len(set(yy)) == 2
    f1 = f1_score(yy, pp >= .5, zero_division=0); auc = roc_auc_score(yy, pp) if two else np.nan
    return dict(n=int(m.sum()), pos=int(yy.sum()), prev=round(yy.mean(), 3), mean_p=round(pp.mean(), 3),
                pred_pos=round((pp >= .5).mean(), 3), f1=round(f1, 4), auc=round(auc, 4) if two else np.nan,
                score=round(.6*f1 + .4*auc, 4) if two else np.nan)

def run_oof(name, folds):
    oof = np.zeros(len(y))
    for k in np.unique(folds):
        va = folds == k; oof[va] = cfg(name).fit(X[~va], y[~va]).predict_proba(X[va])[:, 1]
    return oof

# --- 4. tabla OOF por edad y periodo, modelos ya existentes (loc2, sin reentrenar) ---
od = pd.read_csv(ROOT/"reports/diag/oof_diag.csv"); assert (od.ID.values == tr.ID.values).all()
ref = {"lr_spline_age|dt": od["lr_spline_age|dt|loc2"].values, "et30_mf1|dt": od["et30_mf1|dt|loc2"].values,
       "s04~ (1 semilla)": od[["et30_mf1|dt|loc2", "lgbm_slow|dt|loc2", "lr_spline_age|dt|loc2"]].mean(1).values}
rows = []
for mn, p in ref.items():
    for var, g in [("edad", age_g), ("periodo", per_g)]:
        for lv in (AGE_LAB if var == "edad" else PER_LAB): rows.append(dict(model=mn, var=var, level=lv, **sc((g == lv).values, p)))
    for a in AGE_LAB:
        for q in PER_LAB: rows.append(dict(model=mn, var="edad×periodo", level=f"{a}|{q}", **sc(((age_g == a) & (per_g == q)).values, p)))
pd.DataFrame(rows).to_csv(OUT/"age_period_existing.csv", index=False)

# --- 5. A/B/C con splits fijos loc2 (principal) y dist2 (robustez) ---
oofs = {(n, s): run_oof(n, sp[s].values) for n in "ABC" for s in ("loc2", "dist2")}
glob, paired, byd, agep, prob = [], [], [], [], []
for s in ("loc2", "dist2"):
    for n in "ABC":
        p = oofs[(n, s)]; g = score(y, p)
        chg = int(((p >= .5) != (oofs[("A", s)] >= .5)).sum())
        glob.append(dict(cfg=n, scheme=s, f1=round(g["f1"], 4), auc=round(g["auc"], 4), score=round(g["score"], 4),
                         pred_pos=round((p >= .5).mean(), 3), labels_changed_vs_A=chg, desc=DESC[n]))
        prob.append(dict(cfg=n, scheme=s, **{f"q{int(q*100):02d}": round(np.quantile(p, q), 3) for q in (0, .05, .25, .5, .75, .95, 1)},
                         frac_04_06=round(((p > .4) & (p < .6)).mean(), 3), std=round(p.std(), 3)))
        for k in sorted(sp[s].unique()):
            va = sp[s].values == k; a, b = score(y[va], oofs[("A", s)][va]), score(y[va], p[va])
            paired.append(dict(cfg=n, scheme=s, fold=int(k), n=int(va.sum()), district=dist[va][0] if s == "dist2" else "",
                               score_A=round(a["score"], 4), score=round(b["score"], 4), d_score=round(b["score"]-a["score"], 4),
                               d_f1=round(b["f1"]-a["f1"], 4), d_auc=round(b["auc"]-a["auc"], 4)))
        if s == "loc2":
            for var, gg, labs in [("edad", age_g, AGE_LAB), ("periodo", per_g, PER_LAB)]:
                for lv in labs: agep.append(dict(cfg=n, var=var, level=lv, **sc((gg == lv).values, p)))
pd.DataFrame(glob).to_csv(OUT/"abc_global.csv", index=False); pd.DataFrame(paired).to_csv(OUT/"abc_paired_folds.csv", index=False)
pd.DataFrame(agep).to_csv(OUT/"abc_age_period.csv", index=False); pd.DataFrame(prob).to_csv(OUT/"abc_prob_dist.csv", index=False)

# --- 6. variación de particiones agrupadas (LR determinista: la variación de semilla del estimador es 0) ---
g2 = tr.location.map(norm_location).values; rep = []
for r in range(1, 6):
    f = np.zeros(len(y), int)
    for k, (_, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=SEED + r).split(X, y, g2)): f[va] = k
    sA = score(y, run_oof("A", f))["score"]
    for n in "BC":
        o = run_oof(n, f); rep.append(dict(partition_seed=SEED + r, cfg=n, score_A=round(sA, 4), score=round(score(y, o)["score"], 4),
                                          d_score=round(score(y, o)["score"] - sA, 4)))
pd.DataFrame(rep).to_csv(OUT/"abc_partition_repeats.csv", index=False)
# comprobación: A reproduce lr_spline_age|dt|loc2 de diag
print("A == lr_spline_age dt loc2:", np.abs(oofs[("A", "loc2")] - od["lr_spline_age|dt|loc2"].values).max())
