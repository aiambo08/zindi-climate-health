"""Bloque neuronal acotado (MLP pequeño). Uso:
  python mlp.py            -> cribado N1–N4, random_state 42, splits fijos loc2/dist2
  python mlp.py N [N]      -> confirmación de N con semillas 42/43/44 y particiones loc2 alternativas
                              (+ una única media 50/50 con s04 del mejor confirmado, informada por separado)
Pipeline por fold: VarianceThreshold(0) -> StandardScaler -> MLPClassifier(tanh, lbfgs, alpha=1, max_iter=2000,
max_fun=30000, tol=1e-5, sin early stopping). Sin pesos de clase, umbral 0,5, probabilidades originales."""
import sys, time, warnings, numpy as np, pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold
from common import load, features, score, ROOT, T, norm_location, district
from models import make

OUT = ROOT/"reports/mlp"; OUT.mkdir(parents=True, exist_ok=True)
tr, _ = load(); y = tr[T].values; sp = pd.read_csv(ROOT/"data/splits.csv"); assert (sp.ID.values == tr.ID.values).all()
B = {"dt": ("demo", "time"), "dt_climred": ("demo", "time", "clim_red"), "full": ("demo", "time", "orig", "clim")}
X = {k: features(tr, b) for k, b in B.items()}
for k, x in X.items(): assert not {"ID", "location", "latitude", "longitude"} & set(x.columns), k
dist = tr.location.map(district).values
FOLDS = {"loc2": sp.loc2.values, "dist2": sp.dist2.values}
for r in (43, 44, 45):
    f = np.zeros(len(y), int)
    for k, (_, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=r).split(X["dt"], y, tr.location.map(norm_location))): f[va] = k
    FOLDS[f"loc2_r{r}"] = f
CFG = {"N1": "mlp16|dt", "N2": "mlp32_16|dt", "N3": "mlp16|dt_climred", "N4": "mlp16|full"}
LR = "lr_spline_age|dt"
S04 = ["et30_mf1|dt", "lgbm_slow|dt", "lr_spline_age|dt"]
LOG = []

def fit(m, b, seed, idx, tag):
    t = time.time()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mm = make(m, seed).fit(X[b][idx], y[idx])

    if m.startswith("mlp"):
        est = mm[-1]
        cw = [x for x in w if issubclass(x.category, ConvergenceWarning)]
        LOG.append(dict(tag=tag, model=m, blocks=b, seed=seed, n_iter=int(est.n_iter_), max_iter=est.max_iter,
                        converged=not cw, n_conv_warnings=len(cw), other_warnings=len(w) - len(cw),
                        n_features_in=X[b].shape[1], n_features_used=int(mm[0].get_support().sum()), seconds=round(time.time() - t, 2)))
    assert list(mm.classes_) == [0, 1]
    return mm

def oof(cfg, seed, sch):
    m, b = cfg.split("|"); o = np.zeros(len(y)); fo = FOLDS[sch]
    for f in np.unique(fo):
        va = fo == f; o[va] = fit(m, b, seed, ~va, f"{sch}:f{f}").predict_proba(X[b][va])[:, 1]
    return o
def train_metrics(cfg, seed):
    m, b = cfg.split("|"); idx = np.ones(len(y), bool); return score(y, fit(m, b, seed, idx, "train").predict_proba(X[b])[:, 1])
def s04(sch): return np.mean([np.mean([oof(c, sd, sch) for sd in (42, 43, 44)], 0) for c in S04], 0)

def compare(P, refs, tag, TRAIN):
    glob, paired, err = [], [], []
    for n, d in P.items():
        for sch, p in d.items():
            g = score(y, p); row = dict(cfg=n, scheme=sch, **{k: round(v, 4) for k, v in g.items()}, pred_pos=round((p >= .5).mean(), 3),
                                        p_mean=round(p.mean(), 3), p_q05=round(np.quantile(p, .05), 3), p_q95=round(np.quantile(p, .95), 3))
            for r in refs:
                if r in P and sch in P[r]: row[f"d_vs_{r}"] = round(g["score"] - score(y, P[r][sch])["score"], 4)
            if n in TRAIN: row.update({f"train_{k}": round(v, 4) for k, v in TRAIN[n].items()})
            glob.append(row)
            for r in refs:
                if r == n or r not in P or sch not in P[r]: continue
                q = P[r][sch]
                for f in np.unique(FOLDS[sch]):
                    va = FOLDS[sch] == f; a, b = score(y[va], q[va]), score(y[va], p[va])
                    paired.append(dict(cfg=n, ref=r, scheme=sch, fold=int(f), group=dist[va][0] if sch == "dist2" else "", n=int(va.sum()),
                                       score=round(b["score"], 4), ref_score=round(a["score"], 4), d_score=round(b["score"]-a["score"], 4),
                                       d_f1=round(b["f1"]-a["f1"], 4), d_auc=round(b["auc"]-a["auc"], 4)))
                e1, e2 = (p >= .5) != y, (q >= .5) != y
                err.append(dict(cfg=n, ref=r, scheme=sch, corr=round(np.corrcoef(p, q)[0, 1], 4), labels_changed=int(((p >= .5) != (q >= .5)).sum()),
                                wrong_only_cfg=int((e1 & ~e2).sum()), wrong_only_ref=int((e2 & ~e1).sum()), wrong_both=int((e1 & e2).sum())))
    G, Pd, E = pd.DataFrame(glob), pd.DataFrame(paired), pd.DataFrame(err)
    s = Pd.groupby(["cfg", "ref", "scheme"]).agg(mean_d=("d_score", "mean"), min_d=("d_score", "min"), max_d=("d_score", "max"),
                                                  better=("d_score", lambda x: int((x > 0).sum())), k=("d_score", "size"), worst=("score", "min")).round(4).reset_index()
    for d, nm in ((G, "global"), (Pd, "paired_folds"), (s, "paired_summary"), (E, "errors"), (pd.DataFrame(LOG), "convergence")): d.to_csv(OUT/f"{tag}_{nm}.csv", index=False)
    return G, s, E

pd.set_option("display.width", 250); t0 = time.time()
if len(sys.argv) == 1:
    SCH = ("loc2", "dist2"); TRAIN = {n: train_metrics(c, 42) for n, c in list(CFG.items()) + [("LR", LR)]}
    P = {n: {s: oof(c, 42, s) for s in SCH} for n, c in CFG.items()}
    P["LR"] = {s: oof(LR, 42, s) for s in SCH}; P["s04"] = {s: s04(s) for s in SCH}
    G, s, E = compare(P, ["LR", "s04"], "screen", TRAIN)
else:
    names = sys.argv[1:]; P = {}
    for n in names:
        for sd in (42, 43, 44): P[f"{n}@{sd}"] = {s: oof(CFG[n], sd, s) for s in FOLDS}
    P["LR"] = {s: oof(LR, 42, s) for s in FOLDS}; P["s04"] = {s: s04(s) for s in FOLDS}
    best = names[0]   # el primero indicado = mejor confirmado; media 50/50 única, sin pesos optimizados
    P[f"blend50_{best}_s04"] = {s: 0.5 * np.mean([P[f"{best}@{sd}"][s] for sd in (42, 43, 44)], 0) + 0.5 * P["s04"][s] for s in FOLDS}
    G, s, E = compare(P, ["LR", "s04"], "confirm", {})
print(G.to_string(index=False)); print(s.to_string(index=False)); print(E[E.scheme.isin(["loc2", "dist2"])].to_string(index=False))
L = pd.DataFrame(LOG); print(L.groupby(["model", "blocks"]).agg(fits=("seconds", "size"), n_iter_min=("n_iter", "min"), n_iter_max=("n_iter", "max"),
      nonconv=("converged", lambda x: int((~x).sum())), other_w=("other_warnings", "sum"), sec_total=("seconds", "sum"), sec_max=("seconds", "max")).to_string())
print(f"tiempo total {time.time()-t0:.0f}s")
