"""P2 (experimentos opcionales, ≤60 min c/u). Referencia: s04 = media 1/3 de et30_mf1, lgbm_slow, lr_spline_age sobre demo+time, semillas 42/43/44.
Esquemas: loc2 (principal), dist2 (robustez) y loc2_r43/r44/r45 (particiones alternativas, no validaciones independientes).
  python p2.py dump SEED  -> OOF externos de los 6 componentes de s02/s03/s04 (+2 con interacciones) y, dentro de cada fold externo,
                             OOF internos (StratifiedGroupKFold 5 por localidad loc2, random_state=SEED) solo sobre el train del fold.
                             Caché: data/cache/p2_s{SEED}.npz
  python p2.py eval       -> P1 calibración, P2 interacciones, P3 stacking; informes en reports/p2/
P1 calibración (ajustada con los OOF internos de s04 del fold, aplicada al fold externo; el umbral sigue en 0,5):
  ISOB = isotónica ajustada solo con los puntos internos con p en [0,4, 0,6] y aplicada solo a p en [0,4, 0,6]
  ISOF = isotónica sobre todo el rango; TEMP = temperature scaling (control nulo: deja 0,5 fijo y el orden, no puede cambiar F1 ni AUC)
P2 interacciones (C5/C6 de DIAGNOSTICO.md): dtix = demo+time + age*rural, age*doy_sin, age*doy_cos (edad × mes en forma cíclica).
  ET_ix / LGBM_ix frente al mismo modelo en dt; S04ix = media(ET_ix, LGBM_ix, LR dt) frente a s04.
P3 stacking: regresión logística L2 fuerte (C=0,1, fijado de antemano) sobre logit de los OOF internos de los componentes;
  STK3 = 3 componentes de s04; STK6 = componentes únicos de s02/s03/s04 (dt y full).
Sin pesos de clase, sin ID/localidad/coordenadas como features, probabilidades de los modelos sin modificar salvo la calibración P1."""
import os, sys, time, numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logit, expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from common import load, features, score, ROOT, T, norm_location, district
from models import make

tr, _ = load(); y = tr[T].values; sp = pd.read_csv(ROOT/"data/splits.csv"); assert (sp.ID.values == tr.ID.values).all()
DT, FULL = ("demo", "time"), ("demo", "time", "orig", "clim")
X = {"dt": features(tr, DT), "full": features(tr, FULL)}
d = X["dt"]; X["dtix"] = d.assign(age_x_rural=d.age*d.rural, age_x_doysin=d.age*d.doy_sin, age_x_doycos=d.age*d.doy_cos)
for k, x in X.items(): assert not {"ID", "location", "latitude", "longitude"} & set(x.columns), k
grp = tr.location.map(norm_location).values; dist = tr.location.map(district).values
FOLDS = {"loc2": sp.loc2.values, "dist2": sp.dist2.values}
for r in (43, 44, 45):   # mismas particiones alternativas que frozen_de.py / weekly.py
    f = np.zeros(len(y), int)
    for k, (_, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=r).split(X["dt"], y, grp)): f[va] = k
    FOLDS[f"loc2_r{r}"] = f
S04 = ["et30_mf1|dt", "lgbm_slow|dt", "lr_spline_age|dt"]
FULLC = ["et30_mf1|full", "lgbm_slow|full", "lr_spline_age|full"]
BASE = S04 + FULLC                       # con OOF internos
IX = ["et30_mf1|dtix", "lgbm_slow|dtix"] # solo OOF externos
SEEDS = tuple(int(s) for s in os.environ.get("P2_SEEDS", "42,43,44").split(","))   # P2_SEEDS=42 solo para pruebas rápidas
CACHE = ROOT/"data/cache"; OUT = ROOT/"reports/p2"; CACHE.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)

def fitpred(cfg, seed, tri, tei):
    m, b = cfg.split("|"); mm = make(m, seed).fit(X[b].iloc[tri], y[tri]); assert list(mm.classes_) == [0, 1]
    return mm.predict_proba(X[b].iloc[tei])[:, 1]

def dump(seed):
    C, t0 = {}, time.time()
    for sch, fo in FOLDS.items():
        for cfg in BASE + IX: C[f"out|{cfg}|{sch}"] = np.full(len(y), np.nan)
        for f in np.unique(fo):
            va = np.where(fo == f)[0]; trn = np.where(fo != f)[0]
            inner = list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(trn, y[trn], grp[trn]))
            for cfg in BASE + IX:
                C[f"out|{cfg}|{sch}"][va] = fitpred(cfg, seed, trn, va)
                if cfg in BASE:
                    o = np.full(len(y), np.nan)
                    for a, b in inner: o[trn[b]] = fitpred(cfg, seed, trn[a], trn[b])
                    assert np.isfinite(o[trn]).all() and np.isnan(o[va]).all(); C[f"in|{cfg}|{sch}|{f}"] = o
            print(seed, sch, f, f"{time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(CACHE/f"p2_s{seed}.npz", **C)

def calib(kind, pin, yin, pout):
    if kind == "TEMP":
        L = lambda t: -np.mean(yin*np.log(expit(logit(np.clip(pin, 1e-6, 1-1e-6))/t)) + (1-yin)*np.log(1-expit(logit(np.clip(pin, 1e-6, 1-1e-6))/t)))
        t = minimize_scalar(L, bounds=(0.05, 20), method="bounded").x; return expit(logit(np.clip(pout, 1e-6, 1-1e-6))/t), t
    if kind == "ISOF": return IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(pin, yin).predict(pout), np.nan
    band_in, band_out = (pin >= .4) & (pin <= .6), (pout >= .4) & (pout <= .6); q = pout.copy()
    if band_out.any(): q[band_out] = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(pin[band_in], yin[band_in]).predict(pout[band_out])
    return q, int(band_in.sum())

def evaluate():
    Cs = {s: dict(np.load(CACHE/f"p2_s{s}.npz")) for s in SEEDS}
    P, info = {}, []
    def put(name, sch, p): P.setdefault(name, {})[sch] = p
    for s in SEEDS:
        C = Cs[s]
        for sch, fo in FOLDS.items():
            out = {c: C[f"out|{c}|{sch}"] for c in BASE + IX}
            for c in BASE + IX: put(f"{c}@{s}", sch, out[c])
            put(f"R4@{s}", sch, np.mean([out[c] for c in S04], 0))
            put(f"S04ix@{s}", sch, np.mean([out["et30_mf1|dtix"], out["lgbm_slow|dtix"], out["lr_spline_age|dt"]], 0))
            for k in ("ISOB", "ISOF", "TEMP", "STK3", "STK6"): P.setdefault(f"{k}@{s}", {})[sch] = np.full(len(y), np.nan)
            for f in np.unique(fo):
                va = fo == f; trn = ~va
                inn = {c: C[f"in|{c}|{sch}|{f}"] for c in BASE}
                pin, pout = np.mean([inn[c][trn] for c in S04], 0), np.mean([out[c][va] for c in S04], 0)
                for k in ("ISOB", "ISOF", "TEMP"):
                    q, par = calib(k, pin, y[trn], pout); P[f"{k}@{s}"][sch][va] = q
                    info.append(dict(kind=k, seed=s, scheme=sch, fold=int(f), param=par, labels_changed=int(((q >= .5) != (pout >= .5)).sum())))
                for k, comps in (("STK3", S04), ("STK6", BASE)):
                    Z = lambda D, m: logit(np.clip(np.column_stack([D[c][m] for c in comps]), 1e-4, 1-1e-4))
                    lr = LogisticRegression(C=0.1, max_iter=2000).fit(Z(inn, trn), y[trn]); P[f"{k}@{s}"][sch][va] = lr.predict_proba(Z(out, va))[:, 1]
                    info.append(dict(kind=k, seed=s, scheme=sch, fold=int(f), param=" ".join(f"{v:.3f}" for v in [lr.intercept_[0], *lr.coef_[0]]),
                                     labels_changed=int(((P[f"{k}@{s}"][sch][va] >= .5) != (pout >= .5)).sum())))
    for n in ["R4", "S04ix", "ISOB", "ISOF", "TEMP", "STK3", "STK6"] + BASE + IX:
        P[f"{n}_3s"] = {sch: np.mean([P[f"{n}@{s}"][sch] for s in SEEDS], 0) for sch in FOLDS}
    P["s04"] = P.pop("R4_3s"); P["s02"] = {sch: np.mean([P[f"{c}_3s"][sch] for c in FULLC[:2]], 0) for sch in FOLDS}
    P["s03"] = {sch: np.mean([P[f"{c}_3s"][sch] for c in FULLC], 0) for sch in FOLDS}
    for p in P.values(): assert all(np.isfinite(v).all() for v in p.values())
    pd.DataFrame(info).to_csv(OUT/"fold_info.csv", index=False)
    # pares (candidato, referencia): por semilla frente a su referencia de la misma semilla y 3 semillas frente a s04
    PAIRS = []
    for k in ("ISOB", "ISOF", "TEMP", "STK3", "STK6", "S04ix"):
        PAIRS += [(f"{k}@{s}", f"R4@{s}") for s in SEEDS] + [(f"{k}_3s", "s04")]
    for c in IX:
        r = c.replace("dtix", "dt"); PAIRS += [(f"{c}@{s}", f"{r}@{s}") for s in SEEDS] + [(f"{c}_3s", f"{r}_3s")]
    PAIRS += [("s02", "s04"), ("s03", "s04")]
    glob, paired, err = [], [], []
    for n, r in PAIRS:
        for sch in FOLDS:
            p, q = P[n][sch], P[r][sch]; g, h = score(y, p), score(y, q)
            glob.append(dict(cfg=n, ref=r, scheme=sch, f1=round(g["f1"], 4), auc=round(g["auc"], 4), score=round(g["score"], 4),
                             ref_score=round(h["score"], 4), d_f1=round(g["f1"]-h["f1"], 4), d_auc=round(g["auc"]-h["auc"], 4),
                             d_score=round(g["score"]-h["score"], 4), pred_pos=round((p >= .5).mean(), 3), p_q05=round(np.quantile(p, .05), 3),
                             p_q50=round(np.quantile(p, .5), 3), p_q95=round(np.quantile(p, .95), 3), frac_04_06=round(((p >= .4) & (p <= .6)).mean(), 3)))
            e1, e2 = (p >= .5) != y, (q >= .5) != y
            err.append(dict(cfg=n, ref=r, scheme=sch, corr=round(np.corrcoef(p, q)[0, 1], 4), labels_changed=int(((p >= .5) != (q >= .5)).sum()),
                            wrong_only_cfg=int((e1 & ~e2).sum()), wrong_only_ref=int((e2 & ~e1).sum())))
            for f in np.unique(FOLDS[sch]):
                va = FOLDS[sch] == f; a, b = score(y[va], q[va]), score(y[va], p[va])
                paired.append(dict(cfg=n, ref=r, scheme=sch, fold=int(f), group=dist[va][0] if sch == "dist2" else "", n=int(va.sum()),
                                   score=round(b["score"], 4), ref_score=round(a["score"], 4), d_score=round(b["score"]-a["score"], 4),
                                   d_f1=round(b["f1"]-a["f1"], 4), d_auc=round(b["auc"]-a["auc"], 4)))
    G, Pd, E = pd.DataFrame(glob), pd.DataFrame(paired), pd.DataFrame(err)
    S = Pd.groupby(["cfg", "ref", "scheme"], sort=False).agg(mean_d=("d_score", "mean"), min_d=("d_score", "min"), better=("d_score", lambda x: int((x > 0).sum())),
                                                             k=("d_score", "size")).round(4).reset_index()
    W = G.pivot_table(index=["cfg", "ref"], columns="scheme", values="d_score", sort=False)[list(FOLDS)].reset_index()
    for df, nm in ((G, "global"), (Pd, "paired_folds"), (S, "paired_summary"), (E, "errors"), (W, "delta_wide")): df.to_csv(OUT/f"{nm}.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    print(W.to_string(index=False)); print(S[S.scheme.isin(["loc2", "dist2"])].to_string(index=False))
    print(G[G.scheme.isin(["loc2", "dist2"])].to_string(index=False)); print(E[E.scheme.isin(["loc2", "dist2"])].to_string(index=False))

if __name__ == "__main__":
    t0 = time.time(); dump(int(sys.argv[2])) if sys.argv[1] == "dump" else evaluate(); print(f"tiempo {time.time()-t0:.0f}s")
