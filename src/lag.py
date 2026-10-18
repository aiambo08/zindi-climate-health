"""Bloque L: ventana climática 31–90 días antes de la muerte (≈ rezagos 5–12 semanas; Iganga/Mayuge, BMC Public Health 2025).
  python lag.py confirm -> L4 con semillas 42/43/44 y particiones loc2 alternativas frente a s04
  python lag.py -> cribado L1–L4 (random_state 42, splits fijos loc2/dist2), cada uno frente a su referencia sin clima R1–R4 y s04.
Features nuevas (bloque clim_lag de common.features, derivadas solo de climate_features.csv, sin etiquetas):
  tavg_31_90d = (90·tavg_90d − 30·tavg_30d)/60 ; rain_sum_31_90d = rain_sum_90d − rain_sum_30d.
L1 ET30, L2 LR spline de edad, L3 LightGBM lento (parámetros de models.make); L4/R4 = media simple de los tres (estructura s04, semilla 42).
Sin pesos de clase, umbral 0,5, probabilidades originales."""
import sys, time, warnings, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from common import load, features, score, ROOT, T, norm_location, district
from models import make

OUT = ROOT/"reports/lag"; OUT.mkdir(parents=True, exist_ok=True)
tr, _ = load(); y = tr[T].values; sp = pd.read_csv(ROOT/"data/splits.csv"); assert (sp.ID.values == tr.ID.values).all()
B = {"dt": ("demo", "time"), "dt_lag": ("demo", "time", "clim_lag"), "dt_climred": ("demo", "time", "clim_red"), "full": ("demo", "time", "orig", "clim")}
X = {k: features(tr, b) for k, b in B.items()}
for k, x in X.items(): assert not {"ID", "location", "latitude", "longitude"} & set(x.columns), k
dist = tr.location.map(district).values
FOLDS = {"loc2": sp.loc2.values, "dist2": sp.dist2.values}
for r in (43, 44, 45):
    f = np.zeros(len(y), int)
    for k, (_, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=r).split(X["dt"], y, tr.location.map(norm_location))): f[va] = k
    FOLDS[f"loc2_r{r}"] = f
CFG = {"L1": "et30_mf1|dt_lag", "L2": "lr_spline_age|dt_lag", "L3": "lgbm_slow|dt_lag"}
REF = {"L1": "et30_mf1|dt", "L2": "lr_spline_age|dt", "L3": "lgbm_slow|dt"}
LR = "lr_spline_age|dt"
S04 = ["et30_mf1|dt", "lgbm_slow|dt", "lr_spline_age|dt"]

def fit(m, b, seed, idx, tag):
    t = time.time()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mm = make(m, seed).fit(X[b][idx], y[idx])

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
    for d, nm in ((G, "global"), (Pd, "paired_folds"), (s, "paired_summary"), (E, "errors")): d.to_csv(OUT/f"{tag}_{nm}.csv", index=False)
    return G, s, E

pd.set_option("display.width", 250); t0 = time.time()
if len(sys.argv) == 1:
    SCH = ("loc2", "dist2"); allc = {**CFG, **{f"R{k[1]}": v for k, v in REF.items()}}
    TRAIN = {n: train_metrics(c, 42) for n, c in allc.items()}
    P = {n: {s: oof(c, 42, s) for s in SCH} for n, c in allc.items()}
    P["L4"] = {s: np.mean([P[k][s] for k in CFG], 0) for s in SCH}; P["R4"] = {s: np.mean([P[f"R{k[1]}"][s] for k in CFG], 0) for s in SCH}  # medias simples, estructura s04 con semilla 42
    P["s04"] = {s: s04(s) for s in SCH}
    G, s, E = compare(P, ["R1", "R2", "R3", "R4", "s04"], "screen", TRAIN)
    pair = {f"L{i}": f"R{i}" for i in range(1, 5)}
    s = s[[pair.get(c) == r or r == "s04" for c, r in zip(s.cfg, s.ref)]]; E = E[[pair.get(c) == r or r == "s04" for c, r in zip(E.cfg, E.ref)]]
else:   # python lag.py confirm -> L4 frente a R4 (= s04) con semillas 42/43/44 por separado y en media, splits fijos + 3 particiones loc2 alternativas
    P = {}
    for sd in (42, 43, 44):
        for k in CFG:
            P[f"{k}@{sd}"] = {s: oof(CFG[k], sd, s) for s in FOLDS}; P[f"R{k[1]}@{sd}"] = {s: oof(REF[k], sd, s) for s in FOLDS}
        P[f"L4@{sd}"] = {s: np.mean([P[f"{k}@{sd}"][s] for k in CFG], 0) for s in FOLDS}
        P[f"R4@{sd}"] = {s: np.mean([P[f"R{k[1]}@{sd}"][s] for k in CFG], 0) for s in FOLDS}
    P["L4_3s"] = {s: np.mean([P[f"L4@{sd}"][s] for sd in (42, 43, 44)], 0) for s in FOLDS}
    P["s04"] = {s: np.mean([P[f"R4@{sd}"][s] for sd in (42, 43, 44)], 0) for s in FOLDS}   # idéntico a s04 (media de semillas por componente)
    P = {k: v for k, v in P.items() if k.startswith(("L4", "R4", "s04"))}
    G, s, E = compare(P, ["s04"] + [f"R4@{sd}" for sd in (42, 43, 44)], "confirm", {})
    keep = lambda c, r: r == "s04" and c in ("L4_3s",) or (c.startswith("L4@") and r == "R4@" + c[3:])
    s = s[[keep(c, r) for c, r in zip(s.cfg, s.ref)]]; E = E[[keep(c, r) for c, r in zip(E.cfg, E.ref)]]
    G = G[G.cfg.isin(["L4_3s", "s04"])]
print(G.to_string(index=False)); print(s.to_string(index=False)); print(E[E.scheme.isin(["loc2", "dist2"])].to_string(index=False))
print(f"tiempo total {time.time()-t0:.0f}s")
