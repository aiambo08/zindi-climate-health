"""Bloque W: rezagos climáticos semanales (NASA POWER) de 5–12 semanas (Iganga/Mayuge, BMC Public Health 2025: temperatura ≈5–11 semanas,
lluvia ≈11–12 semanas). Datos: data/external/power_daily.csv (src/download_power.py; NASA POWER público: T2M MERRA-2 y PRECTOTCORR; solo clima, sin etiquetas).
Día 1 = día anterior a la muerte (el día de la muerte se excluye). Semana k = días 7(k−1)+1 … 7k antes de la muerte.
  wk:       tmean_wk5..wk12 (media) y rain_wk5..wk12 (suma) -> 16 features
  lit:      t_w5_11 = media tmean días 29–77 ; r_w11_12 = suma lluvia días 71–84
  lit_anom: lit menos la climatología 2001–2006 (anterior a todas las muertes; 2000 se excluye porque sus ventanas empezarían en 1999) de la misma ventana y día del año en la misma coordenada
W1/W2/W3 = estructura s04 (media simple ET30 + LightGBM lento + LR spline de edad) con dt + wk / lit / lit_anom; R4 = misma estructura con dt.
  python weekly.py          -> cribado (semilla 42, splits fijos loc2/dist2) frente a R4 y s04
  python weekly.py confirm W2 -> W2 con semillas 42/43/44 y particiones loc2 alternativas frente a s04
Sin pesos de clase, umbral 0,5, probabilidades originales."""
import sys, time, warnings, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from common import load, features, score, ROOT, T, norm_location, district
from models import make

WK = range(5, 13)
WCOLS = {"wk": [f"tmean_wk{k}" for k in WK] + [f"rain_wk{k}" for k in WK], "lit": ["t_w5_11", "r_w11_12"], "lit_anom": ["t_w5_11_anom", "r_w11_12_anom"]}
def daily_feats(df):
    """Features climáticas diarias por fila a partir de sumas acumuladas diarias por coordenada; solo días anteriores a la muerte."""
    E = pd.read_csv(ROOT/"data/external/power_daily.csv", parse_dates=["date"])
    out = pd.DataFrame(index=df.index, dtype=float)
    dd = pd.to_datetime(df.deathdate)
    for (lat, lon), g in E.groupby(["lat", "lon"]):
        g = g.set_index("date").sort_index(); assert g.index.is_unique and (g.index.to_series().diff().dropna() == pd.Timedelta("1D")).all()
        ct, cr = g.tmean.cumsum(), g.precip.cumsum()
        rows = (df.latitude.values == lat) & (df.longitude.values == lon)
        if not rows.any(): continue
        e = dd[rows]
        def win(cs, end, a, b):   # suma de los días a..b antes de 'end' (día 1 = end − 1 día)
            hi = cs.reindex(end - pd.Timedelta(days=a)).values; lo = cs.reindex(end - pd.Timedelta(days=b + 1)).values
            return hi - lo
        for k in WK:
            out.loc[rows, f"tmean_wk{k}"] = win(ct, e, 7*(k-1)+1, 7*k) / 7; out.loc[rows, f"rain_wk{k}"] = win(cr, e, 7*(k-1)+1, 7*k)
        t, r = win(ct, e, 29, 77) / 49, win(cr, e, 71, 84)
        out.loc[rows, "t_w5_11"], out.loc[rows, "r_w11_12"] = t, r
        clim_t, clim_r = [], []
        for Y in range(2001, 2007):
            eY = pd.to_datetime(pd.DataFrame(dict(year=Y, month=e.dt.month, day=np.where((e.dt.month == 2) & (e.dt.day == 29), 28, e.dt.day))))
            eY.index = e.index; clim_t.append(win(ct, eY, 29, 77) / 49); clim_r.append(win(cr, eY, 71, 84))
        out.loc[rows, "t_w5_11_anom"] = t - np.mean(clim_t, 0); out.loc[rows, "r_w11_12_anom"] = r - np.mean(clim_r, 0)
    assert out.notna().all().all() and len(out.columns) == 20, out.isna().sum()
    return out

OUT = ROOT/"reports/weekly"; OUT.mkdir(parents=True, exist_ok=True)
tr, _ = load(); y = tr[T].values; sp = pd.read_csv(ROOT/"data/splits.csv"); assert (sp.ID.values == tr.ID.values).all()
B = {"dt": ("demo", "time")}
X = {"dt": features(tr, B["dt"])}
W = daily_feats(tr)
for k, cols in WCOLS.items(): X[f"dt_{k}"] = pd.concat([X["dt"], W[cols]], axis=1)
for k, x in X.items(): assert not {"ID", "location", "latitude", "longitude"} & set(x.columns), k
dist = tr.location.map(district).values
FOLDS = {"loc2": sp.loc2.values, "dist2": sp.dist2.values}
for r in (43, 44, 45):
    f = np.zeros(len(y), int)
    for k, (_, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=r).split(X["dt"], y, tr.location.map(norm_location))): f[va] = k
    FOLDS[f"loc2_r{r}"] = f
COMP = ["et30_mf1", "lgbm_slow", "lr_spline_age"]
WBLK = {"W1": "dt_wk", "W2": "dt_lit", "W3": "dt_lit_anom"}
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
def struct(blk, seed, s): return np.mean([oof(f"{m}|{blk}", seed, s) for m in COMP], 0)
if len(sys.argv) == 1:
    SCH = ("loc2", "dist2")
    P = {w: {s: struct(b, 42, s) for s in SCH} for w, b in {**WBLK, "R4": "dt"}.items()}
    P["s04"] = {s: s04(s) for s in SCH}
    TRAIN = {w: score(y, np.mean([fit(m, b, 42, np.ones(len(y), bool), "train").predict_proba(X[b])[:, 1] for m in COMP], 0))
             for w, b in {**WBLK, "R4": "dt"}.items()}
    G, s, E = compare(P, ["R4", "s04"], "screen", TRAIN)
else:   # python weekly.py confirm Wx
    w = sys.argv[2]; P = {}
    for sd in (42, 43, 44):
        P[f"{w}@{sd}"] = {s: struct(WBLK[w], sd, s) for s in FOLDS}; P[f"R4@{sd}"] = {s: struct("dt", sd, s) for s in FOLDS}
    P[f"{w}_3s"] = {s: np.mean([P[f"{w}@{sd}"][s] for sd in (42, 43, 44)], 0) for s in FOLDS}
    P["s04"] = {s: np.mean([P[f"R4@{sd}"][s] for sd in (42, 43, 44)], 0) for s in FOLDS}
    G, s, E = compare(P, ["s04"] + [f"R4@{sd}" for sd in (42, 43, 44)], f"confirm_{w}", {})
    keep = lambda c, r: (r == "s04" and c == f"{w}_3s") or (c.startswith(f"{w}@") and r == "R4@" + c.split("@")[1])
    s = s[[keep(c, r) for c, r in zip(s.cfg, s.ref)]]; E = E[[keep(c, r) for c, r in zip(E.cfg, E.ref)]]
print(G.to_string(index=False)); print(s.to_string(index=False)); print(E[E.scheme.isin(["loc2", "dist2"])].to_string(index=False))
print(f"tiempo total {time.time()-t0:.0f}s")
