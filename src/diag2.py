"""Diagnóstico parte 2 (lee reports/diag/oof_diag.csv): robustez por tipo de novedad, errores OOF por subgrupo, ablación de ensembles."""
import itertools, numpy as np, pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from common import load, score, ROOT, T, norm_location, district
OUT = ROOT/"reports/diag"; tr, _ = load(); y = tr[T].values
o = pd.read_csv(OUT/"oof_diag.csv"); assert (o.ID.values == tr.ID.values).all()
sp = pd.read_csv(ROOT/"data/splits.csv")
loc2 = tr.location.map(norm_location); dist = tr.location.map(district); nloc = loc2.map(loc2.value_counts())
def sc(mask, p):
    yy, pp = y[mask], p[mask]
    auc = roc_auc_score(yy, pp) if len(set(yy)) == 2 else np.nan; f1 = f1_score(yy, pp >= .5)
    return dict(n=int(mask.sum()), pos=round(yy.mean(),3), pred_pos=round((pp>=.5).mean(),3), f1=round(f1,4), auc=round(auc,4), score=round(.6*f1+.4*auc,4), brier=round(np.mean((pp-yy)**2),4))
E = {"s02~": ["et30_mf1|full", "lgbm_slow|full"], "s03~": ["et30_mf1|full", "lgbm_slow|full", "lr_spline_age|full"],
     "s04~": ["et30_mf1|dt", "lgbm_slow|dt", "lr_spline_age|dt"]}
def ens(names, sch): return np.mean([o[f"{n}|{sch}"].values for n in names], axis=0)
# 6) ablación de ensembles en loc2 y dist2
rows = []
for sch in ("loc", "loc2", "dist2"):
    ind = {c.rsplit("|",1)[0]: o[c].values for c in o.columns if c.endswith("|"+sch)}
    best = max((n for n in ind if not n.startswith("dummy")), key=lambda n: score(y, ind[n])["score"])
    cand = {"best_individual:"+best: [best]}
    for f in ("dt", "full"):
        for a, b in [("et30_mf1","lr_spline_age"), ("lgbm_slow","lr_spline_age"), ("et30_mf1","lgbm_slow"), ("cat_reg","lr_spline_age"), ("et30_mf1","cat_reg")]:
            cand[f"pair:{a}+{b}|{f}"] = [f"{a}|{f}", f"{b}|{f}"]
    for k, v in E.items():
        cand[k] = v
        for drop in v: cand[f"{k} -{drop}"] = [x for x in v if x != drop]
    for k, v in cand.items():
        s = score(y, ens(v, sch)); rows.append(dict(scheme=sch, combo=k, **{a: round(b,4) for a, b in s.items()}))
pd.DataFrame(rows).to_csv(OUT/"ensembles.csv", index=False)
# 3) robustez (loc2 OOF): distrito visto en train del fold vs no; tamaño de localidad. dist2 OOF por distrito retenido.
rob = []
for name, v in [("s03~", E["s03~"]), ("s04~", E["s04~"]), ("et30_mf1|dt", ["et30_mf1|dt"]), ("lr_spline_age|dt", ["lr_spline_age|dt"])]:
    p = ens(v, "loc2")
    seen = np.array([(dist[(sp.loc2 != sp.loc2[i]).values] == dist[i]).any() for i in range(len(y))])
    for lab, m in [("loc2: todo", np.ones(len(y), bool)), ("loc2: localidad nueva, distrito conocido", seen), ("loc2: distrito sin otras localidades en train", ~seen),
                   ("loc2: localidad <60 filas", (nloc < 60).values), ("loc2: localidad 60-119", ((nloc >= 60) & (nloc < 120)).values), ("loc2: localidad >=120", (nloc >= 120).values)]:
        rob.append(dict(model=name, slice=lab, **sc(m, p)))
    p = ens(v, "dist2")
    rob.append(dict(model=name, slice="dist2: todo", **sc(np.ones(len(y), bool), p)))
    for d in sorted(dist.unique()): rob.append(dict(model=name, slice=f"dist2: {d} retenido", **sc((dist == d).values, p)))
pd.DataFrame(rob).to_csv(OUT/"robustness.csv", index=False)
# 5) errores OOF por subgrupo (s04~ y s03~, loc2)
d = pd.to_datetime(tr.deathdate)
grp = {"age": pd.cut(tr.age, [-1, 0.99, 4, 14, 29, 49, 64, 200], labels=["<1","1-4","5-14","15-29","30-49","50-64","65+"]),
       "gender": tr.gender, "zone": tr.zone, "year": pd.cut(d.dt.year, [0,2009,2011,2013,2015,2017,2019,2100], labels=["<=09","10-11","12-13","14-15","16-17","18-19","20+"]),
       "location": loc2, "district": dist}
err = []
for name in ("s03~", "s04~"):
    p = ens(E[name], "loc2")
    for g, s in grp.items():
        for lev in s.dropna().unique():
            m = (s == lev).values
            if m.sum() >= 20: err.append(dict(model=name, var=g, level=str(lev), mean_p=round(p[m].mean(),3), fn=int(((p<.5)&(y==1))[m].sum()), fp=int(((p>=.5)&(y==0))[m].sum()), **sc(m, p)))
pd.DataFrame(err).to_csv(OUT/"errors.csv", index=False)
print("ok")
