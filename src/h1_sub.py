"""Sonda opcional H1 (docs/HIBRIDO_S03.md): media 1/3 de et30_mf1 y lgbm_slow sobre demo+time+orig+clim y lr_spline_age sobre demo+time,
semillas 42/43/44, entrenado en todo el train. No cumple el criterio de candidato; se genera solo como sonda a petición.
Antes reproduce s03 con el mismo código y lo compara con el CSV congelado.  python h1_sub.py"""
import hashlib, json
import numpy as np, pandas as pd
from common import load, features, ROOT
from models import make
from predict import check
tr, te = load(); y = tr["is_climate_sensitive"].values
FULL, DT, SEEDS = ("demo","time","orig","clim"), ("demo","time"), (42, 43, 44)
Xs = {b: features(tr, b) for b in (FULL, DT)}; Xt = {b: features(te, b) for b in (FULL, DT)}
P = {(m, b): np.mean([make(m, s).fit(Xs[b], y).predict_proba(Xt[b])[:, 1] for s in SEEDS], 0)
     for m, b in [("et30_mf1", FULL), ("lgbm_slow", FULL), ("lr_spline_age", FULL), ("lr_spline_age", DT)]}
s03 = np.mean([P["et30_mf1", FULL], P["lgbm_slow", FULL], P["lr_spline_age", FULL]], 0)
fz = pd.read_csv(ROOT/"submissions/frozen/s03_et30mf1_lgbmslow_lrspline.csv"); assert (fz.ID.values == te.ID.values).all()
print(f"s03 repro: maxdiff={np.abs(fz.TargetRAUC.values - s03).max():.2e}, TargetF1 idénticos={(fz.TargetF1.values == (s03 >= .5)).all()}")
p = np.mean([P["et30_mf1", FULL], P["lgbm_slow", FULL], P["lr_spline_age", DT]], 0)
sub = pd.DataFrame({"ID": te.ID.values, "TargetF1": (p >= .5).astype(int), "TargetRAUC": p}); check(sub)
f = ROOT/"submissions/s05_h1_et30full_lgbmfull_lrdt.csv"; sub.to_csv(f, index=False)
h = hashlib.sha256(f.read_bytes()).hexdigest()
with open(ROOT/"reports/submissions.jsonl", "a") as fh:
    fh.write(json.dumps({"file": f.name, "sha256_16": h[:16], "models": ["et30_mf1|full", "lgbm_slow|full", "lr_spline_age|dt"],
                         "blocks": "mixed", "seeds": list(SEEDS), "pos_rate": round(float(sub.TargetF1.mean()), 3), "note": "sonda H1"}) + "\n")
print(f.name, h, "pos_rate", round(sub.TargetF1.mean(), 3), "etiquetas distintas de s03:", int((sub.TargetF1.values != fz.TargetF1.values).sum()),
      "corr TargetRAUC con s03:", round(np.corrcoef(p, fz.TargetRAUC.values)[0, 1], 4))
