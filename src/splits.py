"""Fase 2: genera y guarda particiones (data/splits.csv)."""
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from common import load, ROOT, T, SEED
tr, _ = load(); y = tr[T].values
s = pd.DataFrame({"ID": tr.ID})
for name, cv, g in [("strat", StratifiedKFold(5, shuffle=True, random_state=SEED), None),
                    ("loc", StratifiedGroupKFold(5, shuffle=True, random_state=SEED), tr.location),
                    ("date", StratifiedGroupKFold(5, shuffle=True, random_state=SEED), tr.deathdate)]:
    s[name] = -1
    for k, (_, va) in enumerate(cv.split(tr, y, g)): s.loc[va, name] = k
s["year"] = pd.cut(pd.to_datetime(tr.deathdate).dt.year, [0, 2010, 2013, 2016, 2019, 2100], labels=False)
(ROOT/"data").mkdir(exist_ok=True); s.to_csv(ROOT/"data/splits.csv", index=False)
for c in ["strat","loc","date","year"]: print(c, s[c].value_counts().sort_index().to_dict(), tr.groupby(s[c])[T].mean().round(3).to_dict())
