"""Diagnóstico (bloque 3): modelos individuales train/OOF por fold, robustez por tipo de novedad,
errores OOF por subgrupo y ablación de ensembles. Esquemas: loc (antiguo), loc2 (Brac fusionado), dist2 (distrito retenido)."""
import json, numpy as np, pandas as pd
from common import load, features, score, ROOT, T, SEED, norm_location, district
from models import make
OUT = ROOT/"reports/diag"; OUT.mkdir(parents=True, exist_ok=True)
tr, _ = load(); y = tr[T].values; sp = pd.read_csv(ROOT/"data/splits.csv"); assert (sp.ID.values == tr.ID.values).all()
B = ("demo","time","orig","clim"); DT = ("demo","time")
MODELS = ["dummy","lr","lr_spline_age","et30_mf1","rf20","lgbm_slow","cat_reg","svm_rbf_c03"]
SCH = ["loc","loc2","dist2"]
rows, folds, params, oofs = [], [], {}, {}
for fs_name, bl in [("full", B), ("dt", DT)]:
    X = features(tr, bl)
    for mn in MODELS:
        m0 = make(mn, SEED); params[mn] = {k: str(v) for k, v in m0.get_params().items() if "__" not in k or mn.startswith(("lr","svm"))}
        full = make(mn, SEED).fit(X, y); ptr = full.predict_proba(X)[:, 1]; st = score(y, ptr)
        for sch in SCH:
            oof = np.zeros(len(y)); trs = []
            for k in sorted(sp[sch].unique()):
                va = sp[sch].values == k; m = make(mn, SEED).fit(X[~va], y[~va]); assert list(m.classes_) == [0, 1]
                oof[va] = m.predict_proba(X[va])[:, 1]; ptk = m.predict_proba(X[~va])[:, 1]
                f = score(y[va], oof[va]); ft = score(y[~va], ptk)
                folds.append(dict(model=mn, feats=fs_name, scheme=sch, fold=int(k), n=int(va.sum()), **{f"oof_{a}": round(b, 4) for a, b in f.items()},
                                  **{f"train_{a}": round(b, 4) for a, b in ft.items()}))
            g = score(y, oof); oofs[(mn, fs_name, sch)] = oof
            fsc = [r["oof_score"] for r in folds if (r["model"], r["feats"], r["scheme"]) == (mn, fs_name, sch)]
            rows.append(dict(model=mn, feats=fs_name, scheme=sch, train_f1=round(st["f1"],4), train_auc=round(st["auc"],4), train_score=round(st["score"],4),
                             oof_f1=round(g["f1"],4), oof_auc=round(g["auc"],4), oof_score=round(g["score"],4),
                             fold_mean=round(np.mean(fsc),4), fold_std=round(np.std(fsc),4), fold_min=round(min(fsc),4), pos_rate=round((oof>=.5).mean(),3)))
            print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(OUT/"individual.csv", index=False); pd.DataFrame(folds).to_csv(OUT/"folds.csv", index=False)
json.dump(params, open(OUT/"params.json","w"), indent=1)
o = pd.DataFrame({"ID": tr.ID}); 
for (mn, fs_name, sch), v in oofs.items(): o[f"{mn}|{fs_name}|{sch}"] = v
o.to_csv(OUT/"oof_diag.csv", index=False)
