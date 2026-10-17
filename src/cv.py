"""Harness de validación: OOF por esquema, métricas globales y por fold, registro."""
import json, time, hashlib
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix
from common import load, features, score, ROOT, T, SEED
from models import make

def run(name, model, blocks, schemes=("strat","loc","date"), save=True, verbose=True, seeds=(SEED,)):
    tr, te = load(); y = tr[T].values; X = features(tr, blocks); sp = pd.read_csv(ROOT/"data/splits.csv")
    assert (sp.ID.values == tr.ID.values).all()
    res = {"name": name, "model": model, "blocks": list(blocks), "seeds": list(seeds)}
    oofs = {}
    for sch in schemes:
        oof = np.zeros(len(y)); fs = []; t0 = time.time()
        for k in sorted(sp[sch].unique()):
            va = sp[sch].values == k; p = 0
            for s in seeds:
                m = make(model, s).fit(X[~va], y[~va]); assert list(m.classes_) == [0, 1]
                p = p + m.predict_proba(X[va])[:, 1] / len(seeds)
            oof[va] = p; fs.append(score(y[va], p))
        g = score(y, oof); tn, fp, fn, tp = confusion_matrix(y, oof >= .5).ravel()
        res[sch] = {**{k: round(v, 4) for k, v in g.items()},
                    "fold_mean": round(np.mean([f["score"] for f in fs]), 4), "fold_std": round(np.std([f["score"] for f in fs]), 4),
                    "cm": [int(tn), int(fp), int(fn), int(tp)], "pos_rate": round(float((oof >= .5).mean()), 3), "sec": round(time.time()-t0, 1)}
        oofs[sch] = oof
    if verbose:
        print(name, " | ".join(f"{s}: S={res[s]['score']:.4f} F1={res[s]['f1']:.4f} AUC={res[s]['auc']:.4f} ±{res[s]['fold_std']:.3f} pos={res[s]['pos_rate']}" for s in schemes), flush=True)
    if save:
        d = ROOT/"oof"; d.mkdir(exist_ok=True)
        pd.DataFrame({"ID": tr.ID, **{f"oof_{s}": o for s, o in oofs.items()}}).to_csv(d/f"{name}.csv", index=False)
        with open(ROOT/"reports/experiments.jsonl", "a") as f: f.write(json.dumps(res) + "\n")
    return res, oofs

def fit_predict(model, blocks, seeds=(SEED,)):
    tr, te = load(); y = tr[T].values; X = features(tr, blocks); Xt = features(te, blocks); p = 0
    for s in seeds:
        m = make(model, s).fit(X, y); assert list(m.classes_) == [0, 1]; p = p + m.predict_proba(Xt)[:, 1] / len(seeds)
    return te.ID.values, p
