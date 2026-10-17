"""Genera un CSV de submission validado. Uso: python predict.py NOMBRE MODELO[,MODELO...] BLOQUES [SEEDS]
Ej: python predict.py et_base et demo,time,orig,clim 5"""
import sys, hashlib, json
import numpy as np, pandas as pd
from common import ROOT, RAW, SEED
from cv import fit_predict

def check(sub):
    ss = pd.read_csv(RAW/"SampleSubmission.csv")
    assert list(sub.columns) == ["ID","TargetF1","TargetRAUC"] and len(sub) == 1030 == len(ss)
    assert (sub.ID.values == ss.ID.values).all() and sub.ID.is_unique
    assert set(sub.TargetF1.unique()) <= {0, 1}
    assert np.isfinite(sub.TargetRAUC).all() and sub.TargetRAUC.between(0, 1).all()
    assert (sub.TargetF1.values == (sub.TargetRAUC.values >= 0.5).astype(int)).all()

def main(name, models, blocks, nseeds=1):
    seeds = tuple(SEED + i for i in range(nseeds)); ps = []
    for m in models:
        ids, p = fit_predict(m, blocks, seeds); ps.append(p)
    p = np.mean(ps, axis=0)
    sub = pd.DataFrame({"ID": ids, "TargetF1": (p >= 0.5).astype(int), "TargetRAUC": p}); check(sub)
    d = ROOT/"submissions"; d.mkdir(exist_ok=True); f = d/f"{name}.csv"; sub.to_csv(f, index=False)
    h = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    with open(ROOT/"reports/submissions.jsonl", "a") as fh:
        fh.write(json.dumps({"file": f.name, "sha256_16": h, "models": models, "blocks": blocks, "seeds": list(seeds),
                             "pos_rate": round(float(sub.TargetF1.mean()), 3)}) + "\n")
    print(f, h, "pos_rate", sub.TargetF1.mean().round(3))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(","), tuple(sys.argv[3].split(",")), int(sys.argv[4]) if len(sys.argv) > 4 else 1)
