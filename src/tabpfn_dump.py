"""Ajusta TabPFN v2 por fold y guarda las probabilidades OOF/train (se ejecuta en el venv aislado de TabPFN,
ver docs/TABPFN.md; TabPFN 2.2.1 exige scikit-learn<1.7, incompatible con requirements.txt).
Uso: ~/venv-tabpfn/bin/python tabpfn_dump.py CFG SEED SCH [SCH ...]   (CFG en T1/T2; SCH en reports/tabpfn/folds.npz)
Pesos públicos Prior-Labs/TabPFN-v2-clf (tabpfn-v2-classifier.ckpt); sin ajuste de hiperparámetros, sin pesos de clase."""
import sys, time, json, numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from tabpfn import TabPFNClassifier
from common import load, features, ROOT, T

OUT = ROOT/"reports/tabpfn"
B = {"T1": ("demo", "time"), "T2": ("demo", "time", "orig", "clim")}
cfg, seed, schs = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
tr, _ = load(); y = tr[T].values; X = features(tr, B[cfg]); FO = np.load(OUT/"folds.npz")
def mk(): return make_pipeline(VarianceThreshold(0.0), TabPFNClassifier(n_estimators=4, device="cpu", random_state=seed, ignore_pretraining_limits=True))
def fit(idx):
    t = time.time(); m = mk().fit(X[idx], y[idx]); assert list(m.classes_) == [0, 1]; return m, time.time() - t
log = []
for sch in schs:
    fo = FO[sch]; o = np.full(len(y), np.nan)
    for f in np.unique(fo):
        va = fo == f; m, s = fit(~va); t = time.time(); o[va] = m.predict_proba(X[va])[:, 1]
        log.append(dict(cfg=cfg, seed=seed, scheme=sch, fold=int(f), n_train=int((~va).sum()), fit_s=round(s, 1), pred_s=round(time.time() - t, 1)))
        print(log[-1], flush=True)
    assert not np.isnan(o).any(); np.save(OUT/f"oof_{cfg}_s{seed}_{sch}.npy", o)
if "train" in schs or not schs: pass
m, s = fit(np.ones(len(y), bool)); np.save(OUT/f"train_{cfg}_s{seed}.npy", m.predict_proba(X)[:, 1])
log.append(dict(cfg=cfg, seed=seed, scheme="train", fold=-1, n_train=len(y), fit_s=round(s, 1), pred_s=-1))
with open(OUT/f"log_{cfg}_s{seed}.jsonl", "a") as fh:
    for r in log: fh.write(json.dumps(r) + "\n")
