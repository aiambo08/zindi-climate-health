"""Evalúa promedios simples de probabilidades OOF (sin pesos optimizados). Uso: python ens.py a,b,c [a,b ...]"""
import sys, numpy as np, pandas as pd
from common import load, score, ROOT, T
tr, _ = load(); y = tr[T].values
for combo in sys.argv[1:]:
    names = combo.split(","); out = []
    for sch in ["strat","loc","date"]:
        p = np.mean([pd.read_csv(ROOT/f"oof/{n}.csv")[f"oof_{sch}"].values for n in names], axis=0); s = score(y, p)
        out.append(f"{sch}: S={s['score']:.4f} F1={s['f1']:.4f} AUC={s['auc']:.4f} pos={(p>=.5).mean():.3f}")
    print(combo, " | ".join(out))
