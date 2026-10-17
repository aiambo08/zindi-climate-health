"""Diagnóstico: ¿se agrupan las etiquetas en el tiempo dentro de una localidad? (solo train, no feature)"""
import numpy as np, pandas as pd
from common import load, T
tr, te = load(); tr["d"] = pd.to_datetime(tr.deathdate)
tr = tr.sort_values(["location","d"])
for w in [0, 3, 15, 60]:
    r=[]
    for _, g in tr.groupby("location"):
        dd=g.d.values; yy=g[T].values
        for i in range(len(g)):
            m=(np.abs((dd-dd[i]).astype("timedelta64[D]").astype(int))<=w); m[i]=False
            if m.any(): r.append((yy[i], yy[m].mean()))
    r=np.array(r); print("window",w,"n",len(r),"corr(y, mean neighbours y)",np.corrcoef(r.T)[0,1].round(3))
both=pd.concat([tr,te.assign(d=pd.to_datetime(te.deathdate))])
c=both.groupby(["location","deathdate"]).size(); print("records per (loc,date) dist", c.value_counts().head(6).to_dict())
tr["n_same"]=tr.set_index(["location","deathdate"]).index.map(c)
print(tr.groupby(tr.n_same.clip(upper=4))[T].agg(["mean","size"]))
print("age<5 rate vs label by year"); print(tr.groupby(tr.d.dt.year).agg(y=(T,"mean"),u5=("age",lambda a:(a<5).mean())).round(2))
