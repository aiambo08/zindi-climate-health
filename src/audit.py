"""Fase 1: auditoría de datos. Uso: python src/audit.py"""
import pandas as pd, numpy as np
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "data/raw"
tr = pd.read_csv(R/"Train.csv"); te = pd.read_csv(R/"Test.csv")
cl = pd.read_csv(R/"climate_features.csv"); ss = pd.read_csv(R/"SampleSubmission.csv")
T="is_climate_sensitive"
P=print
P("shapes", tr.shape, te.shape, cl.shape, ss.shape)
P("labels", tr[T].value_counts().to_dict())
P("ID unique", tr.ID.is_unique, te.ID.is_unique, cl.ID.is_unique, "overlap tr/te", len(set(tr.ID)&set(te.ID)))
P("climate covers", set(cl.ID)==set(tr.ID)|set(te.ID))
P("ss order == test order", (ss.ID.values==te.ID.values).all())
P("\nnulls train\n", tr.isna().sum()[tr.isna().sum()>0]); P("nulls test\n", te.isna().sum()[te.isna().sum()>0]); P("nulls clim\n", cl.isna().sum()[cl.isna().sum()>0])
P(tr.dtypes)
for c in ["zone","gender","location"]:
    P("\n",c, tr[c].nunique(), te[c].nunique(), "test-new:", sorted(set(te[c].dropna())-set(tr[c].dropna())))
    if c!="location": P(pd.crosstab(tr[c],tr[T],margins=True)); P(te[c].value_counts())
P(tr.describe().T)
for d in (tr,te): d["dd"]=pd.to_datetime(d.deathdate)
P("\ndate range", tr.dd.min(), tr.dd.max(), te.dd.min(), te.dd.max())
P(pd.crosstab(tr.dd.dt.year, tr[T])); P(te.dd.dt.year.value_counts().sort_index())
P("dates train",tr.deathdate.nunique(),"test",te.deathdate.nunique(),"shared",len(set(tr.deathdate)&set(te.deathdate)), "test rows w/ date in train", te.deathdate.isin(tr.deathdate).mean())
m = pd.concat([tr.assign(s="tr"),te.assign(s="te")]).merge(cl, on="ID", how="left", validate="one_to_one", suffixes=("","_cl"))
P("date match main vs climate", (m.deathdate==m.deathdate_cl).mean())
# temps constant per date?
for c in ["avg_temperature","max_temperature","min_temperature","precipitation"]:
    g=m.groupby("deathdate")[c].nunique(); P(c,"dates with >1 value:",(g>1).sum(),"/",len(g))
g=m.groupby("deathdate").location.nunique(); P("dates with >1 location", (g>1).sum())
for c in ["tavg_7d","rain_sum_30d","ndvi_30d","elevation"]:
    g=m.groupby("deathdate")[c].nunique(); P(c,"dates>1 val",(g>1).sum())
    g=m.groupby(["latitude","longitude"])[c].nunique(); P(c,"coords>1 val",(g>1).sum(), "/", len(g))
P("\nlocations",m.location.nunique(),"coord pairs",m.groupby(["latitude","longitude"]).ngroups)
lc=m.groupby("location")[["latitude","longitude"]].nunique(); P("locations w/ >1 coords\n", lc[(lc>1).any(axis=1)])
cl_=m.groupby(["latitude","longitude"]).location.nunique(); P("coords w/ >1 name", (cl_>1).sum())
P("district", m.location.str.split(", ").str[1].value_counts())
# label by location
lt=tr.groupby("location")[T].agg(["mean","size"]).sort_values("size"); P(lt)
feat=[c for c in tr.columns if c not in("ID",T,"dd")]
P("\nexact dup rows (no ID/label) train", tr.duplicated(feat).sum(), "train+test", pd.concat([tr[feat],te[feat]]).duplicated().sum())
d=tr[tr.duplicated(feat,keep=False)]; P("dup groups label-consistent", d.groupby(feat)[T].nunique().value_counts().to_dict() if len(d) else None)
num=m.select_dtypes("number").drop(columns=[T]); cc=num.corr().abs(); 
pairs=[(a,b,round(cc.loc[a,b],3)) for i,a in enumerate(cc) for b in cc.columns[i+1:] if cc.loc[a,b]>0.95]; P("corr>0.95",pairs)
P("constant cols", [c for c in m if m[c].nunique()<=1])
P("age vs label"); P(tr.groupby(pd.cut(tr.age,[-1,0.99,4,14,29,49,64,79,200]),observed=True)[T].agg(["mean","size"]))
P("month vs label"); P(tr.groupby(tr.dd.dt.month)[T].agg(["mean","size"]))
P("train/test shift (mean)"); P(pd.DataFrame({"tr":m[m.s=="tr"][num.columns].mean(),"te":m[m.s=="te"][num.columns].mean()}))
