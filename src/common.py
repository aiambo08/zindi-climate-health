"""Carga, unión y features compartidas."""
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
T = "is_climate_sensitive"
SEED = 42

def load():
    tr = pd.read_csv(RAW / "Train.csv"); te = pd.read_csv(RAW / "Test.csv")
    cl = pd.read_csv(RAW / "climate_features.csv")
    out = []
    for d in (tr, te):
        n = len(d); m = d.merge(cl, on="ID", how="left", validate="one_to_one", suffixes=("", "_cl"))
        assert len(m) == n and (m.ID.values == d.ID.values).all()
        assert (m.deathdate == m.deathdate_cl).all()
        out.append(m.drop(columns=["deathdate_cl"]))
    return out

CLIM = ["elevation","max_daily_rain_30d","ndvi_30d","ndvi_90d","rain_days_30d","rain_sum_30d","rain_sum_7d",
        "rain_sum_90d","slope","tavg_30d","tavg_7d","tavg_90d","temp_range_mean_30d","tmax_30d","tmin_30d"]
ORIG = ["avg_temperature","max_temperature","min_temperature","precipitation"]

def features(df, blocks=("demo","time","orig","clim")):
    d = pd.to_datetime(df.deathdate); X = pd.DataFrame(index=df.index)
    if "demo" in blocks:
        X["age"] = df.age; X["male"] = (df.gender == "Male").astype(int); X["rural"] = (df.zone == "Rural").astype(int)
    if "time" in blocks:
        X["year"] = d.dt.year + d.dt.dayofyear / 366; X["doy_sin"] = np.sin(2*np.pi*d.dt.dayofyear/365.25); X["doy_cos"] = np.cos(2*np.pi*d.dt.dayofyear/365.25)
    if "orig" in blocks:
        for c in ORIG: X[c] = df[c]
    if "clim" in blocks:
        for c in CLIM: X[c] = df[c]
    if "agex" in blocks:
        X["log_age"] = np.log1p(df.age); X["u1"] = (df.age < 1).astype(int); X["u5"] = (df.age < 5).astype(int)
        X["age65"] = (df.age >= 65).astype(int); X["age_x_year"] = np.log1p(df.age) * (d.dt.year - 2015)
    if "clim_fe" in blocks:
        X["dtemp_7_90"] = df.tavg_7d - df.tavg_90d; X["dtemp_30_90"] = df.tavg_30d - df.tavg_90d
        X["rain7_rate_vs90"] = df.rain_sum_7d / 7 - df.rain_sum_90d / 90; X["rain30_rate_vs90"] = df.rain_sum_30d / 30 - df.rain_sum_90d / 90
        X["dndvi"] = df.ndvi_30d - df.ndvi_90d; X["trange_day"] = df.max_temperature - df.min_temperature
    if "geo" in blocks:
        X["latitude"] = df.latitude; X["longitude"] = df.longitude
    return X

def score(y, p):
    from sklearn.metrics import f1_score, roc_auc_score
    f1 = f1_score(y, (p >= 0.5).astype(int)); auc = roc_auc_score(y, p)
    return dict(f1=f1, auc=auc, score=0.6*f1 + 0.4*auc)
