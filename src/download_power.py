"""Descarga series diarias NASA POWER (API pública sin clave; T2M = temperatura media a 2 m de MERRA-2, PRECTOTCORR = precipitación
corregida) para cada coordenada única de Train+Test, 2000-01-01..2022-12-31 (2000–2006 = climatología previa a todas las muertes).
Solo clima: no usa etiquetas. Caché por coordenada en data/external/power_cache/. Salida: data/external/power_daily.csv
(lat, lon, date, tmean, precip). Se usa en lugar de ERA5/Open-Meteo porque el límite diario de esa API impidió completar las 55 coordenadas."""
import time, json, urllib.request, hashlib, pandas as pd
from common import load, ROOT
OUT = ROOT/"data/external"; CA = OUT/"power_cache"; CA.mkdir(parents=True, exist_ok=True)
tr, te = load(); C = pd.concat([tr, te])[["latitude", "longitude"]].drop_duplicates().sort_values(["latitude", "longitude"])
URL = ("https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,PRECTOTCORR&community=AG"
       "&longitude={:.6f}&latitude={:.6f}&start=20000101&end=20221231&format=JSON")
rows = []
for lat, lon in C.itertuples(index=False):
    f = CA/f"{lat:.7f}_{lon:.7f}.json"
    if not f.exists():
        for a in range(8):
            try: r = json.load(urllib.request.urlopen(URL.format(lon, lat), timeout=180)); break
            except Exception as e: print("retry", lat, lon, e, flush=True); time.sleep(30 * (a + 1))
        else: raise SystemExit(f"fallo {lat},{lon}")
        f.write_text(json.dumps(r)); time.sleep(1)
    r = json.loads(f.read_text()); p = r["properties"]["parameter"]
    d = pd.DataFrame(dict(lat=lat, lon=lon, date=pd.to_datetime(list(p["T2M"]), format="%Y%m%d"), tmean=list(p["T2M"].values()), precip=list(p["PRECTOTCORR"].values())))
    assert len(d) == 8401 and not (d[["tmean", "precip"]] == -999).any().any(), (lat, lon)
    rows.append(d); print(lat, lon, r["geometry"]["coordinates"][:2], flush=True)
D = pd.concat(rows); D.to_csv(OUT/"power_daily.csv", index=False)
print(len(D), D.groupby(["lat", "lon"]).ngroups, hashlib.sha256((OUT/"power_daily.csv").read_bytes()).hexdigest())
