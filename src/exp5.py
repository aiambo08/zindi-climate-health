"""Fase 2: modelos aditivos suaves (splines edad/año) y CV por distrito (dist)."""
from cv import run
S = ("strat", "loc", "date", "dist"); B = ("demo","time","orig","clim"); DT = ("demo","time")
for nm, mdl, bl in [("lr_spline_age", "lr_spline_age", B), ("lr_spline_age_dt", "lr_spline_age", DT),
                    ("lr_spl_ay", "lr_spl_ay", B), ("lr_spl_ay_dt", "lr_spl_ay", DT),
                    ("lr_spl_ay_c01_dt", "lr_spl_ay_c01", DT), ("lr_spl_ay_int_dt", "lr_spl_ay_int", DT),
                    ("lr_spl_ay_int", "lr_spl_ay_int", B),
                    ("et30_mf1", "et30_mf1", B), ("et30_mf1_dt", "et30_mf1", DT),
                    ("lgbm_slow", "lgbm_slow", B), ("lgbm_slow_dt", "lgbm_slow", DT)]:
    run(nm + "_v2", mdl, bl, schemes=S)
