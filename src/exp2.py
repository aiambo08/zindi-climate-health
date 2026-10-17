from cv import run
B = ("demo","time","orig","clim")
run("lr_agex", "lr", B + ("agex",)); run("lr_agex_climfe", "lr", B + ("agex","clim_fe"))
run("et_agex", "et", B + ("agex",)); run("et10", "et10", B); run("et20", "et20", B)
run("et_demo_time", "et", ("demo","time")); run("et_climfe", "et", B + ("clim_fe",))
run("lgbm_slow", "lgbm_slow", B); run("lgbm_slow_agex", "lgbm_slow", B + ("agex",))
