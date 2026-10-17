from cv import run
B = ("demo","time","orig","clim")
run("et30_mf1", "et30_mf1", B); run("et10_mf1", "et10_mf1", B)
run("et20_mf1_agex", "et20_mf1", B + ("agex",)); run("et20_mf1_dto", "et20_mf1", ("demo","time","orig"))
run("lgbm_slow_s3", "lgbm_slow", B, seeds=(42,43,44)); run("et20_mf1_s3", "et20_mf1", B, seeds=(42,43,44))
