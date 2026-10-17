from cv import run
B = ("demo","time","orig","clim")
for m in ["dummy","lr","lr_spline_age","et","rf","lgbm","cat"]: run(f"base_{m}", m, B)
run("cat_demo_only", "cat", ("demo",)); run("cat_demo_time", "cat", ("demo","time"))
run("cat_with_geo", "cat", B + ("geo",)); run("cat_no_clim", "cat", ("demo","time","orig"))
