"""SVM RBF con probabilidades Platt (SVC probability=True), solo demo+time y con todas las features."""
from cv import run
S = ("strat", "loc", "date", "dist"); B = ("demo","time","orig","clim"); DT = ("demo","time")
for nm, mdl, bl in [("svm_rbf_dt", "svm_rbf", DT), ("svm_rbf_c03_dt", "svm_rbf_c03", DT),
                    ("svm_rbf_c3_dt", "svm_rbf_c3", DT), ("svm_rbf", "svm_rbf", B)]:
    run(nm, mdl, bl, schemes=S)
