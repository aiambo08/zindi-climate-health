"""Fábrica de modelos con configuraciones explícitas (sin búsqueda automática)."""
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def _spl(cols, C=1.0, k=6, inter=False):
    tf = [(c, SplineTransformer(n_knots=k, degree=3), [c]) for c in cols]
    steps = [ColumnTransformer(tf, remainder=StandardScaler())]
    if inter: steps.append(PolynomialFeatures(2, interaction_only=True, include_bias=False))
    return make_pipeline(*steps, StandardScaler(), LogisticRegression(C=C, max_iter=5000))

def make(name, seed=42):
    if name == "svm_rbf": return make_pipeline(StandardScaler(), SVC(C=1.0, gamma="scale", probability=True, random_state=seed))
    if name == "svm_rbf_c03": return make_pipeline(StandardScaler(), SVC(C=0.3, gamma="scale", probability=True, random_state=seed))
    if name == "svm_rbf_c3": return make_pipeline(StandardScaler(), SVC(C=3.0, gamma="scale", probability=True, random_state=seed))
    if name == "lr_spl_ay": return _spl(["age", "year"])
    if name == "lr_spl_ay_c01": return _spl(["age", "year"], C=0.1)
    if name == "lr_spl_ay_int": return _spl(["age", "year"], C=0.05, k=5, inter=True)
    if name == "dummy": return DummyClassifier(strategy="prior")
    if name == "lr": return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    if name == "lr_spline_age":
        ct = ColumnTransformer([("age", SplineTransformer(n_knots=6, degree=3), ["age"])], remainder=StandardScaler())
        return make_pipeline(ct, LogisticRegression(C=1.0, max_iter=3000))
    if name == "et": return ExtraTreesClassifier(500, min_samples_leaf=5, max_features=0.5, n_jobs=8, random_state=seed)
    if name == "et20": return ExtraTreesClassifier(1000, min_samples_leaf=20, max_features=0.5, n_jobs=8, random_state=seed)
    if name == "et10": return ExtraTreesClassifier(1000, min_samples_leaf=10, max_features=0.5, n_jobs=8, random_state=seed)
    if name == "lgbm_slow": return LGBMClassifier(n_estimators=600, learning_rate=0.01, num_leaves=7, min_child_samples=50,
                                             subsample=0.7, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0,
                                             random_state=seed, n_jobs=8, verbose=-1)
    if name == "et40": return ExtraTreesClassifier(1000, min_samples_leaf=40, max_features=0.5, n_jobs=8, random_state=seed)
    if name == "et20_mf1": return ExtraTreesClassifier(1000, min_samples_leaf=20, max_features=1.0, n_jobs=8, random_state=seed)
    if name == "rf20": return RandomForestClassifier(1000, min_samples_leaf=20, max_features=0.5, n_jobs=8, random_state=seed)
    if name == "cat_reg": return CatBoostClassifier(iterations=800, learning_rate=0.02, depth=4, l2_leaf_reg=20, rsm=0.7,
                                                random_seed=seed, verbose=0, thread_count=8)
    if name == "et30_mf1": return ExtraTreesClassifier(1000, min_samples_leaf=30, max_features=1.0, n_jobs=8, random_state=seed)
    if name == "et10_mf1": return ExtraTreesClassifier(1000, min_samples_leaf=10, max_features=1.0, n_jobs=8, random_state=seed)
    if name == "rf": return RandomForestClassifier(500, min_samples_leaf=5, max_features=0.5, n_jobs=8, random_state=seed)
    if name == "lgbm": return LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                             random_state=seed, n_jobs=8, verbose=-1)
    if name == "cat": return CatBoostClassifier(iterations=600, learning_rate=0.03, depth=5, l2_leaf_reg=5,
                                                random_seed=seed, verbose=0, thread_count=8)
    raise KeyError(name)
