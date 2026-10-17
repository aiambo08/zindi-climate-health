#!/usr/bin/env bash
# Regenera particiones y los dos CSV candidatos desde data/raw/. Uso: bash reproduce.sh
set -euo pipefail
cd "$(dirname "$0")/src"
python splits.py
python predict.py s02_et30mf1_lgbmslow et30_mf1,lgbm_slow demo,time,orig,clim 3
python predict.py s03_et30mf1_lgbmslow_lrspline et30_mf1,lgbm_slow,lr_spline_age demo,time,orig,clim 3
sha256sum ../submissions/s02_et30mf1_lgbmslow.csv ../submissions/s03_et30mf1_lgbmslow_lrspline.csv
