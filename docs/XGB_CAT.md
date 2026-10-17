# Bloque manual XGBoost / CatBoost regularizado

Script: `src/xgb_cat.py`. Salidas en `reports/xgb_cat/` (`screen_*.csv`). Cribado con la semilla 42 y los splits fijos `loc2` (principal) y `dist2` (robustez).
- Iteraciones fijas, sin early stopping ni búsqueda.
- Sin pesos de clase.
- Umbral 0,5.
- Sin submissions nuevas.

## Comprobación previa
- **XGBoost no se había probado.** No aparece ni en el código, ni en los informes, ni en `requirements.txt`. Se añade `xgboost==2.1.4`.
- **CatBoost existentes:**
  - `cat`: iterations 600, lr 0.03, depth 5, l2_leaf_reg 5.
  - `cat_reg`: iterations 800, lr 0.02, depth 4, l2_leaf_reg 20, rsm 0.7.
  - Ninguno usa pesos de clase ni early stopping. `cat_d3_l10` (depth 3, L2 10) no equivale a ninguno de los dos.

## Configuraciones
El mensaje fijaba solo profundidad y L2. El resto se fijó antes de ejecutar, copiando los modelos existentes:

| nombre | parámetros |
|---|---|
| `xgb_d2` / `xgb_d4` | n_estimators 600, learning_rate 0.01, max_depth 2 / 4, min_child_weight 10, subsample 0.7, colsample_bytree 0.6, reg_lambda 5, tree_method hist. Mismo lr, iteraciones, submuestreo y L2 que `lgbm_slow` |
| `cat_d3_l10` | `cat_reg` con depth 3 y l2_leaf_reg 10 (iterations 800, lr 0.02, rsm 0.7) |

Bloques: `dt` (edad, sexo, zona, fecha) y `full` (más temperaturas originales y clima), los mismos de s04 y s02.

## Resultados (semilla 42)

| cfg | train score | loc2 F1 / AUC / score | Δ vs ET (folds mejores) | Δ vs s04 (folds) | dist2 score | Δ vs ET (folds) | Δ vs s04 (folds) |
|---|---|---|---|---|---|---|---|
| xgb_d2 dt | 0,826 | 0,810 / 0,814 / 0,8115 | −0,0069 (0/5) | −0,0065 (0/5) | 0,8130 | −0,0018 (3/6) | −0,0032 (2/6) |
| xgb_d2 full | 0,833 | 0,813 / 0,816 / 0,8141 | −0,0043 (1/5) | −0,0039 (1/5) | 0,8080 | −0,0068 (2/6) | −0,0082 (2/6) |
| xgb_d4 dt | 0,851 | 0,813 / 0,817 / 0,8147 | −0,0036 (2/5) | −0,0032 (2/5) | 0,8110 | −0,0039 (3/6) | −0,0053 (3/6) |
| xgb_d4 full | 0,876 | 0,813 / 0,816 / 0,8140 | −0,0043 (1/5) | −0,0039 (1/5) | 0,8063 | −0,0086 (2/6) | −0,0100 (1/6) |
| cat_d3_l10 dt | 0,844 | 0,810 / 0,815 / 0,8119 | −0,0064 (1/5) | −0,0060 (1/5) | 0,8044 | −0,0105 (2/6) | −0,0119 (2/6) |
| cat_d3_l10 full | 0,864 | 0,809 / 0,814 / 0,8110 | −0,0074 (1/5) | −0,0069 (1/5) | 0,8021 | −0,0127 (2/6) | −0,0141 (2/6) |
| **ref ET30 dt** (mejor individual) | 0,838 | 0,815 / 0,823 / 0,8183 | – | +0,0004 (2/5) | 0,8148 | – | −0,0014 (3/6) |
| lgbm_slow full | 0,859 | 0,819 / 0,819 / 0,8189 | +0,0005 (3/5) | +0,0010 (2/5) | 0,8073 | −0,0075 (3/6) | −0,0089 (3/6) |
| cat_reg dt / full | 0,853 / 0,879 | 0,8125 / 0,8125 | −0,006 | −0,005 | 0,8057 / 0,8013 | −0,009 / −0,014 | −0,011 / −0,015 |
| s04 (exacto, 3 semillas) | – | 0,8179 | – | – | 0,8162 | – | – |

- "Folds mejores" cuenta los folds emparejados con mejor score. Las Δ son globales. La media, mínimo y máximo por fold están en `screen_paired_summary.csv`.
- ET30 dt es el mejor individual: está a 0,0006 de `lgbm_slow full` en loc2 y le saca 0,0075 en dist2.

### Por distrito (dist2, cada fold es un distrito retenido)

| distrito | xgb_d2 dt | xgb_d2 full | xgb_d4 dt | xgb_d4 full | cat_d3 dt | cat_d3 full | ET30 dt | s04 |
|---|---|---|---|---|---|---|---|---|
| Bugweri | 0,800 | 0,795 | 0,807 | 0,800 | 0,799 | 0,797 | 0,818 | 0,809 |
| Entebbe City | 0,787 | 0,791 | 0,799 | 0,788 | 0,790 | 0,792 | 0,770 | 0,788 |
| Iganga | 0,812 | 0,806 | 0,810 | 0,803 | 0,802 | 0,799 | 0,814 | 0,816 |
| Kikuube | 0,812 | 0,834 | 0,772 | 0,817 | 0,791 | 0,815 | 0,818 | 0,817 |
| Mayuge | 0,826 | 0,821 | 0,823 | 0,821 | 0,831 | 0,830 | 0,823 | 0,822 |
| Mbale City | 0,841 | 0,826 | 0,841 | 0,845 | 0,828 | 0,817 | 0,830 | 0,833 |

- Ninguna configuración mejora en Iganga (2339 filas), que domina el global.
- Hay mejoras aisladas en Mbale City (+0,008 a +0,012 para XGB) y Entebbe (n = 54). No son sistemáticas.
- xgb_d4 dt cae 0,045 en Kikuube.

### Complementariedad de errores frente a s04 (loc2, descriptiva, sin combinar)

| cfg | corr | etiquetas cambiadas | errores solo en cfg | errores solo en s04 |
|---|---|---|---|---|
| xgb_d2 dt | 0,992 | 102 | 63 | 39 |
| xgb_d2 full | 0,985 | 128 | 71 | 57 |
| xgb_d4 dt | 0,985 | 166 | 92 | 74 |
| xgb_d4 full | 0,971 | 216 | 115 | 101 |
| cat_d3_l10 dt | 0,985 | 165 | 93 | 72 |
| cat_d3_l10 full | 0,964 | 232 | 131 | 101 |

Las seis configuraciones añaden más errores propios de los que corrigen de s04, en loc2 y en dist2 (`screen_errors.csv`). Su correlación con s04 es igual o mayor que la de los componentes ya incluidos (ET 0,987, LGBM full 0,980). No hay complementariedad concreta.

## Decisión
Ninguna configuración muestra una mejora prometedora ni complementariedad concreta, así que, siguiendo el punto 5 del protocolo, no se confirma con 42/43/44 ni con particiones alternativas. El modo de confirmación (`python xgb_cat.py CFG`) queda implementado pero no se ha ejecutado.
- Más capacidad sube el score de train (de 0,826 a 0,876) pero no el OOF, igual que con RF y CatBoost en el diagnóstico.
- Cerramos con los candidatos congelados s02, s03 y s04.
