# Evaluación exacta de s02/s03/s04 y bloque D/E

Script: `src/frozen_de.py` (≈15 min en 8 CPU). Salidas en `reports/frozen_de/`. Sin submissions nuevas, umbral fijo 0,5, probabilidades sin modificar.

## 1. Candidatos congelados: configuración exacta

Configuración tomada de `reports/submissions.jsonl`, la misma que usó `src/predict.py`:

| Candidato | Modelos (media simple) | Bloques | Semillas |
|---|---|---|---|
| s02 | et30_mf1, lgbm_slow | demo, time, orig, clim | 42, 43, 44 |
| s03 | et30_mf1, lgbm_slow, lr_spline_age | demo, time, orig, clim | 42, 43, 44 |
| s04 | et30_mf1, lgbm_slow, lr_spline_age | demo, time | 42, 43, 44 |

Cada modelo se promedia sobre las 3 semillas y después se promedian los modelos con pesos iguales, igual que en `predict.py`.

**Verificación de exactitud** (`frozen_repro.txt`): se reentrenó cada candidato en todo train y se comparó con el CSV congelado. La diferencia máxima en `TargetRAUC` es 3,3e-16 (suma en paralelo de ExtraTrees) y `TargetF1` es idéntico en las 1030 filas en los tres. **El OOF de abajo no tiene ninguna aproximación de configuración.** La única diferencia con un modelo final es la inherente a la validación cruzada: cada fold entrena con menos filas.

Las aproximaciones anteriores `s0X~` (`reports/diag/paired_vs_s04.csv`) usaban solo la semilla 42. Coinciden con `s0X@seed42`. Diferencia frente al exacto en loc2: s02 −0,0006, s03 +0,0001, s04 −0,0002.

### Global OOF (3 semillas, splits fijos)

| | loc2 F1 | loc2 AUC | loc2 score | dist2 F1 | dist2 AUC | dist2 score |
|---|---|---|---|---|---|---|
| s02 | 0,8199 | 0,8221 | **0,8208** | 0,8069 | 0,8155 | 0,8104 |
| s03 | 0,8198 | 0,8198 | 0,8198 | 0,8085 | 0,8129 | 0,8103 |
| s04 | 0,8165 | 0,8200 | 0,8179 | 0,8156 | 0,8171 | **0,8162** |

### Diferencias emparejadas frente a s04 (mismos folds)

| | esquema | Δ global | Δ media fold | min | max | folds mejores | peor fold |
|---|---|---|---|---|---|---|---|
| s02 | loc2 | +0,0029 | +0,0030 | −0,0001 | +0,0070 | 4/5 | 0,8114 (s04: 0,8054) |
| s02 | dist2 | −0,0058 | −0,0010 | −0,0080 | +0,0045 | 3/6 | 0,7868 (s04: 0,7881) |
| s03 | loc2 | +0,0019 | +0,0018 | −0,0023 | +0,0056 | 4/5 | 0,8088 |
| s03 | dist2 | −0,0059 | −0,0049 | −0,0372 | +0,0093 | 2/6 | 0,7881 |

El Δ global de dist2 en s02/s03 (−0,006) es mayor que la media por distrito, porque Iganga (2339 filas) domina el OOF global.

### Por distrito (dist2: cada fold es un distrito retenido completo)

| distrito (n) | s02 | s03 | s04 | s04:et30 | s04:lgbm | s04:lr |
|---|---|---|---|---|---|---|
| Bugweri (199) | 0,8024 | 0,8088 | 0,8089 | 0,8186 | 0,8094 | 0,7900 |
| Entebbe City (54) | 0,7868 | 0,7881 | 0,7881 | 0,7702 | 0,8015 | 0,7815 |
| Iganga (2339) | 0,8081 | 0,8101 | 0,8161 | 0,8138 | 0,8090 | 0,8141 |
| Kikuube (128) | 0,8206 | 0,8267 | 0,8174 | 0,8141 | 0,7866 | 0,8462 |
| Mayuge (268) | 0,8239 | 0,8262 | 0,8217 | 0,8226 | 0,8273 | 0,8270 |
| Mbale City (158) | 0,8375 | **0,7958** | 0,8330 | 0,8279 | 0,8309 | 0,8226 |

- s03 cae en Mbale City (−0,037 frente a s04). La causa es su componente LR con clima absoluto, que da 0,344 en ese distrito.
- s02 no tiene ese componente y no colapsa en Mbale (0,8375), pero empeora en Iganga (−0,008) y en Bugweri (−0,007).

### Variación de semilla del estimador (splits fijos, ensemble con una sola semilla)

| | loc2 (seed 42/43/44) | dist2 (seed 42/43/44) | etiquetas cambiadas vs s04 de 3 semillas (loc2) |
|---|---|---|---|
| s02 | 0,8202 / 0,8204 / 0,8202 | 0,8101 / 0,8102 / 0,8110 | 141 / 136 / 132 |
| s03 | 0,8199 / 0,8205 / 0,8198 | 0,8106 / 0,8108 / 0,8105 | 78 / 76 / 75 |
| s04 | 0,8177 / 0,8183 / 0,8182 | 0,8160 / 0,8161 / 0,8158 | 5 / 2 / 6 |

La semilla mueve el score como mucho ±0,0006, menos que las diferencias entre candidatos. Las semillas no son validaciones independientes: comparten los mismos splits.

### Variación de partición (loc2 con StratifiedGroupKFold, random_state 43/44/45; 3 semillas)

| | fijo | r43 | r44 | r45 | Δ vs s04 (fijo, r43, r44, r45) |
|---|---|---|---|---|---|
| s02 | 0,8208 | 0,8191 | 0,8178 | 0,8183 | +0,0029, −0,0009, −0,0017, +0,0003 |
| s03 | 0,8198 | 0,8171 | 0,8194 | 0,8182 | +0,0019, −0,0029, −0,0001, +0,0002 |
| s04 | 0,8179 | 0,8200 | 0,8195 | 0,8181 | – |

La ventaja de s02/s03 en loc2 aparece solo en la partición fija. En las tres alternativas es nula o negativa. Las particiones se solapan (mismas localidades), así que tampoco son validaciones independientes.

### Edad, periodo y probabilidades (loc2)

- Edad: F1 casi idéntico en <1 y 1-4 (constante a 0,5). Las diferencias aparecen en 30-49 (s02 0,621, s03 0,629, s04 0,584) y 50-64 (0,390 / 0,478 / 0,453). El AUC dentro de cada grupo varía ≤0,016 entre candidatos.
- Periodo: 18-19 y 20+ siguen siendo los peores (F1 0,64–0,74, AUC 0,61–0,63) en los tres. Ningún candidato lo corrige.
- Probabilidades: s02 tiene más masa entre 0,4 y 0,6 (26,5 %) que s03 y s04 (22 %). Mediana 0,63–0,65 en los tres. Detalle en `frozen_prob_dist.csv` y `frozen_age_period.csv`.
- Etiquetas OOF que cambian frente a s04 en loc2: s02 135, s03 76.

**Lectura.** Ningún congelado domina a los demás:
- s02 y s03 ganan en la partición fija de loc2 (4/5 folds), pero esa ventaja no se repite en las particiones alternativas.
- s04 es mejor en dist2 global, en Iganga y frente al colapso de s03 en Mbale City.
- La recomendación de finalistas s03 + s04 se mantiene como cobertura de dos comportamientos distintos. Si se prioriza robustez geográfica, s04 + s02 sería la alternativa sin el componente LR con clima.

## 2. Bloque D y E (cribado con semilla 42; semillas 43/44 aparte)

Referencia: `et30_mf1` sobre demo+time (ExtraTrees 1000 árboles, `min_samples_leaf=30`, `max_features=1.0`, sin pesos de clase). Es exactamente el componente ET de s04 con una semilla.

- **D:** `et10_mf1`, idéntico salvo `min_samples_leaf=10`, mismas features.
- **E:** `et30_mf1` sobre demo+time+`clim_red`, que añade tres anomalías relativas a 90 días: `tavg_30d − tavg_90d`, `rain_sum_30d/30 − rain_sum_90d/90` y `ndvi_30d − ndvi_90d`. Son diferencias por fila, sin parámetros ajustados, así que no hay transformación que ajustar dentro del fold.

| | loc2 score (F1 / AUC) | Δ loc2, folds mejores | dist2 score | Δ dist2, folds mejores | etiq. cambiadas loc2 |
|---|---|---|---|---|---|
| ref | 0,8183 (0,8151 / 0,8232) | – | 0,8148 | – | – |
| D leaf 10 | 0,8098 (0,8026 / 0,8206) | −0,0085, 0/5 | 0,8068 | −0,0080, 0/6 | 189 |
| E clima reducido | 0,8181 (0,8154 / 0,8221) | −0,0003, 3/5 | 0,8136 | −0,0012, 1/6 | 20 |

**D** empeora en todos los folds y distritos en las tres semillas (loc2 Δ −0,0055 a −0,0086) y en las tres particiones alternativas (−0,0063 a −0,0085).
- Peores caídas: Kikuube −0,054, Mbale City −0,016, 18-19 F1 0,621 → 0,553.
- Las probabilidades son más extremas (q95 0,987 frente a 0,970).
- Más capacidad en ET no ayuda. La hipótesis de exceso de regularización queda contradicha para ET en este rango.

**E** apenas cambia nada (20 etiquetas).
- Δ loc2 entre −0,0008 y −0,0003 según la semilla y entre −0,0007 y +0,0005 según la partición. Δ dist2 entre −0,0012 y +0,0002.
- No hay colapso geográfico: el peor distrito cambia como mucho ±0,008.
- La variación de semilla de la propia referencia (±0,0007 en loc2 y hasta −0,0018 en dist2) es del mismo orden.
- Con ET y estas tres anomalías no se detecta mejora. Esto no demuestra que el clima no tenga señal: solo que esta representación no añade nada al ET sobre edad y fecha.

**Conclusión:** ni D ni E justifican un candidato nuevo. Se conservan s02, s03 y s04 congelados.
