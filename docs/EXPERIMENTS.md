# Registro de experimentos

Todas las cifras son scores OOF globales (no medias por fold), con umbral 0.5. `loc` es la validación agrupada por localidad y es el esquema principal. El detalle completo está en `reports/experiments.jsonl`.

## Hallazgos de la auditoría

- El contrato de datos se cumple: 3146×13, 1030×12 y 4176×18 filas×columnas; 2047 positivos y 1099 negativos; no hay nulos; el join por ID es 1:1 y las dos fechas coinciden al 100 %.
- El test es espacial: 1004 de sus 1030 filas son de 10 localidades nuevas y ninguna coordenada del test está en train. Hay 49 nombres de localidad y 55 pares de coordenadas; 7 nombres tienen 2 pares de coordenadas.
- `avg/max/min_temperature` y `precipitation` son constantes dentro de cada fecha e independientes de la ubicación.
- `hot_days_30d` es constante (0).
- Solo hay 2 duplicados exactos, y su etiqueta es coherente.
- Las etiquetas no se agrupan en el tiempo dentro de una localidad: la correlación entre la etiqueta y la de sus vecinos es ≤ 0.06.
- La edad es la señal dominante: <5 años ≈ 0.79–0.92 de positivos y ≥65 ≈ 0.27–0.32. La tasa de positivos baja con los años (0.77 en 2009, 0.5 en 2019–2021).

## Hipótesis y resultados

| Hipótesis | Experimento | loc S | Conclusión |
|---|---|---|---|
| Control | dummy | 0.668 | — |
| Un modelo lineal basta | lr / lr_spline_age | 0.792 / 0.790 | Peor que los árboles, sobre todo en `loc` |
| Árboles por defecto | et / rf / lgbm / cat | 0.808 / 0.802 / 0.806 / 0.804 | Todos dan lo mismo |
| ¿Aporta el clima? | cat sin clima / solo demo+tiempo | 0.803 / 0.802 | El clima no aporta |
| ¿Aporta la geografía? | cat + lat/lon | 0.803 | No (test espacial) |
| Más regularización | et10 → et20 → et40 (min_samples_leaf) | 0.810 → 0.815 → 0.815 | Mejora; F1 sube con más suavizado |
| Features de edad explícitas | et/lr/lgbm + agex | ≤ base | No (los árboles ya captan la edad) |
| Contrastes de clima | et + clim_fe | 0.806 | No |
| max_features=1.0 | et20_mf1 / et30_mf1 | 0.812 / 0.815 | Mejor AUC (0.822) |
| LightGBM lento y regularizado | lgbm_slow | 0.811 | Mejora frente a lgbm (0.806) |
| Promedio simple de modelos complementarios | et30_mf1 + lgbm_slow (3 semillas) | 0.815 | Candidato s02 |
| + modelo lineal | + lr_spline_age | 0.814 | Candidato s03 (alternativa más suave) |

Las diferencias entre las mejores configuraciones (±0.003) están por debajo de la dispersión entre folds (≈0.01), así que no se siguió ajustando.

## Lo que no se probó y por qué

- Target encoding de la localidad: el 97,5 % del test son localidades nuevas.
- Stacking y pesos optimizados: el riesgo de sobreajustar las OOF es mayor que la ganancia esperable.
- Datos climáticos externos: las variables climáticas proporcionadas no aportan señal en ningún esquema, así que es improbable que otras más finas cambien el resultado.

## Fase 2 (2026-10-17)

**Leaderboard público** (`reports/public_lb.jsonl`): s02 = 0.8343 (AUC 0.8370, F1 0.8325); s03 = 0.8368 (AUC 0.8408, F1 0.8341). Ambos por encima del OOF por localidad (≈0.815): la validación es algo pesimista pero no está rota, y el orden s03 > s02 coincide con la hipótesis de que lo más suave generaliza mejor.

**Nuevo esquema `dist`** (StratifiedGroupKFold por distrito, 2.º campo de `location`): el test incluye 4 distritos no vistos en train (Lwemiyaga, Kagoonya, Kamuli, Mitooma). Las columnas antiguas de `data/splits.csv` no cambian.

**Resultados (`src/exp5.py`)**, score loc / dist:
| modelo | demo+time+orig+clim | solo demo+time |
|---|---|---|
| et30_mf1 | 0.8153 / 0.8158 | 0.8163 / 0.8182 |
| lgbm_slow | 0.8110 / 0.8122 | 0.8161 / 0.8120 |
| lr_spline_age | 0.7900 / 0.7959 | 0.8134 / 0.8138 |
| lr_spl_ay (splines edad y año) | 0.7916 / 0.8027 | 0.8105 / 0.8141 |

Las features de clima/temperatura **perjudican** fuera de localidad, sobre todo a los modelos lineales (inestabilidad: ±0.04–0.07 entre folds en dist). Edad sola: 0.797.

**Descartado:** conteos de defunciones por localidad en ventanas de ±3/7/30/90 días y fracción de menores de 5 años en la ventana (sin etiquetas, calculados con train+test): correlación con y < 0.07 y sin mejora (loc 0.8163 → 0.8155).

**Candidato s04** = media(et30_mf1, lgbm_slow, lr_spline_age), solo demo+time y 3 semillas: loc 0.8179, estratificado 0.8167, fecha 0.8190 (s03: 0.8140 / 0.8170 / 0.8167). Correlación con s03 de 0.994, con 24 de 1030 predicciones F1 cambiadas.
