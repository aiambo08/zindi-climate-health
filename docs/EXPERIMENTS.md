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
