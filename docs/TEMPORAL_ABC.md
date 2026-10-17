# Bloque temporal A/B/C

Reproducción: `cd src && python temporal.py && python paired.py`. Salidas en `reports/temporal/`.

## Protocolo
- **Validación:** principal `loc2`, robustez `dist2` (un distrito retenido por fold). Los splits son los de `data/splits.csv` y se mantienen fijos.
- **Comparación:** cada configuración se compara contra A en los mismos folds.
- **Features:** `dt` = age, male, rural, year (año + día/366), doy_sin, doy_cos.
- **Fijo en todas:** umbral 0,5, `class_weight=None` y ninguna calibración ni desplazamiento.
- **Ajuste dentro del fold:** todas las transformaciones (nodos de spline, medias y desviaciones del escalado) se ajustan dentro de cada fold mediante `Pipeline`.
- **Modelo:** `LogisticRegression(C=1.0)` con penalización L2 y solver lbfgs, que es determinista. Por eso no hay variación de semilla del estimador.
- **Variación de partición:** se mide aparte, repitiendo `StratifiedGroupKFold` por `loc2` con `random_state` de 43 a 47. Esos 5 repartos no son validaciones independientes: reutilizan los mismos datos.
- **Grupos predefinidos (sin buscar cortes):**
  - Edad: <1, 1–4, 5–14, 15–29, 30–49, 50–64, 65+.
  - Periodo: ≤2009, 10–11, 12–13, 14–15, 16–17, 18–19, 20+.
  - Son los mismos cortes de `diag2.py`, fijados antes de este bloque.

## Configuraciones

| cfg | término de edad | término de año | interacción | otras features | nº columnas |
|---|---|---|---|---|---|
| A (= `lr_spline_age` dt) | spline, 6 nodos, grado 3 → 8 bases, sin escalar | lineal estandarizado | – | 4, estandarizadas | 13 |
| B | igual que A | spline, 4 nodos → 6 bases | – | igual | 18 |
| C | igual que A | igual que B | 8 bases de edad × año estandarizado | igual | 26 |

Comprobación: A reproduce el OOF de `lr_spline_age|dt|loc2` de `diag.py` con un error máximo de 1e-16.

## Tabla OOF previa por edad y periodo (sin reentrenar)
Fichero `reports/temporal/age_period_existing.csv`, con `lr_spline_age`, `et30` y s04~ en `loc2`. Incluye edad, periodo y el cruce edad × periodo.

- **Etiquetas frente a ordenamiento:**
  - <1 y 1–4 años: el 100 % de las predicciones queda por encima de 0,5, así que las etiquetas no varían. Aun así, el AUC dentro del grupo es 0,78 en <1 y 0,56–0,58 en 1–4.
  - 65+: casi todas las predicciones quedan por debajo de 0,5 (2–4 %), con un AUC de 0,62.
- **Calibración en media por periodo:**
  - 18–19: probabilidad media 0,55–0,59 frente a una prevalencia de 0,49.
  - 20+: probabilidad media 0,55 frente a una prevalencia de 0,55.
  - En ambos periodos el AUC dentro del periodo es de solo 0,54–0,63.

## Resultados (una semilla)

| cfg | loc2 F1 | loc2 AUC | loc2 score | Δ vs A | folds mejor (loc2) | dist2 score | Δ vs A | folds mejor (dist2) | etiquetas cambiadas vs A (loc2 / dist2) |
|---|---|---|---|---|---|---|---|---|---|
| A | 0,8154 | 0,8095 | 0,8130 | – | – | 0,8146 | – | – | – |
| B | 0,8159 | 0,8097 | 0,8134 | +0,0004 | 3/5 | 0,8135 | −0,0011 | 0/6 | 31 / 33 |
| C | 0,8121 | 0,8108 | 0,8116 | −0,0014 | 2/5 | 0,8132 | −0,0014 | 2/6 | 93 / 126 |

**Δ emparejado por fold** (`abc_paired_folds.csv`):

| cfg | loc2, rango por fold | dist2, por distrito |
|---|---|---|
| B | −0,002 a +0,004 | Bugweri −0,006, Kikuube −0,005, Mayuge −0,004; resto ≈ 0 |
| C | −0,007 a +0,005 | Bugweri +0,010, Mbale +0,003, Entebbe −0,013 (n = 54), Kikuube −0,015, Mayuge −0,008, Iganga −0,001 |

**Variación de partición** (`abc_partition_repeats.csv`, 5 repartos `loc2` alternativos):
- B − A: de −0,0003 a +0,0013, positivo en 4 de 5.
- C − A: de −0,0020 a +0,0008, negativo en 4 de 5.

**Distribución de probabilidades** (`abc_prob_dist.csv`):
- C amplía los extremos (p95 0,963 frente a 0,940).
- C sube la fracción de probabilidades entre 0,4 y 0,6 (0,25 frente a 0,14).
- Con la misma ordenación global, C produce más casos cerca del umbral.

**Por edad y periodo** (`abc_age_period.csv`, loc2):
- B y C apenas mueven el AUC dentro del periodo: en 18–19 pasa de 0,539 (A) a 0,543 (B) y 0,564 (C); en 20+ de 0,612 a 0,614 y 0,628.
- C empeora el F1 en 50–64 años (0,357 frente a 0,503) y en 65+, porque desplaza esos grupos por debajo de 0,5.

## Lectura
- **B:** el spline de año no aporta una mejora consistente. Es +0,0004 en `loc2` y −0,0011 en `dist2`, y no mejora en ningún distrito retenido.
- **C:** la interacción edad × año mejora un poco el AUC dentro de 2018+, pero empeora el score global en los dos esquemas y en 4 de 5 repartos alternativos. Además, empeora de forma sistemática en tres distritos pequeños.
- **Hipótesis:** con esta parametrización no se confirma que el déficit en 2018+ venga de una interacción edad × año que una LR pueda captar.
- **Candidatos:** ninguno de los dos justifica un candidato. s02, s03 y s04 siguen congelados.
- **Pendiente:** D/E y la evaluación exacta (3 semillas) de los congelados en `loc2`/`dist2`, a la espera de decisión.
