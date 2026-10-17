# Diagnóstico (bloque post-s04)

> Revisión: la validación principal es `loc2` y la de robustez es `dist2`; `loc` se conserva solo como histórico. Los resultados con sufijo `~` son de una semilla, salvo que se indique otra cosa. Las secciones marcadas *[Corregido]* se han reescrito tras la revisión.

s02, s03 y s04 están congelados en `submissions/frozen/` (solo lectura, `SHA256SUMS`). En este bloque no se ha generado ningún candidato nuevo ni se ha hecho ninguna submission.

Reproducción: `cd src && python splits.py && python diag.py && python diag2.py`. Salidas en `reports/diag/`:
`individual.csv`, `folds.csv`, `params.json`, `oof_diag.csv`, `ensembles.csv`, `robustness.csv` y `errors.csv`.

## 1. Modelos individuales (semilla 42)
- Bloques de features:
  - `dt` = age, male, rural, year (continuo), doy_sin, doy_cos.
  - `full` = `dt` + temperaturas originales + clima descargado.
- Ponderación de clases: **ninguna**. En todos los modelos `class_weight=None` (comprobado con `get_params()` en `params.json`).
- Iteraciones: **fijas y elegidas a mano antes de evaluar**. No hay early stopping ni búsqueda de hiperparámetros.
  - ExtraTrees y RF: 1000 árboles.
  - LightGBM: 600 iteraciones, lr 0,01.
  - CatBoost: 800 iteraciones, lr 0,02.
  - LR: lbfgs con max_iter 2000.
- Umbral fijo de 0,5.
- Métrica: score = 0,6·F1 + 0,4·AUC.

**OOF score por esquema.** `loc` = nombre original; `loc2` = nombre normalizado (Brac fusionado); `dist2` = distrito retenido.

| modelo | feats | train | loc | loc2 | dist2 | dist2 fold min |
|---|---|---|---|---|---|---|
| dummy | – | 0,673 | 0,668 | 0,667 | 0,671 | 0,668 |
| lr | full | 0,814 | 0,792 | 0,810 | **0,786** | **0,354** |
| lr | dt | 0,813 | 0,813 | 0,812 | 0,812 | 0,788 |
| lr_spline_age | full | 0,818 | 0,790 | 0,812 | **0,788** | **0,344** |
| lr_spline_age | dt | 0,816 | 0,813 | 0,813 | 0,815 | 0,782 |
| et30_mf1 | full | 0,852 | 0,815 | 0,818 | 0,812 | 0,772 |
| et30_mf1 | dt | 0,838 | 0,816 | 0,818 | 0,815 | 0,770 |
| rf20 | full | 0,890 | 0,808 | 0,811 | 0,806 | 0,775 |
| rf20 | dt | 0,865 | 0,812 | 0,816 | 0,810 | 0,769 |
| lgbm_slow | full | 0,859 | 0,811 | 0,819 | 0,807 | 0,797 |
| lgbm_slow | dt | 0,843 | 0,816 | 0,815 | 0,811 | 0,788 |
| cat_reg | full | 0,879 | 0,806 | 0,813 | 0,801 | 0,784 |
| cat_reg | dt | 0,853 | 0,812 | 0,813 | 0,806 | 0,790 |
| svm_rbf_c03 | dt | 0,816 | 0,806 | 0,806 | 0,778 | 0,769 |

Lectura:
- **No hay indicios de exceso de regularización.** Los modelos con más capacidad (RF y CatBoost) tienen un train más alto, pero su OOF es igual o peor. El hueco train–OOF crece con la capacidad: es de 0,02 en LR, 0,02–0,04 en ET y LightGBM y de 0,05–0,08 en RF y CatBoost.
- **Todos los modelos reales quedan dentro de ±0,005 en score global.** *[Corregido]* Esto no prueba que sean equivalentes. Las diferencias emparejadas por fold sobre los mismos splits están en `reports/diag/paired_vs_s04.csv` (ver la sección 6).
- **Riesgo de extrapolación del clima en LR.** Al retener Mbale City (fold 5 de `dist2`), LR con clima cae a un score de 0,35: predice positivos en solo el 46 % de los casos, frente a un 64 % real. Con `dt` el mismo fold da 0,81–0,82. Los árboles no sufren esa caída porque no extrapolan.
  - Es relevante porque el 36 % de las filas de test (371) son de 4 distritos que no aparecen en train: Kalangala, Kamuli, Mitooma y Sembabule.
  - Esto **no** demuestra que el clima no tenga señal. Muestra que el clima en niveles absolutos, metido en un modelo lineal, extrapola mal.

## 2. Auditoría de agrupación
Fichero: `reports/location_audit.csv`, que cruza nombre normalizado, coordenadas y número de filas en train/test.

- **Brac Uganda Iganga.** Las variantes con y sin ", Eastern Region" tienen coordenadas idénticas, (0.608396, 33.482668), y suman 211 + 207 filas. Son el mismo lugar. Están fusionadas solo en `loc2`; el esquema `loc` original se conserva sin cambios.
- **Mismo nombre con coordenadas distintas (97 m–1,4 km).** No se fusionan porque no hay justificación para hacerlo:
  - En train: Nakavule B, Nawanzu, Wairama, Walanga y Wante. Ambas coordenadas están en train y comparten nombre, así que ya están en el mismo grupo tanto en `loc` como en `loc2`.
  - Train/test: Nakigo I. Un punto está en train y otro, a 739 m, en test (26 filas).
  - Solo test: Bubago.
- **Distrito.** `district()` toma el último componente tras quitar "Uganda" y las regiones o subregiones. El resultado se ha comprobado a mano para las 51 localidades. El primer esquema `dist` usaba `split(", ")[1]`, que es incorrecto para direcciones largas; se conserva solo como registro histórico.
- **Finalistas reevaluados con `loc2`** (aproximación con semilla única de los mismos componentes):

| | loc | loc2 | dist2 |
|---|---|---|---|
| s02~ | 0,815* | 0,820 | 0,810 |
| s03~ | 0,814* | 0,820 | 0,811 |
| s04~ | 0,818* | 0,818 | **0,816** |

\* Valores con 3 semillas, de fases anteriores.

  - Corregir el grupo de Brac sube el OOF de todos en torno a +0,003. No cambia el orden de forma significativa.
  - Con clima, el score de un modelo **lineal** varía hasta 0,02 solo por cómo caen los folds (LR full: 0,792 en `loc` frente a 0,810 en `loc2`). Ese bloque es inestable.

## 3. Robustez (`robustness.csv`)
"Distrito conocido" significa que el fold de entrenamiento contiene al menos otra localidad del mismo distrito. Se calcula por fila usando los folds de `loc2`.

| slice (loc2) | n | s03~ | s04~ | et30 dt | lr_spline dt |
|---|---|---|---|---|---|
| localidad nueva, distrito conocido | 2860 | 0,818 | 0,817 | 0,819 | 0,812 |
| distrito sin otras localidades en train | 286 | 0,837 | 0,825 | 0,818 | 0,828 |
| localidad <60 filas | 426 | 0,807 | 0,803 | 0,805 | 0,791 |
| localidad 60–119 | 1595 | 0,826 | 0,826 | 0,826 | 0,821 |
| localidad ≥120 | 1125 | 0,816 | 0,812 | 0,813 | 0,810 |

Distrito completamente retenido (`dist2`):
- s04~ va de 0,788 (Entebbe, n = 54) a 0,833 (Mbale).
- s03~ cae a **0,795 en Mbale**, porque su componente lineal con clima predice demasiados negativos (46 % de positivos frente a un 64 % real).

Limitaciones:
- Solo hay 6 distritos en train, y uno de ellos (Iganga) tiene el 74 % de las filas.
- El slice "distrito sin otras localidades" son en realidad 2 localidades (Kikuube y Mbale, 286 filas). Sus cifras tienen un error de ±0,03 o más.

## 5. Errores OOF de s03~ (`errors.csv`, solo descriptivo)
- **Edad.**
  - Menores de 1 año y de 1–4 años: se predicen siempre como positivos (pred_pos = 1,0). Dentro de esos grupos el AUC es 0,57–0,80 y hay 236 + 48 FP.
  - 65+: se predicen casi siempre como negativos (3 %), lo que da 159 FN.
  - 15–64: F1 de 0,49–0,65. La mayor parte de la confusión está aquí.
  - *[Corregido]* Una predicción constante a 0,5 dentro de un grupo de edad no implica que el modelo no ordene dentro de ese grupo. El AUC dentro de grupo es 0,78 en <1 año, 0,56 en 1–4 y 0,62 en 65+. Sí hay ordenamiento en <1 año; es débil en 1–4 y en 5–64.
- **Año.**
  - 2018–19: el 72 % de las predicciones queda por encima de 0,5, frente a una prevalencia del 49 %; la probabilidad media es 0,59. En 2020+ el 78 % queda por encima de 0,5, frente a una prevalencia del 55 %, pero la probabilidad media (0,55) coincide con la prevalencia. *[Corregido]* En 2020+ no hay una sobreestimación del nivel. Lo que falla es el ordenamiento dentro del periodo (AUC 0,61).
  - El score en esos años es de 0,63–0,69, frente a 0,84–0,86 en los anteriores.
  - La prevalencia cae en los años recientes. El modelo lo sigue en media, pero ordena mal dentro de esos periodos. El bloque A/B/C (`docs/TEMPORAL_ABC.md`) contrasta esta hipótesis.
  - El test tiene la misma distribución de años que train.
- **Género y zona:** diferencias pequeñas (0,81–0,83).
- **Localidad:** el rango es 0,77–0,87, sin outliers claros más allá del tamaño.

Ninguno de estos patrones se ha convertido en una regla.

## 6. Ensembles (`ensembles.csv`)
Media simple, sin pesos, semilla única.

| | loc2 | dist2 |
|---|---|---|
| mejor individual (lgbm full / et30 dt) | 0,819 | 0,815 |
| et30 + lgbm (full) = s02~ | **0,820** | 0,810 |
| et30 + lr_spline (dt) | 0,818 | 0,816 |
| lgbm + lr_spline (dt) | 0,816 | **0,817** |
| s03~ (full) | 0,820 | 0,811 |
| s04~ (dt) | 0,818 | 0,816 |

Ablaciones:
- Quitar cualquier componente de s04~ cambia el score en ±0,003.
- En `dist2`, quitar la LR de s03~ no le hace daño.

**Conclusión (corregida).** Todas las cifras son de una semilla salvo que se indique otra cosa. Diferencias emparejadas frente a s04~ en `reports/diag/paired_vs_s04.csv`:

- **loc2:**
  - s02~ y s03~ mejoran en 4 de 5 folds, con una diferencia media de +0,002 a +0,003.
  - `et30` y `lgbm` en solitario mejoran en solo 2 de 5.
  - `lr_spline` en solitario empeora en los 5 folds (−0,005).
- **dist2:**
  - s03~ pierde 0,038 en el fold de Mbale.
  - `lgbm+lr_spline` mejora en 5 de 6 folds (+0,002).
- **Peor fold en dist2:**
  - s04~ = 0,788.
  - `et30` solo = 0,770 y `lr_spline` solo = 0,782; frente a estos dos, el ensemble sí protege.
  - `lgbm` full solo = 0,797, más alto que s04~; frente a él el ensemble no protege.
  - No se afirma que el ensemble proteja el peor fold en general.

## Respuesta a la pregunta del bloque
- **Grupos de validación.** Brac estaba partido. Corregirlo cambia ~0,003 y no modifica el ranking. **No es la limitación principal.**
- **Exceso de regularización.** No hay indicios: más capacidad sube el train sin subir el OOF. Falta probar una capacidad moderada con features de edad más ricas (C1–C2 abajo).
- **Representación de la edad e interacciones.** Es la hipótesis más plausible:
  - El ordenamiento dentro de grupo es débil en 1–4 y en 5–64 años.
  - En 2018+ el ordenamiento dentro del periodo es pobre. Hipótesis: interacción edad × año. *Contrastada en A/B/C: no se confirma (ver `docs/TEMPORAL_ABC.md`).*
  - Sin probar todavía.

## 4. Matriz propuesta (NO ejecutada; ≤ 8 configuraciones)
Protocolo común:
- Esquemas `loc`, `loc2` y `dist2`, con 3 semillas.
- Umbral 0,5, sin pesos de clase y sin calibración.
- Hiperparámetros fijados ahora, antes de ver resultados.
- `age_fe` = log_age, u1, u5, age65, age_x_year (ya definidas en `features()`).
- `clim_red` = dtemp_30_90, rain30_rate_vs90, dndvi: anomalías relativas a 90 días, menos propensas a extrapolar que los niveles absolutos.

| # | Modelo | Features | Hipótesis | Compara con |
|---|---|---|---|---|
| C1 | LGBM capacidad baja: 4 hojas, min_child 100, 400 iteraciones, lr 0,02 | dt + age_fe | Capacidad baja suficiente | C2 |
| C2 | LGBM capacidad moderada: 15 hojas, min_child 30, 400 iteraciones, lr 0,02 | dt + age_fe | El exceso de regularización limita | C1 (mismas features) |
| C3 | ET min_leaf 10 | dt | Capacidad moderada en ET | et30 dt (min_leaf 30) |
| C4 | LR con spline de edad (8 nodos) + spline de año (4 nodos) + interacción spline_edad × año lineal, C = 0,3 | dt | Interacción edad × año (deriva en 2018+) | lr_spline_age dt |
| C5 | LR con spline de edad × rural y × male, C = 0,3 | dt | Interacción edad × sexo/zona | lr_spline_age dt |
| C6 | ET min_leaf 30 | dt + age_fe | Las features explícitas de edad ayudan a los árboles | et30 dt |
| C7 | ET min_leaf 30 | dt + clim_red | El clima reducido aporta señal sin extrapolar | et30 dt (sin clima) |
| C8 | LR de C4 | dt + clim_red | El clima reducido es robusto en un modelo lineal (prueba especialmente sensible en Mbale) | C4 |

Criterio para proponer un candidato:
- Mejora ≥ 0,005 sobre su comparador en `loc2` **y** en `dist2`, mejorando en al menos 4 de 5 folds de `loc2`.
- Ningún distrito retenido cae más de 0,02.
- Si se cumple, se construye un ensemble con ablación y solo entonces se propone una submission.
- Si ninguna configuración lo cumple, se mantienen s02, s03 y s04.
