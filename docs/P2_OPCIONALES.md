# P2 — Experimentos opcionales: calibración, interacciones edad × zona/mes y stacking

Script: `src/p2.py` (`python p2.py dump 42 && python p2.py dump 43 && python p2.py dump 44 && python p2.py eval`, ≈45 min de cómputo en total).
Informes: `reports/p2/` (`delta_wide.csv`, `global.csv`, `paired_folds.csv`, `paired_summary.csv`, `errors.csv`, `fold_info.csv`, `eval.log`).
La caché de OOF (`data/cache/`) no se versiona.

**Resultado: ningún experimento cumple el criterio de aceptación. s02, s03 y s04 siguen congelados (hashes sin cambios) y no se generan submissions.**

## Protocolo común

- Referencia: s04 = media 1/3 de `et30_mf1`, `lgbm_slow` y `lr_spline_age` sobre demo+time.
  - Por semilla (`R4@s`): componentes con semilla `s`.
  - `s04`: media de las semillas 42/43/44.
- El harness reproduce los valores previos:
  - s04: loc2 0,8179 / dist2 0,8162.
  - `R4@42`: 0,8177 / 0,8160.
  - s02: 0,8208 / 0,8104.
  - s03: 0,8198 / 0,8103.
- Esquemas:
  - `loc2` (principal).
  - `dist2` (robustez).
  - `loc2_r43/r44/r45`: particiones alternativas agrupadas por localidad. Miden la variación de partición, no son validaciones independientes.
- Cada candidato se compara de dos formas:
  - por semilla, frente a la referencia de la misma semilla;
  - como media de 3 semillas (`_3s`), frente a s04.
  - Siempre con diferencias emparejadas sobre los mismos folds.
- Todo lo que se ajusta (calibrador, stacker) usa **OOF internos**:
  - `StratifiedGroupKFold` de 5 folds por localidad (loc2), `random_state` = semilla.
  - Se construyen solo sobre el train de cada fold externo y se aplican al fold externo. El fold de validación nunca se ve.
- Restricciones:
  - Umbral 0,5 fijo.
  - Sin pesos de clase.
  - Sin ID, localidad ni coordenadas como features.
  - Parámetros fijados antes de ejecutar.

Criterio de aceptación (fijado en el encargo): mejora emparejada en loc2 **y** en dist2, ≥4/5 folds loc2, confirmación con 3 semillas y ≥2 particiones alternativas.

## 1. Calibración en [0,4, 0,6]

| Variante | Qué hace |
|---|---|
| ISOB | Isotónica ajustada solo con los puntos internos con p ∈ [0,4, 0,6] y aplicada solo a p ∈ [0,4, 0,6]; fuera de la banda, p sin cambios |
| ISOF | Isotónica en todo el rango (comparación) |
| TEMP | Temperature scaling, p' = σ(logit(p)/T), T por log-loss interna |

TEMP es un **control nulo**: deja 0,5 fijo y conserva el orden de las probabilidades, así que no puede cambiar el F1 a 0,5 ni el AUC. Se verifica que cambia 0 etiquetas y da Δ = 0,0000 en todos los esquemas.

Δ score frente a la referencia emparejada:

| | loc2 | dist2 | r43 | r44 | r45 |
|---|---:|---:|---:|---:|---:|
| ISOB@42 | +0,0009 | −0,0011 | −0,0029 | −0,0009 | +0,0004 |
| ISOB@43 | +0,0012 | −0,0013 | −0,0015 | +0,0003 | +0,0013 |
| ISOB@44 | −0,0012 | −0,0018 | 0,0000 | −0,0010 | +0,0014 |
| **ISOB 3 semillas vs s04** | **+0,0010** (4/5 folds) | **−0,0008** (2/6 distritos) | −0,0005 | +0,0003 | 0,0000 |
| ISOF 3 semillas vs s04 | +0,0001 | −0,0002 | −0,0018 | −0,0006 | −0,0007 |
| TEMP 3 semillas vs s04 | 0,0000 | 0,0000 | 0,0000 | 0,0000 | 0,0000 |

- **Componentes de ISOB (3 semillas):**
  - loc2: F1 +0,0021, AUC −0,0006.
  - dist2: F1 +0,0004, AUC −0,0026.
  - La isotónica crea empates dentro de la banda, y eso reduce el AUC.
- **Etiquetas y errores:** cambia 27 etiquetas en loc2 (11 errores nuevos, 16 corregidos) y 66 en dist2 (38 nuevos, 28 corregidos).
- **Por distrito (dist2):**
  - Mejora en Kikuube (+0,0068) y Mayuge (+0,0011).
  - Empeora en Iganga (−0,0014, n = 2339), Bugweri (−0,0012), Entebbe City y Mbale City.
- **Veredicto:** cumple ≥4/5 folds en loc2, pero empeora dist2 y no se confirma en las particiones alternativas. **Rechazado.**

En la práctica, ISOB solo mueve etiquetas cerca de 0,5, así que su efecto equivale a desplazar el umbral localmente. Por eso no se aceptaría con una evidencia débil.

## 2. Interacciones C5/C6 (edad × zona y edad × mes) en árboles

- Bloque `dtix` = demo+time más tres interacciones:
  - `age_x_rural` = age·rural;
  - `age_x_doysin` = age·sin(día del año);
  - `age_x_doycos` = age·cos(día del año).
- El mes se representa en forma cíclica, igual que en `time`.
- Cada modelo se compara con el mismo modelo en dt.

| Δ vs mismo modelo en dt (3 semillas) | loc2 | dist2 | r43 | r44 | r45 |
|---|---:|---:|---:|---:|---:|
| ET30 + ix | −0,0020 (0/5) | −0,0118 | −0,0010 | −0,0009 | −0,0009 |
| LGBM lento + ix | −0,0023 (2/5) | −0,0068 | +0,0011 | −0,0005 | −0,0015 |
| S04ix (ET_ix, LGBM_ix, LR dt) vs s04 | −0,0004 (2/5) | −0,0039 | +0,0002 | +0,0010 | 0,0000 |

- **Signo por semilla:** consistente. ET + ix es negativo en loc2 y dist2 con las 3 semillas.
- **Errores:**
  - En dist2, ET + ix cambia 359 etiquetas (230 errores nuevos frente a 129 corregidos).
  - Sube la fracción de positivos predichos de 0,68 a 0,78.
  - Las interacciones con la edad extrapolan peor a distritos no vistos.
- **Por distrito, S04ix:** mejora en Bugweri, Kikuube, Mayuge y Mbale City, pero pierde 0,0064 en Iganga (n = 2339) y 0,0087 en Entebbe City.
- **Veredicto:** **rechazado.**

## 3. Stacking (regresión logística L2 fuerte)

- Stacker: `LogisticRegression(C=0.1)` sobre logit(p) de los OOF internos.
- C fijado de antemano, sin búsqueda.
- Dos variantes:
  - STK3: los 3 componentes de s04.
  - STK6: los componentes únicos de s02/s03/s04 (ET, LGBM y LR en dt y en full).

| Δ vs referencia | loc2 | dist2 | r43 | r44 | r45 |
|---|---:|---:|---:|---:|---:|
| STK3 3 semillas vs s04 | −0,0005 (3/5) | −0,0017 | −0,0019 | −0,0020 | −0,0008 |
| STK6 3 semillas vs s04 | −0,0005 (2/5) | −0,0030 | −0,0044 | −0,0023 | −0,0020 |

- **Métricas:** el stacking sube algo el AUC en loc2 (+0,0020/+0,0024), pero baja el F1 (−0,0022/−0,0025).
- **Coeficientes:**
  - Ponderan sobre todo ET.
  - En STK3, los pesos de ET, LGBM y LR rondan 0,5–0,75, 0,2–0,5 y 0–0,15.
  - Intercepto ≈ −0,03.
  - Los coeficientes varían entre folds (`fold_info.csv`).
- **Veredicto:** negativo en todos los esquemas y semillas salvo un caso aislado. **Rechazado.**

## 4. W3 (anomalías NASA POWER)

Sin recálculo; ver `docs/REZAGOS_SEMANALES.md`.
- Con 3 semillas frente a s04: +0,0013 en loc2 y +0,0022 en dist2 en los splits fijos.
- En las particiones alternativas: −0,0005, −0,0001 y +0,0006.
- No se usa para la selección final. Como mucho serviría como sonda informativa en el leaderboard, y solo con aprobación explícita.

## Conclusión

Ninguno de los tres experimentos supera a s04 de forma emparejada y repetible en loc2 y dist2.
- La única ganancia en loc2 es la de ISOB (+0,0010, 4/5 folds). Viene de mover unas 27 etiquetas cerca de 0,5 y no se sostiene en dist2 ni en las particiones alternativas.
- Esto no demuestra que no exista mejora posible; solo que estas variantes no la aportan con la evidencia disponible.

Se mantienen s02, s03 y s04.
