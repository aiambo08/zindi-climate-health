# Bloque neuronal acotado (MLP pequeño)

Objetivo: comprobar si un MLP pequeño aporta generalización o complementariedad frente a la logística de referencia (`lr_spline_age`, bloques `demo,time`) y s04. s02/s03/s04 no se modifican; no se generan submissions.

## Configuraciones (fijadas antes de ejecutar)

| | `hidden_layer_sizes` | Features |
|---|---|---|
| N1 | (16,) | dt = `demo,time` (7 columnas) |
| N2 | (32, 16) | dt |
| N3 | (16,) | dt + `clim_red` (3 anomalías relativas 30 d vs 90 d) |
| N4 | (16,) | full = `demo,time,orig,clim` |

Comunes: `activation="tanh"`, `solver="lbfgs"`, `alpha=1.0`, `max_iter=2000`, `max_fun=30000`, `tol=1e-5`, `early_stopping=False`, `random_state=42`.
Pipeline ajustado dentro del train de cada fold: `VarianceThreshold(0)` → `StandardScaler` → `MLPClassifier`. Sin ID, localidad ni coordenadas (comprobado por aserción), sin pesos de clase, probabilidades originales, umbral 0,5. Splits fijos `loc2` (principal) y `dist2` (robustez). s04 se evalúa con su configuración exacta (3 modelos × semillas 42/43/44).

Presupuesto fijado antes de ejecutar: 60 min de cómputo. Consumido: 6,5 min (cribado).

Reproducción: `cd src && python mlp.py` (cribado) · `python mlp.py N1` (confirmación, no ejecutada).

## Resultados globales (OOF, semilla 42)

| | train score | loc2 F1 / AUC / score | dist2 F1 / AUC / score | Δ vs LR loc2 · dist2 | Δ vs s04 loc2 · dist2 |
|---|---|---|---|---|---|
| N1 | 0,843 | 0,795 / 0,801 / 0,7970 | 0,802 / 0,767 / 0,7879 | −0,0160 · −0,0268 | −0,0209 · −0,0283 |
| N2 | 0,950 | 0,775 / 0,747 / 0,7637 | 0,765 / 0,723 / 0,7481 | −0,0494 · −0,0665 | −0,0543 · −0,0681 |
| N3 | 0,844 | 0,783 / 0,777 / 0,7808 | 0,777 / 0,740 / 0,7621 | −0,0322 · −0,0525 | −0,0371 · −0,0541 |
| N4 | 0,903 | 0,759 / 0,717 / 0,7423 | 0,729 / 0,682 / 0,7104 | −0,0708 · −0,1043 | −0,0757 · −0,1058 |
| LR ref. | 0,816 | 0,815 / 0,810 / 0,8130 | 0,818 / 0,810 / 0,8146 | – | −0,0049 · −0,0016 |
| s04 | – | 0,817 / 0,820 / 0,8179 | 0,816 / 0,817 / 0,8162 | +0,0049 · +0,0016 | – |

## Diferencias emparejadas por fold frente a s04

loc2 (5 folds) — ningún MLP mejora en ningún fold:

| fold | N1 | N2 | N3 | N4 |
|---|---|---|---|---|
| 0 | −0,0141 | −0,0421 | −0,0287 | −0,0605 |
| 1 | −0,0217 | −0,0675 | −0,0264 | −0,0712 |
| 2 | −0,0300 | −0,0775 | −0,0477 | −0,0963 |
| 3 | −0,0120 | −0,0429 | −0,0329 | −0,0726 |
| 4 | −0,0280 | −0,0429 | −0,0535 | −0,0785 |

dist2 (distrito retenido; score de s04 en la última columna):

| distrito | n | N1 | N2 | N3 | N4 | s04 |
|---|---|---|---|---|---|---|
| Bugweri | 199 | −0,0273 | −0,0727 | −0,0274 | −0,0293 | 0,8089 |
| Entebbe City | 54 | −0,0164 | −0,0818 | −0,0278 | −0,0610 | 0,7881 |
| Iganga | 2339 | −0,0299 | −0,0681 | −0,0560 | −0,1126 | 0,8161 |
| Kikuube | 128 | −0,0154 | −0,0575 | −0,0632 | −0,1430 | 0,8174 |
| Mayuge | 268 | −0,0205 | −0,0428 | −0,0340 | −0,0654 | 0,8217 |
| Mbale City | 158 | −0,0311 | −0,1120 | −0,0624 | −0,1416 | 0,8330 |

Frente a la LR de referencia el patrón es el mismo: 0/5 folds loc2 y 0/6 distritos mejores para N1–N4 (`reports/mlp/screen_paired_summary.csv`).

## Complementariedad de errores frente a s04 (loc2)

| | corr. prob. | etiquetas distintas | errores solo MLP | errores solo s04 |
|---|---|---|---|---|
| N1 | 0,929 | 339 | 208 | 131 |
| N2 | 0,689 | 663 | 407 | 256 |
| N3 | 0,853 | 515 | 320 | 195 |
| N4 | 0,636 | 694 | 458 | 236 |

La baja correlación no es complementariedad útil: cada MLP introduce bastantes más errores propios de los que corrige.

## Convergencia, advertencias y tiempo (12 ajustes por configuración: 5 loc2 + 6 dist2 + 1 train)

| | iteraciones (min–max) | sin converger | otras advertencias | tiempo total |
|---|---|---|---|---|
| N1 | 314–697 | 0 | 0 | 2,9 s |
| N2 | 2000–2000 | 12 (`ConvergenceWarning`, alcanza `max_iter`) | 0 | 252,5 s |
| N3 | 278–871 | 0 | 0 | 3,2 s |
| N4 | 416–1717 | 0 | 0 | 72,7 s |

Detalle por ajuste en `reports/mlp/screen_convergence.csv`.

## Interpretación

- Con estos parámetros, ningún MLP mejora a la LR ni a s04 en ningún fold de loc2 ni en ningún distrito retenido.
- Más capacidad (N2) o más features (N3, N4) suben el score de train (0,843 → 0,903–0,950) y bajan el OOF: sobreajuste. Las probabilidades de N2/N4 se saturan (percentil 5 ≈ 0,003–0,02, percentil 95 = 1,00), lo que castiga el AUC.
- N2 no converge en ninguno de sus 12 ajustes; su resultado corresponde al estado tras 2000 iteraciones y no a un óptimo.
- N1, el más parecido a la LR, es el menos malo (−0,021 loc2 vs s04), pero sigue peor en todos los folds.
- No se ha probado más regularización ni otras arquitecturas (fuera del bloque aprobado). Esto no demuestra que un MLP no pueda funcionar aquí; solo que estas cuatro configuraciones no lo hacen.

## Decisión

Sin mejora prometedora ni complementariedad concreta, no se ejecuta la confirmación con semillas 42/43/44 y particiones alternativas, ni la media 50/50 con s04 (el modo queda implementado en `src/mlp.py`). No hay submissions nuevas; se mantienen s02/s03/s04.
