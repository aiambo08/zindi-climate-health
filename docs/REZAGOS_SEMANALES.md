# Rezagos climáticos semanales de 5–12 semanas (bloque W)

**Conclusión: no hay mejora repetible.** No se propone ningún candidato nuevo: s02, s03 y s04 siguen congelados y no se ha hecho ninguna submission.

## Hipótesis
El estudio de mortalidad por malaria de Iganga/Mayuge (BMC Public Health 2025, DLNM) encuentra asociaciones con la temperatura a unas 5–11 semanas y con la lluvia a unas 11–12 semanas. En `docs/LITERATURA.md` se aproximó con la ventana 31–90 días, que solo se puede derivar de `climate_features.csv`. Aquí se construyen los rezagos semana a semana a partir de series climáticas diarias públicas.

## Datos (solo clima público, sin etiquetas)
- **Primer intento: ERA5 vía la API de archivo de Open-Meteo.** Se descargaron 42 de las 55 coordenadas y después la API devolvió `Daily API request limit exceeded`. Para no mezclar fuentes entre localidades se descartó por completo.
- **Fuente usada: NASA POWER** (API pública sin clave), con `src/download_power.py`:
  - `T2M`: temperatura media diaria a 2 m (MERRA-2).
  - `PRECTOTCORR`: precipitación diaria corregida.
  - Cubre las 55 coordenadas únicas de Train+Test, de 2000-01-01 a 2022-12-31: 462 055 filas sin valores de relleno.
  - El CSV resultante (`data/external/power_daily.csv`) tiene sha256 `d9a9f0e90917287e0b2e7f246b75d2953d3188b810aa1f270e79956229ba20e4`. Se excluye de git y se regenera con el script.
- **Comprobaciones:**
  - La lluvia de los 30 días previos calculada con POWER correlaciona 0,84 con `rain_sum_30d`, y la temperatura media de 30 días correlaciona 0,84 con `tavg_30d`.
  - Las semanas 5 y 11 y la lluvia de las semanas 11–12 de una fila se recalcularon a mano y coinciden con la implementación.

## Features (fijadas antes de ver resultados; `src/weekly.py`)
El día 1 es el día anterior a la muerte; el día de la muerte se excluye. La semana k abarca los días 7(k−1)+1 a 7k antes de la muerte.
- **wk** (16 features): `tmean_wk5…wk12` (media) y `rain_wk5…wk12` (suma).
- **lit** (2 features): `t_w5_11` (media de temperatura en los días 29–77) y `r_w11_12` (suma de lluvia en los días 71–84).
- **lit_anom** (2 features): `lit` menos la climatología 2001–2006 de la misma ventana, en la misma coordenada y el mismo día del año.
  - Esa climatología es anterior a todas las muertes (la primera es de 2007-10).
  - El año 2000 se excluye porque algunas ventanas empezarían en 1999.
  - La anomalía pretende evitar extrapolar valores absolutos a distritos no vistos (el colapso en Mbale de `docs/DIAGNOSTICO.md`).

Las configuraciones W1, W2 y W3 usan la estructura s04 (media simple de ET30, LightGBM lento y LR con spline de edad, con los parámetros de `models.make`), con las features dt más wk, lit o lit_anom respectivamente. La referencia R4 es la misma estructura solo con dt. En todas: sin pesos de clase, umbral 0,5, probabilidades originales y ningún ID, localidad ni coordenada como feature.

## Cribado (semilla 42, splits fijos)
Δ es la diferencia de score de W frente a R4 sobre los mismos folds (media de las diferencias por fold, con el número de folds en que mejora entre paréntesis).

| | score train | loc2 | Δ loc2 | dist2 | Δ dist2 | etiquetas OOF cambiadas (loc2) |
|---|---|---|---|---|---|---|
| W1 (wk) | 0,8420 | 0,8142 | −0,0040 (1/5) | 0,8125 | −0,0015 (3/6) | 80 |
| W2 (lit) | 0,8358 | 0,8187 | +0,0007 (3/5) | 0,8154 | −0,0006 (4/6) | 39 |
| W3 (lit_anom) | 0,8373 | 0,8189 | +0,0011 (4/5) | 0,8161 | +0,0020 (5/6) | 24 |
| R4 | 0,8354 | 0,8177 | – | 0,8160 | – | – |
| s04 | – | 0,8179 | – | 0,8162 | – | – |

- **W1:** los 16 rezagos semanales sobreajustan. Sube el score de train y baja el OOF en 4 de 5 folds.
- **W3 por distrito** (Δ frente a R4): Bugweri +0,0010, Entebbe City +0,0007, Iganga −0,0008 (2339 filas), Kikuube +0,0025, Mayuge +0,0054 y Mbale City +0,0032. No colapsa en ningún distrito retenido.
- **Errores frente a R4 en loc2:** W3 tiene 9 errores propios y corrige 15 de los de R4, con correlación 0,9986.

## Confirmación de W3 (semillas 42/43/44 y particiones loc2 alternativas)
La variación por semilla se presenta separada de la variación por partición. Las particiones alternativas no son validaciones independientes.

Media de las diferencias por fold de W3 frente a R4 con la misma semilla (folds en que mejora):

| | loc2 | dist2 | loc2_r43 | loc2_r44 | loc2_r45 |
|---|---|---|---|---|---|
| semilla 42 | +0,0011 (4/5) | +0,0020 (5/6) | −0,0001 (2/5) | −0,0004 (2/5) | −0,0007 (1/5) |
| semilla 43 | +0,0007 (3/5) | +0,0013 (3/6) | −0,0013 (1/5) | −0,0007 (2/5) | +0,0006 (2/5) |
| semilla 44 | +0,0001 (3/5) | +0,0019 (4/6) | 0,0000 (2/5) | −0,0001 (2/5) | +0,0013 (3/5) |
| W3 (3 semillas) vs s04 | +0,0013 (4/5) | +0,0022 (4/6) | −0,0005 (2/5) | −0,0001 (3/5) | +0,0006 (2/5) |

En score global OOF, W3 con 3 semillas frente a s04 da: loc2 0,8193 frente a 0,8179; dist2 0,8158 frente a 0,8162; r43 0,8197 frente a 0,8200; r44 0,8195 frente a 0,8195; r45 0,8187 frente a 0,8181.

## Decisión
- **W3 no se confirma.** La mejora se limita a los splits fijos. En las tres particiones loc2 alternativas es nula o negativa en promedio, y en score global dist2 queda por debajo de s04.
- Es el mismo patrón que la ventana 31–90 días de `docs/LITERATURA.md`: puede haber una señal climática retardada muy débil (W3 mejora 5 de 6 distritos con semilla 42), pero su tamaño (unas 30 etiquetas cambiadas) no basta para distinguirla de la variación por partición.
- Esto no demuestra que el clima carezca de señal ni que se haya alcanzado el techo del problema.
- W1 y W2 no se confirmaron porque no eran prometedores en el cribado.

## Reproducción
```bash
cd src
python download_power.py        # ~1 min; data/external/power_daily.csv (sha256 arriba)
python weekly.py                # cribado W1–W3 → reports/weekly/screen_*.csv (~2,5 min)
python weekly.py confirm W3     # confirmación → reports/weekly/confirm_W3_*.csv (~4 min)
```
