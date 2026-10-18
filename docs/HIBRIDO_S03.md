# Híbridos de s03 (H1, H2)

Script: `src/hybrid.py`. Usa la caché OOF de `src/p2.py`; si no existe, primero `python p2.py dump 42/43/44`. Informes en `reports/hybrid/` y `reports/hybrid.log`.

Las dos variantes se fijaron antes de evaluar. Se usan 3 semillas y diferencias emparejadas por fold en loc2, dist2 y loc2_r43/r44/r45. Umbral 0,5, sin pesos de clase.

| Variante | Definición |
|---|---|
| H1 | Media 1/3 de `et30_mf1` y `lgbm_slow` con todas las features (como s03) y `lr_spline_age` solo con demo+time. Es s03 sin clima en la LR, cuyo clima absoluto colapsa en Mbale City. |
| H2 | Media 50/50 de s03 y s04. |

## Resultados

| Δ score | loc2 | dist2 | r43 | r44 | r45 |
|---|---:|---:|---:|---:|---:|
| H1 vs s03 | −0,0007 (1/5 folds) | +0,0035 (4/6) | +0,0019 | −0,0003 | +0,0008 |
| H1 vs s04 | +0,0011 (4/5) | −0,0025 (4/6) | −0,0010 | −0,0004 | +0,0010 |
| H2 vs s03 | −0,0019 (1/5) | +0,0040 (5/6) | +0,0020 | +0,0004 | +0,0003 |
| H2 vs s04 | 0,0000 (3/5) | −0,0019 (4/6) | −0,0009 | +0,0002 | +0,0005 |

Scores globales:
- H1: loc2 0,8190 (F1 0,818 / AUC 0,820), dist2 0,8138.
- H2: loc2 0,8179, dist2 0,8143.
- Referencias: s03 0,8198 / 0,8103 y s04 0,8179 / 0,8162.

Por semilla: frente a s03, H1 y H2 tienen el mismo signo con las 3 semillas en loc2 (negativo) y dist2 (positivo). Frente a s04, H1 da +0,0009/+0,0015/0,0000 en loc2 y H2 cambia de signo entre semillas (+0,0003/−0,0001/−0,0005).

Por distrito (H1 vs s03):
- Casi toda la ganancia en dist2 viene de Mbale City (+0,042), donde H1 corrige el colapso de la LR con clima.
- Bugweri pierde 0,006 y Kikuube 0,003.
- Iganga gana 0,002.

## Conclusión

Ningún híbrido domina a la vez a s03 y a s04. H1 y H2 quedan entre ambos:
- frente a s03, cambian algo de loc2 por robustez en dist2;
- frente a s04, ocurre lo contrario.

Según el criterio fijado, no se genera candidato. s02, s03 y s04 siguen congelados. Hay una copia adicional de s03 fuera del repo, con el mismo sha256.
