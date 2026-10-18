# Bloque de literatura: TabPFN v2 y rezagos climáticos 31–90 días (sin candidatos nuevos)

Cribado con semilla 42 sobre los splits fijos `loc2` (principal) y `dist2` (robustez); umbral 0,5, probabilidades originales, sin pesos de clase, sin AutoML.
s02/s03/s04 congelados sin cambios; sin submissions.

## 1. Fuentes revisadas y relación con hipótesis

| Fuente | Qué aporta | Hipótesis derivada | Estado |
|---|---|---|---|
| Reglas del reto (`docs/rules_snapshot_2026-10-17.txt`) | Solo datos del reto + datos climáticos públicos; modelos preentrenados abiertos permitidos; AutoML prohibido | Admite TabPFN (pesos públicos); excluye datos externos etiquetados | Aplicado |
| APHRC/Climate_and_Health (Hugging Face) | Dataset público etiquetado relacionado con el origen de los datos | Usarlo sería fuga / dato no climático no permitido | **Descartado sin abrirlo para entrenar** |
| Hollmann et al., *Nature* 2025 (TabPFN v2) | Modelo fundacional tabular para n < 10 000 | T1/T2: TabPFN en dt y full; media 50/50 con s04 | Ejecutado (§2) |
| Malaria mortality, Iganga/Mayuge, *BMC Public Health* 2025 (DLNM) | Riesgo asociado a temperatura con rezago 5–11 semanas y lluvia 11–12 semanas, heterogéneo por edad/sexo | Ventana climática 31–90 días antes de la muerte (L1–L4) | Ejecutado (§3) |
| Gorishniy et al. 2024 (TabM), Holzmüller et al. 2024 (RealMLP) | MLP tabulares con ensembles internos / defaults afinados | Alternativa neuronal | No ejecutado (el bloque MLP previo fue claramente negativo) |
| Literatura de verbal autopsy (Jeblee et al. 2019, etc.) | La causa se infiere de síntomas/narrativas, no disponibles aquí; edad y sexo son las únicas variables estructuradas compartidas | Explica que la señal disponible sea casi toda edad/año | Contexto |

## 2. TabPFN v2 (T1 dt, T2 full)

- `tabpfn==2.2.1`, checkpoint por defecto `tabpfn-v2-classifier-finetuned-zk73skhh.ckpt` (sha256 `cf8c519c…0e49`, descarga pública de Prior-Labs en Hugging Face).
  `TabPFNClassifier(n_estimators=4, device="cpu", random_state=42, ignore_pretraining_limits=True)` tras `VarianceThreshold(0)`; sin ajuste de hiperparámetros.
- TabPFN 2.2.1 exige `scikit-learn<1.7`, incompatible con `requirements.txt`. Se ejecuta en un venv aparte (`requirements-tabpfn.txt`) que solo guarda probabilidades en `reports/tabpfn/*.npy`; la comparación se hace en el entorno principal. Los folds se pasan exactos vía `reports/tabpfn/folds.npz`.
- Tiempo CPU: T1 51 s, T2 144 s (11 ajustes cada uno).

| | train score | loc2 F1/AUC/score | dist2 score | Δ vs s04 loc2 (folds mejores) | Δ vs s04 dist2 (distritos mejores) |
|---|---|---|---|---|---|
| T1 TabPFN dt | 0,841 | 0,811 / 0,824 / 0,8159 | 0,8112 | −0,0019 (2/5) | −0,0053 (2/6) |
| T2 TabPFN full | 0,850 | 0,812 / 0,823 / 0,8164 | 0,8124 | −0,0012 (3/5) | −0,0026 (1/6) |
| media 50/50 T1+s04 | – | 0,8185 | 0,8136 | +0,0007 (3/5) | −0,0032 (1/6) |
| media 50/50 T2+s04 | – | 0,8187 | 0,8139 | +0,0009 (3/5) | −0,0006 (2/6) |
| ET30 dt (mejor individual) | 0,838 | 0,8183 | 0,8148 | +0,0005 | −0,0019 |
| s04 | – | 0,8179 | 0,8162 | – | – |

- TabPFN ordena tan bien como ET30 (AUC 0,823–0,824) pero su F1 a 0,5 es menor (0,811–0,812 frente a 0,815–0,817).
- Por distrito (dist2, Δ vs s04): T2 cae en Mbale City (−0,0125) y Bugweri (−0,0065) y gana en Entebbe City (+0,0078, n = 54). La media 50/50 T2+s04 queda entre −0,0028 y +0,0022 en todos los distritos.
- Errores frente a s04 (loc2): correlación 0,979–0,984; T1 comete 66 errores propios y corrige 57. No hay complementariedad útil.
- La media 50/50 gana +0,0007/+0,0009 en loc2 pero pierde en dist2, así que no hay mejora repetible. **No se confirma.**

## 3. Ventana climática 31–90 días (L1–L4)

Features (bloque `clim_lag`, derivadas exclusivamente de `climate_features.csv`, sin etiquetas ni agregados entre filas):
`tavg_31_90d = (90·tavg_90d − 30·tavg_30d)/60` (temperatura media de los días 31–90 previos, ≈ rezago 5–13 semanas) y `rain_sum_31_90d = rain_sum_90d − rain_sum_30d`.
Cada modelo se compara con el mismo modelo sin esas dos columnas (R1–R4); L4/R4 = media simple ET30 + LightGBM lento + LR spline (estructura s04).

Cribado (semilla 42):

| | loc2 score | Δ vs referencia loc2 (folds mejores) | dist2 score | Δ vs referencia dist2 (distritos mejores) | etiquetas cambiadas loc2 |
|---|---|---|---|---|---|
| L1 ET30 | 0,8186 | +0,0003 (3/5) | 0,8136 | −0,0025 (0/6) | 23 |
| L2 LR spline | 0,8146 | +0,0013 (3/5) | 0,8153 | +0,0012 (2/6) | 55 |
| L3 LightGBM lento | 0,8155 | +0,0004 (2/5) | 0,8104 | +0,0026 (3/6) | 115 |
| L4 media | 0,8183 | +0,0004 (3/5) | 0,8162 | +0,0015 (3/6) | 39 |

L4 se llevó a confirmación porque fue el único con signo positivo en ambos esquemas. Se usaron las semillas 42/43/44 (variación del estimador) y las particiones loc2 alternativas r43/r44/r45 (variación de partición, que no son validaciones independientes):

| Δ L4 − R4 | loc2 fijo | dist2 | loc2_r43 | loc2_r44 | loc2_r45 |
|---|---|---|---|---|---|
| semilla 42 | +0,0004 | +0,0015 | −0,0011 | −0,0008 | −0,0007 |
| semilla 43 | +0,0002 | +0,0021 | −0,0011 | −0,0007 | +0,0008 |
| semilla 44 | −0,0002 | +0,0022 | −0,0007 | −0,0006 | 0,0000 |
| media 3 semillas vs s04 | +0,0002 | +0,0024 | −0,0011 | −0,0009 | −0,0001 |

- dist2 mejora de forma estable con las tres semillas (+0,0015 a +0,0024), sobre todo en Kikuube (+0,006) y Mbale City (+0,005).
- En las particiones loc2 alternativas, en cambio, la diferencia es negativa en 2 de 3, y en la loc2 fija es prácticamente nula. Cambian unas 40 etiquetas (correlación 0,997–0,998).
- **No se confirma:** la mejora no es repetible entre particiones del esquema principal. El resultado es compatible con una señal climática de rezago muy débil, pero no demuestra que el clima carezca de señal.

## 4. Decisión

Ningún enfoque derivado de la literatura mejora de forma repetible; se mantienen s02/s03/s04. Sin submissions.

## Reproducción

```bash
cd src
python tabpfn_exp.py folds                                   # entorno principal: escribe reports/tabpfn/folds.npz
python -m venv ~/venv-tabpfn && ~/venv-tabpfn/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 && ~/venv-tabpfn/bin/pip install -r ../requirements-tabpfn.txt
~/venv-tabpfn/bin/python tabpfn_dump.py T1 42 loc2 dist2      # descarga el checkpoint en el primer uso
~/venv-tabpfn/bin/python tabpfn_dump.py T2 42 loc2 dist2
python tabpfn_exp.py                                         # entorno principal: tablas → reports/tabpfn/
python lag.py && python lag.py confirm                       # bloque L → reports/lag/
```
