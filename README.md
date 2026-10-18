# zindi-climate-health

Clasificador reproducible para el *Climate Risk and Health Prediction Challenge* (Zindi). Predice si una muerte registrada en Uganda es "climate-sensitive" a partir de la edad, el sexo, la zona, la fecha y el clima. Score = 0.6·F1(umbral 0.5) + 0.4·ROC-AUC.

## Resultado actual (validación OOF, 5 folds)

| Candidato | Agrupado por localidad (principal) | Estratificado | Agrupado por fecha |
|---|---|---|---|
| `s02` ExtraTrees(leaf 30) + LightGBM lento, 3 semillas | **0.8151** (F1 0.811, AUC 0.821) | 0.8154 | 0.8201 |
| `s03` = s02 + LogReg con spline de edad | 0.8140 (F1 0.812, AUC 0.817) | 0.8170 | 0.8167 |
| Benchmark oficial (LB público) | 0.763 | | |

El esquema principal es el agrupado por localidad: el 97,5 % de las filas de test pertenecen a localidades ausentes de train ([`reports/audit_raw.txt`](reports/audit_raw.txt)).

## Pasos exactos

```bash
# 1. Python 3.11
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 2. Copiar los 7 ficheros del reto (Train.csv, Test.csv, climate_features.csv, SampleSubmission.csv, ...) en data/raw/
# 3. Regenerar particiones y los dos CSV candidatos (~2 min en 8 CPU, sin GPU)
bash reproduce.sh
```

Comprobado en un venv limpio con `requirements.txt`: las particiones (`data/splits.csv`) salen idénticas byte a byte; los CSV tienen `TargetF1` idéntico y `TargetRAUC` con diferencia máxima de 3e-16. ExtraTrees con `n_jobs=8` suma los árboles en orden no determinista, así que el hash SHA-256 puede variar. Para comparar un CSV regenerado con otro de referencia:

```bash
python src/verify.py submissions/s02_et30mf1_lgbmslow.csv /ruta/a/referencia.csv   # imprime "OK maxdiff ..."
```

`predict.py` aborta si falla algún control final: 1030 filas, columnas exactas `ID,TargetF1,TargetRAUC`, mismo orden de ID que `SampleSubmission`, `TargetF1 == (TargetRAUC >= 0.5)`, probabilidades finitas en [0,1] sin redondear.

## Otros comandos

| Comando (desde `src/`) | Qué hace |
|---|---|
| `python audit.py` | Auditoría de datos (fase 1) |
| `python splits.py` | Particiones `strat`, `loc`, `date`, `year` → `data/splits.csv` |
| `python exp_baselines.py`, `exp2.py`, `exp3.py`, `exp4.py` | Bloques de experimentos; añaden líneas a `reports/experiments.jsonl` y guardan las OOF en `oof/` |
| `python ens.py a,b c,d` | Score OOF de promedios simples de probabilidades |
| `python predict.py NOMBRE m1,m2 bloques SEMILLAS` | Entrena con todo train y escribe un CSV validado |
| `python mlp.py` | Bloque MLP N1–N4 (cribado, semilla 42, loc2/dist2) → `reports/mlp/`, informe en `docs/MLP.md` |
| `python tabpfn_exp.py` | TabPFN v2 T1/T2 (probabilidades de `tabpfn_dump.py` en venv aparte, `requirements-tabpfn.txt`) → `reports/tabpfn/`, informe en `docs/LITERATURA.md` |
| `python lag.py [confirm]` | Ventana climática 31–90 días L1–L4 (cribado / confirmación 3 semillas + particiones alternativas) → `reports/lag/`, informe en `docs/LITERATURA.md` |

## Estructura

- `src/common.py`: carga, join `validate="one_to_one"`, bloques de features y métrica.
- `src/models.py`: configuraciones explícitas de los modelos (sin búsqueda automática).
- `src/cv.py`: OOF por esquema, métricas globales y por fold, matriz de confusión.
- `docs/`: cumplimiento (fase 0), copia de las reglas y [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## Cumplimiento

No se usa AutoML, no se ajusta el umbral, no se redondean probabilidades y no se usan datos externos. El ID solo sirve para los joins. Detalle en [`docs/fase0_cumplimiento.md`](docs/fase0_cumplimiento.md).
