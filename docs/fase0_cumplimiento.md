# Fase 0 — Cumplimiento y recursos

Consultado: 2026-10-17 12:40 UTC, vía API pública `api.zindi.africa/v1/competitions/climate-risk-health-prediction-challenge` (copia literal: `rules_snapshot_2026-10-17.txt`).

## Calendario
- Cierre: **2026-10-18 21:59 UTC** (= 23:59 hora peninsular, CEST). Quedan ~33 h desde la consulta.
- Congelación interna propuesta: 2026-10-18 19:00 UTC (≈3 h de margen).
- Elección de 2 finalistas antes del cierre; si no se eligen, Zindi toma las 2 mejores públicas.

## Submissions
- 10/día, 300 en total. Las que fallan en procesamiento no cuentan.
- No verificado: zona horaria del reinicio diario y cuántas lleva el usuario (requiere su cuenta).
- Split público/privado inconsistente en las reglas: resumen dice 30/70, texto completo dice 20/80. Se asume que la parte privada es mayoritaria; no usar el LB como oráculo.

## Restricciones
- Solo herramientas/paquetes open-source y públicos; nada de servicios de pago ni pruebas con tarjeta.
- Datos: solo los del reto + datasets climáticos públicos (la página de datos cita ERA5, CHIRPS, NDVI, LST).
- Modelos preentrenados: permitidos si son abiertos.
- AutoML prohibido.
- Prohibido fijar umbrales o redondear probabilidades; TargetF1 = umbral 0.5 (página Evaluation).
- Prohibido explotar fugas (data leaks).
- Revisión de código: top 10 tiene 48 h; semillas fijas; el código debe reproducir el score; sin paquetes custom.
- Compartir código fuera del equipo: prohibido salvo en el foro público.

## Puntos ambiguos (bloquean la fase competitiva)
1. Las reglas no mencionan asistentes de IA/agentes de código. Devin es un servicio de pago; la regla "no paid services" se refiere a herramientas para construir el modelo. El código y el modelo final serían 100 % open-source y reproducibles sin Devin, pero el encaje es ambiguo.
2. El código vive en una VM de Cognition: ¿cuenta como "compartir código fuera del equipo"? Probablemente no (es tu herramienta), pero no está escrito.

## Recursos (VM de Devin)
- 8 CPU, 31 GB RAM, sin GPU, 103 GB libres. Python 3.11.13; pandas/sklearn/lightgbm/catboost no instalados (instalables con pip).

## Benchmark oficial (público)
- AUC 0.738, F1 0.780 → score ≈ 0.763.

## Datos
- No están en la VM. La descarga requiere sesión de Zindi. Ficheros esperados: Train.csv, Test.csv, climate_features.csv, SampleSubmission.csv, data_dictionary.csv, downloaded_climate_features_data_dictionary.csv, starter notebook.
