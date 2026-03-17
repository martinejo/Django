# Predicción temprana de anomalías (MVP web en Python)

Este repositorio contiene una **base inicial** para construir una aplicación web enfocada en:

1. Importar datos clínicos desde CSV (señales biológicas de monitores de anestesia).
2. Elegir el modelo a entrenar desde la interfaz web.
3. Configurar hiperparámetros por modelo.
4. Entrenar y evaluar rápidamente para comparar enfoques.
5. Visualizar señales por paciente (una o varias en la misma gráfica).

## Arquitectura propuesta

- `src/anomaly_web/config.py`: catálogo de modelos y validación de hiperparámetros.
- `src/anomaly_web/training.py`: utilidades de carga de datos, entrenamiento y métricas.
- `streamlit_app.py`: interfaz web para subir CSV, elegir modelo e hiperparámetros.
- `data/examples/`: CSV simulados para probar la app sin subir archivos propios.
- `deploy/`: plantillas de servicio y script de despliegue en VPS.
- `tests/`: pruebas unitarias de validación de configuración.

## Modelos iniciales incluidos

- `random_forest`
- `gradient_boosting`
- `mlp`

> Todos están implementados con `scikit-learn` en esta primera versión.

## Arranque rápido local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Formato mínimo esperado del CSV

- Columna de paciente: `patient_id`.
- Columna temporal: `minute` (simulación de 20 minutos por paciente).
- Columnas de entrada: señales (ejemplo: `hr`, `spo2`, `map`, `etco2`, `rr`, `bis`).
- Columna objetivo binaria: `anomaly` (minuto del evento = 1, resto = 0).

El entrenamiento se hace con ventana deslizante:
- `tam_ventana`: cuántas muestras se usan para construir cada ejemplo.
- `horizonte_prediccion`: cuántas muestras antes del evento se consideran positivas.

Así, las ventanas dentro del horizonte previo al evento se etiquetan como 1 para detección anticipada.

## CSV de ejemplo incluidos

Desde la interfaz puedes elegir `Usar CSV de ejemplo` y pulsar `Cargar CSV de ejemplo`.

- `hipotension_induccion.csv`: episodio de caída de MAP durante inducción anestésica.
- `hipoxemia_ventilacion.csv`: descenso transitorio de SpO2 con cambios respiratorios.
- `evento_mixto_hemodinamico.csv`: inestabilidad combinada (FC/MAP/BIS).

Todos los ejemplos incluyen:
- 1000 pacientes simulados.
- 20 minutos por paciente.
- Evento entre minuto 10 y 12 en el 30% de pacientes.
- Señales típicas de quirófano (`hr`, `spo2`, `map`, `etco2`, `rr`, `bis`) y columna `anomaly`.

## Despliegue en VPS (sin dominio)

Esta base ya incluye una ruta de despliegue por IP/host con `systemd` + `nginx`.

### 1) Copiar el proyecto al servidor

```bash
git clone <tu_repo> /opt/anomaly-web
cd /opt/anomaly-web
```

### 2) Ejecutar setup automático en el VPS

```bash
bash deploy/setup_vps.sh
```

> Este setup está diseñado para **no tocar firewall ni configuración de red** (no usa `ufw`, no reinicia interfaces). Así se minimiza el riesgo de perder conexión a Internet/SSH durante la instalación.

### 3) Verificar servicios

```bash
systemctl status anomaly-web --no-pager
systemctl status nginx --no-pager
journalctl -u anomaly-web -n 100 --no-pager
```

La app quedará accesible por `http://<IP-o-host-del-servidor>/`.

## Próximos pasos recomendados

- Añadir pipelines de preprocesamiento (escalado, imputación, feature engineering temporal).
- Añadir validación cruzada y búsqueda de hiperparámetros (GridSearch/Optuna).
- Versionado de experimentos (MLflow o equivalente).
- Módulo de inferencia en tiempo real y alertas.
- Endurecer seguridad y trazabilidad (auditoría de datasets y modelos).
