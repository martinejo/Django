# Predicción temprana de anomalías (MVP web en Python)

Este repositorio contiene una **base inicial** para construir una aplicación web enfocada en:

1. Importar datos clínicos desde CSV (señales biológicas de monitores de anestesia).
2. Elegir el modelo a entrenar desde la interfaz web.
3. Configurar hiperparámetros por modelo.
4. Entrenar y evaluar rápidamente para comparar enfoques.

## Arquitectura propuesta

- `src/anomaly_web/config.py`: catálogo de modelos y validación de hiperparámetros.
- `src/anomaly_web/training.py`: utilidades de carga de datos, entrenamiento y métricas.
- `streamlit_app.py`: interfaz web para subir CSV, elegir modelo e hiperparámetros.
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

- Columnas de entrada: señales (ejemplo: `hr`, `spo2`, `map`, etc.).
- Columna objetivo binaria/multiclase: definida por el usuario en la app.

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
