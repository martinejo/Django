"""Interfaz web simple para entrenar modelos de anomalías desde CSV."""

import ast
from pathlib import Path

import pandas as pd
import streamlit as st

from src.anomaly_web.config import MODEL_REGISTRY
from src.anomaly_web.training import read_csv, train_and_evaluate

EXAMPLES_DIR = Path(__file__).parent / "data" / "examples"
EXAMPLE_DATASETS = {
    "Hipotensión durante inducción": {
        "file": "hipotension_induccion.csv",
        "description": "Caídas de presión arterial media (MAP) con compensación de frecuencia cardíaca.",
    },
    "Hipoxemia por ventilación subóptima": {
        "file": "hipoxemia_ventilacion.csv",
        "description": "Descenso de SpO2 y alteraciones de EtCO2 con cambios respiratorios.",
    },
    "Evento hemodinámico mixto": {
        "file": "evento_mixto_hemodinamico.csv",
        "description": "Inestabilidad combinada de MAP, FC y BIS en diferentes momentos.",
    },
}

st.set_page_config(page_title="Predicción temprana de anomalías", layout="wide")
st.title("Predicción temprana de anomalías")
st.caption("Carga CSV, elige modelo e hiperparámetros, y entrena en minutos.")

source = st.radio(
    "Fuente de datos",
    options=["Subir CSV", "Usar CSV de ejemplo"],
    horizontal=True,
)

data = None

if source == "Subir CSV":
    uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Esperando archivo CSV...")
        st.stop()

    try:
        data = read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"No se pudo leer el CSV: {exc}")
        st.stop()
else:
    example_key = st.selectbox("Selecciona un CSV de ejemplo", options=list(EXAMPLE_DATASETS.keys()))
    st.caption(EXAMPLE_DATASETS[example_key]["description"])

    if st.button("Cargar CSV de ejemplo"):
        st.session_state["selected_example_file"] = EXAMPLE_DATASETS[example_key]["file"]
        st.session_state["selected_example_name"] = example_key

    selected_file = st.session_state.get("selected_example_file")
    selected_name = st.session_state.get("selected_example_name")

    if selected_file:
        example_path = EXAMPLES_DIR / selected_file
        try:
            data = read_csv(example_path)
            st.success(f"CSV cargado: {selected_name}")
        except Exception as exc:
            st.error(f"No se pudo cargar el CSV de ejemplo: {exc}")
            st.stop()
    else:
        st.info("Selecciona un CSV de ejemplo y pulsa el botón de carga.")
        st.stop()


st.subheader("Vista rápida del dataset")
st.dataframe(data.head(20), use_container_width=True)

columns = list(data.columns)
default_target_index = columns.index("anomaly") if "anomaly" in columns else 0
target_col = st.selectbox("Selecciona la columna objetivo", options=columns, index=default_target_index)

st.subheader("Visualización de señales")
time_column = None
if "second" in data.columns:
    time_column = "second"
elif "minute" in data.columns:
    time_column = "minute"

if "patient_id" in data.columns and time_column is not None:
    anomaly_column = "anomaly" if "anomaly" in data.columns else target_col
    patient_summary = (
        data.groupby("patient_id", dropna=False)[anomaly_column]
        .max()
        .reset_index()
        .rename(columns={anomaly_column: "has_anomaly"})
    )

    patient_options = patient_summary["patient_id"].tolist()
    anomaly_map = dict(
        zip(patient_summary["patient_id"], patient_summary["has_anomaly"].astype(int).tolist())
    )
    selected_patient = st.selectbox(
        "Paciente para visualizar",
        options=patient_options,
        format_func=lambda pid: f"{pid} {'(anomalía)' if anomaly_map.get(pid, 0) == 1 else ''}".strip(),
    )

    excluded = {"patient_id", time_column, target_col}
    signal_candidates = [
        col for col in data.columns if col not in excluded and pd.api.types.is_numeric_dtype(data[col])
    ]
    selected_signals = st.multiselect(
        "Señales a mostrar",
        options=signal_candidates,
        default=signal_candidates[:1],
    )

    if selected_signals:
        patient_df = (
            data[data["patient_id"] == selected_patient]
            .sort_values(time_column)
            .set_index(time_column)[selected_signals]
        )
        st.line_chart(patient_df, use_container_width=True)
    else:
        st.info("Selecciona al menos una señal para mostrar la gráfica.")
else:
    st.info(
        "Para graficar por paciente se esperan columnas 'patient_id' y una temporal ('second' o 'minute')."
    )

st.subheader("Configuración de entrenamiento por ventana")
window_size = st.number_input(
    "tam_ventana (muestras por ventana)",
    min_value=2,
    max_value=120,
    value=5,
    step=1,
)
prediction_horizon = st.number_input(
    "horizonte_prediccion (muestras de antelación)",
    min_value=1,
    max_value=120,
    value=3,
    step=1,
)

model_key = st.selectbox(
    "Modelo",
    options=list(MODEL_REGISTRY.keys()),
    format_func=lambda key: MODEL_REGISTRY[key].display_name,
)

default_params = MODEL_REGISTRY[model_key].defaults
params_text = st.text_area(
    "Hiperparámetros (dict Python)",
    value=str(default_params),
    height=180,
)

if st.button("Entrenar"):
    try:
        user_params = ast.literal_eval(params_text)
        if not isinstance(user_params, dict):
            raise ValueError("Debe ser un diccionario.")
    except Exception as exc:
        st.error(f"Hiperparámetros inválidos: {exc}")
        st.stop()

    with st.spinner("Entrenando modelo..."):
        try:
            result = train_and_evaluate(
                df=data,
                target_column=target_col,
                model_key=model_key,
                hyperparams=user_params,
                window_size=int(window_size),
                prediction_horizon=int(prediction_horizon),
            )
        except Exception as exc:
            st.error(f"Error durante entrenamiento: {exc}")
            st.stop()

    st.session_state["train_result"] = result
    st.success("Entrenamiento completado")

train_result = st.session_state.get("train_result")
if train_result is not None:
    st.metric("Accuracy", f"{train_result['accuracy']:.4f}")
    st.metric("Ventanas usadas", f"{train_result['num_windows']}")
    st.metric("Tasa positiva", f"{train_result['positive_rate']:.2%}")
    st.text("Classification report")
    st.code(train_result["report"])

    event_cases = train_result.get("event_case_summaries", [])
    if event_cases:
        st.subheader("Casos de test con evento")
        selected_case = st.selectbox(
            "Selecciona un caso para analizar antelación",
            options=event_cases,
            format_func=lambda case: (
                f"Paciente {case['patient_id']} | "
                f"evento={case['event_time']}s | "
                + (
                    f"detención={case['first_detection_time']}s | "
                    f"antelación={case['lead_time_seconds']}s"
                    if case["lead_time_seconds"] is not None
                    else "sin detección previa"
                )
            ),
        )

        st.write(
            "Resumen:",
            {
                "patient_id": selected_case["patient_id"],
                "event_time_s": selected_case["event_time"],
                "first_detection_time_s": selected_case["first_detection_time"],
                "lead_time_s": selected_case["lead_time_seconds"],
            },
        )

        selected_patient = selected_case["patient_id"]
        patient_case_df = data[data["patient_id"].astype(str) == str(selected_patient)].sort_values(
            time_column
        )
        signal_cols = [
            col
            for col in patient_case_df.columns
            if col not in {"patient_id", time_column, target_col}
            and pd.api.types.is_numeric_dtype(patient_case_df[col])
        ]
        if signal_cols:
            st.line_chart(
                patient_case_df.set_index(time_column)[signal_cols],
                use_container_width=True,
            )

        timeline_rows = train_result.get("event_case_timelines", {}).get(str(selected_patient), [])
        if timeline_rows:
            timeline_df = pd.DataFrame(timeline_rows).set_index("time")
            st.line_chart(timeline_df[["pred_positive", "true_event"]], use_container_width=True)
