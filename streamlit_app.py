"""Interfaz web simple para entrenar modelos de anomalías desde CSV."""

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.anomaly_web.config import MODEL_REGISTRY
from src.anomaly_web.training import read_csv, train_and_evaluate

EXAMPLES_DIR = Path(__file__).parent / "data" / "examples"
BANNER_PATH = Path(__file__).parent / "Gemini_Generated_Image_vjzlxxvjzlxxvjzl.png"
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

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #03131c 0%, #062433 100%);
    }
    .stApp, .stApp p, .stApp span, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3 {
        color: #9fe9ff !important;
        letter-spacing: 0.2px;
    }
    .stMarkdown, .stCaption {
        color: #9fe9ff !important;
    }
    [data-testid="stSidebar"] * {
        color: #9fe9ff !important;
    }
    div[data-testid="stMetric"] {
        background: rgba(10, 33, 46, 0.85);
        border: 1px solid #2b6d84;
        border-radius: 12px;
        padding: 10px 14px;
    }
    [data-baseweb="select"] > div,
    .stTextInput > div > div > input,
    .stTextArea textarea,
    .stNumberInput input {
        background: rgba(6, 27, 39, 0.95) !important;
        color: #b6f0ff !important;
        border-color: #2b6d84 !important;
    }
    .stButton > button {
        background: #0f766e;
        color: #ffffff;
        border-radius: 8px;
        border: none;
    }
    .stButton > button:hover {
        background: #0b5f58;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if BANNER_PATH.exists():
    st.image(str(BANNER_PATH), width="stretch")

st.title("Predicción temprana de anomalías")
st.caption("Panel clínico de soporte para detección anticipada de eventos perioperatorios.")

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
st.dataframe(data.head(20), width="stretch")

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
    anomaly_values = pd.to_numeric(data[anomaly_column], errors="coerce").fillna(0)
    patient_has_anomaly = anomaly_values.groupby(data["patient_id"], dropna=False).max().gt(0)
    patient_summary = (
        patient_has_anomaly.rename("has_anomaly")
        .reset_index()
    )

    patient_options = patient_summary["patient_id"].tolist()
    anomaly_map = dict(
        zip(patient_summary["patient_id"].tolist(), patient_summary["has_anomaly"].astype(bool).tolist())
    )
    selected_patient = st.selectbox(
        "Paciente para visualizar",
        options=patient_options,
        format_func=lambda pid: (
            f"Paciente {pid} {'• ALERTA' if anomaly_map.get(pid, False) else '• estable'}"
        ),
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
        patient_df_full = (
            data[data["patient_id"] == selected_patient]
            .sort_values(time_column)
        )
        event_times = patient_df_full.loc[patient_df_full[target_col] == 1, time_column].to_numpy()
        if len(event_times) > 0:
            event_time = int(event_times[0])
            start_t = event_time - 120
            end_t = event_time + 60
            patient_df = patient_df_full[
                (patient_df_full[time_column] >= start_t) & (patient_df_full[time_column] <= end_t)
            ]
            st.caption(
                f"Ventana mostrada: de {start_t}s a {end_t}s respecto al evento en {event_time}s."
            )
        else:
            patient_df = patient_df_full.head(180)
            st.caption("Paciente sin evento: se muestran 180 segundos iniciales como referencia.")

        st.line_chart(
            patient_df.set_index(time_column)[selected_signals],
            width="stretch",
            height=460,
        )
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
detection_threshold = st.number_input(
    "umbral_deteccion (0-1, menor = más sensible)",
    min_value=0.05,
    max_value=0.95,
    value=0.35,
    step=0.05,
)
alarm_min_consecutive = st.number_input(
    "alarmas_consecutivas_min",
    min_value=1,
    max_value=10,
    value=2,
    step=1,
)
alarm_refractory = st.number_input(
    "refractory_segundos",
    min_value=0,
    max_value=120,
    value=10,
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
                detection_threshold=float(detection_threshold),
                alarm_min_consecutive=int(alarm_min_consecutive),
                alarm_refractory=int(alarm_refractory),
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
    st.metric("Umbral detección", f"{train_result.get('detection_threshold', 0.35):.2f}")
    st.text("Classification report")
    st.code(train_result["report"])

    test_cases = train_result.get("test_patient_summaries", [])
    test_payloads = train_result.get("test_patient_payloads", {})
    if test_cases:
        st.subheader("Análisis final (solo pacientes del set test)")
        selected_case = st.selectbox(
            "Paciente (test set)",
            options=test_cases,
            format_func=lambda case: (
                f"Paciente {case['patient_id']} | "
                f"{'con evento' if case['has_event'] else 'sin evento'} | "
                + (
                    f"lead={case['lead_time_seconds']}s"
                    if case["lead_time_seconds"] is not None
                    else "sin detección previa"
                )
            ),
        )

        selected_patient = selected_case["patient_id"]
        payload = test_payloads.get(str(selected_patient), {})
        pred_times = np.asarray(payload.get("pred_times", []), dtype=float)
        pred_proba = np.asarray(payload.get("pred_proba", []), dtype=float)
        alarm_times = np.asarray(payload.get("alarm_times", []), dtype=float)
        ev_start_t = payload.get("event_start_time")
        ev_end_t = payload.get("event_end_time")

        patient_df = data[data["patient_id"].astype(str) == str(selected_patient)].sort_values(time_column)
        preferred_signals = ["hr", "spo2", "map", "etco2", "HR", "SpO2", "MAP", "EtCO2"]
        signal_cols = [col for col in preferred_signals if col in patient_df.columns]
        if not signal_cols:
            signal_cols = [
                col
                for col in patient_df.columns
                if col not in {"patient_id", time_column, target_col}
                and pd.api.types.is_numeric_dtype(patient_df[col])
            ][:4]

        if len(pred_times) > 0 and len(signal_cols) > 0:
            use_minutes = True
            if use_minutes:
                x_pred = pred_times / 60.0
                x_alarm = alarm_times / 60.0 if len(alarm_times) else np.array([])
                xx = patient_df[time_column].to_numpy(dtype=float) / 60.0
                xlabel = "Tiempo (minutos)"
                ev_start_x = None if ev_start_t is None else ev_start_t / 60.0
                ev_end_x = None if ev_end_t is None else ev_end_t / 60.0
                margin_before = 180 / 60.0
                margin_after = 180 / 60.0
            else:
                x_pred = pred_times
                x_alarm = alarm_times if len(alarm_times) else np.array([])
                xx = patient_df[time_column].to_numpy(dtype=float)
                xlabel = "Tiempo (segundos)"
                ev_start_x = ev_start_t
                ev_end_x = ev_end_t
                margin_before = 180
                margin_after = 180

            if ev_start_x is not None:
                x_min = max(float(np.min(x_pred)), ev_start_x - margin_before)
                x_max = min(float(np.max(x_pred)), ev_end_x + margin_after)
            else:
                x_min = float(np.min(x_pred))
                x_max = float(np.max(x_pred))

            threshold = float(train_result.get("detection_threshold", detection_threshold))
            fig = plt.figure(figsize=(12, 7))

            ax1 = fig.add_axes([0.08, 0.56, 0.88, 0.36])
            ax1.plot(x_pred, pred_proba, color="tab:blue", linewidth=2, marker="x", label="P(evento en horizonte)")
            ax1.fill_between(x_pred, pred_proba, alpha=0.2, color="tab:blue")
            ax1.axhline(
                threshold,
                linestyle="--",
                linewidth=2,
                color="tab:orange",
                label=f"Threshold={threshold:.2f}",
            )
            if ev_start_x is not None:
                ax1.axvspan(ev_start_x, ev_end_x, alpha=0.15, color="tab:red", label="Evento real")
            if len(x_alarm):
                ax1.scatter(x_alarm, np.full_like(x_alarm, threshold), marker="^", s=80, color="tab:green", label="Alarma")
                for xa in x_alarm:
                    ax1.axvline(xa, alpha=0.15, color="tab:green", linewidth=2)
            ax1.set_title("Streaming: probabilidad y alarmas", fontsize=13)
            ax1.set_xlabel(xlabel)
            ax1.set_ylabel("Probabilidad")
            ax1.set_ylim(0, 1)
            ax1.grid(alpha=0.3)
            ax1.legend(loc="upper left")

            metrics_lines = []
            if ev_start_x is not None:
                metrics_lines.append(f"Evento real: sí (inicio={ev_start_x:.2f}min)")
            else:
                metrics_lines.append("Evento real: no")
            if selected_case.get("first_alarm_time") is not None:
                fa = selected_case["first_alarm_time"] / 60.0
                metrics_lines.append(f"Primera alarma: {fa:.2f}min")
            else:
                metrics_lines.append("Primera alarma: (no hubo)")
            if selected_case.get("lead_time_seconds") is not None:
                lt = selected_case["lead_time_seconds"] / 60.0
                metrics_lines.append(f"Anticipación (lead time): {lt:.2f}min")
            metrics_lines.append(f"Falsas alarmas: {selected_case.get('false_alarm_count', 0)}")
            ax1.text(0.99, 0.02, "\n".join(metrics_lines), transform=ax1.transAxes, ha="right", va="bottom", fontsize=10)

            ax2 = fig.add_axes([0.08, 0.10, 0.88, 0.36])
            for col in signal_cols:
                ax2.plot(xx, patient_df[col].to_numpy(), label=col, linewidth=1.5)
            if ev_start_x is not None:
                ax2.axvspan(ev_start_x, ev_end_x, alpha=0.15, color="tab:red")
            ax2.set_title("Señales del paciente (inspección visual)", fontsize=13)
            ax2.set_xlabel(xlabel)
            ax2.set_ylabel("Valor (escalas distintas)")
            ax2.grid(alpha=0.3)
            ax2.legend(loc="upper left", ncol=4)

            ax1.set_xlim(x_min, x_max)
            ax2.set_xlim(x_min, x_max)
            st.pyplot(fig, clear_figure=True)
