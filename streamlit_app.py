"""Interfaz web simple para entrenar modelos de anomalías desde CSV."""

import ast
from pathlib import Path

import altair as alt
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
            event_time = int(selected_case["event_time"])
            start_t = event_time - 120
            end_t = event_time + 60
            patient_case_window = patient_case_df[
                (patient_case_df[time_column] >= start_t) & (patient_case_df[time_column] <= end_t)
            ]
            signal_long = patient_case_window[[time_column] + signal_cols].melt(
                id_vars=[time_column], var_name="signal", value_name="value"
            )

            bands = []
            detection_time = selected_case["first_detection_time"]
            if detection_time is not None:
                win_size = int(train_result.get("window_size", int(window_size)))
                detection_start = max(start_t, int(detection_time) - win_size + 1)
                detection_end = min(end_t, int(detection_time))
                if detection_start <= detection_end:
                    bands.append(
                        {
                            "x_start": detection_start,
                            "x_end": detection_end,
                            "label": "Ventana detectada",
                            "color": "#6aa7ff",
                            "opacity": 0.18,
                        }
                    )
            bands.append(
                {
                    "x_start": max(start_t, event_time - 0.5),
                    "x_end": min(end_t, event_time + 0.5),
                    "label": "Evento",
                    "color": "#2563eb",
                    "opacity": 0.38,
                }
            )

            base = alt.Chart(signal_long).encode(
                x=alt.X(f"{time_column}:Q", title="Tiempo (s)"),
                y=alt.Y("value:Q", title="Valor señal"),
                color=alt.Color("signal:N", title="Señal"),
            )
            line_layer = base.mark_line(strokeWidth=2)

            chart = line_layer
            if bands:
                band_df = pd.DataFrame(bands)
                band_layer = alt.Chart(band_df).mark_rect().encode(
                    x="x_start:Q",
                    x2="x_end:Q",
                    color=alt.Color(
                        "label:N",
                        scale=alt.Scale(
                            domain=["Ventana detectada", "Evento"],
                            range=["#6aa7ff", "#2563eb"],
                        ),
                        legend=alt.Legend(title="Sombras"),
                    ),
                    opacity=alt.Opacity("opacity:Q", legend=None),
                )
                chart = band_layer + line_layer

            st.altair_chart(chart.properties(height=460), use_container_width=True)

        timeline_rows = train_result.get("event_case_timelines", {}).get(str(selected_patient), [])
        if timeline_rows:
            timeline_df = pd.DataFrame(timeline_rows)
            event_time = int(selected_case["event_time"])
            start_t = event_time - 120
            end_t = event_time + 60
            timeline_df = timeline_df[
                (timeline_df["time"] >= start_t) & (timeline_df["time"] <= end_t)
            ].set_index("time")
            tl_reset = timeline_df.reset_index()
            tl_long = tl_reset.melt(
                id_vars=["time"], value_vars=["pred_positive", "true_event"], var_name="series", value_name="value"
            )

            tl_bands = []
            detection_time = selected_case["first_detection_time"]
            if detection_time is not None:
                win_size = int(train_result.get("window_size", int(window_size)))
                detection_start = max(start_t, int(detection_time) - win_size + 1)
                detection_end = min(end_t, int(detection_time))
                if detection_start <= detection_end:
                    tl_bands.append(
                        {
                            "x_start": detection_start,
                            "x_end": detection_end,
                            "label": "Ventana detectada",
                            "color": "#6aa7ff",
                            "opacity": 0.2,
                        }
                    )
            tl_bands.append(
                {
                    "x_start": max(start_t, event_time - 0.5),
                    "x_end": min(end_t, event_time + 0.5),
                    "label": "Evento",
                    "color": "#2563eb",
                    "opacity": 0.4,
                }
            )

            tl_line = alt.Chart(tl_long).mark_line(strokeWidth=2).encode(
                x=alt.X("time:Q", title="Tiempo (s)"),
                y=alt.Y("value:Q", title="Predicción / Evento"),
                color=alt.Color("series:N", title="Serie"),
            )
            tl_band = alt.Chart(pd.DataFrame(tl_bands)).mark_rect().encode(
                x="x_start:Q",
                x2="x_end:Q",
                color=alt.Color(
                    "label:N",
                    scale=alt.Scale(
                        domain=["Ventana detectada", "Evento"],
                        range=["#6aa7ff", "#2563eb"],
                    ),
                    legend=None,
                ),
                opacity=alt.Opacity("opacity:Q", legend=None),
            )
            st.altair_chart((tl_band + tl_line).properties(height=260), use_container_width=True)
