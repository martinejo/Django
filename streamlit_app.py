"""Interfaz web simple para entrenar modelos de anomalías desde CSV."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import time

import altair as alt
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


def simulate_patient_signals(
    patient_id: int,
    duration_sec: int = 30 * 60,
    fs: int = 100,
    event_prob: float = 0.35,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Simula señales fisiológicas y evento con fase pre-evento."""
    rng = rng or np.random.default_rng()
    n = duration_sec * fs
    t = np.arange(n) / fs

    hr_base = rng.normal(75, 8)
    spo2_base = rng.normal(98, 0.6)
    map_base = rng.normal(80, 10)
    etco2_base = rng.normal(35, 4)

    hr = hr_base + 3 * np.sin(2 * np.pi * t / 90) + rng.normal(0, 1.5, n)
    spo2 = spo2_base + 0.2 * np.sin(2 * np.pi * t / 120) + rng.normal(0, 0.15, n)
    map_ = map_base + 4 * np.sin(2 * np.pi * t / 110) + rng.normal(0, 2.0, n)
    etco2 = etco2_base + 1.5 * np.sin(2 * np.pi * t / 80) + rng.normal(0, 0.8, n)

    n_artifacts = rng.integers(3, 9)
    for _ in range(n_artifacts):
        center = rng.integers(0, n)
        width = rng.integers(3, 12)
        start = max(0, center - width // 2)
        end = min(n, center + width // 2)
        kind = rng.choice(["spo2_drop", "hr_spike", "map_drop", "etco2_spike"])
        if kind == "spo2_drop":
            spo2[start:end] -= rng.uniform(2, 6)
        elif kind == "hr_spike":
            hr[start:end] += rng.uniform(10, 25)
        elif kind == "map_drop":
            map_[start:end] -= rng.uniform(10, 25)
        else:
            etco2[start:end] += rng.uniform(5, 12)

    has_event = rng.random() < event_prob
    event = np.zeros(n, dtype=int)
    if has_event:
        min_event = int(duration_sec / 2 * fs)
        max_event = max(min_event + 1, int((duration_sec - 2 * 60) * fs))
        t_event = int(rng.integers(min_event, max_event))
        event_len = int(rng.integers(10 * fs, 30 * fs))
        event_end = min(n, t_event + event_len)
        event[t_event:event_end] = 1

        pre_len = int(rng.integers(30 * fs, 60 * fs))
        pre_start = max(0, t_event - pre_len)
        pre_t = np.linspace(0, 1, t_event - pre_start, endpoint=False)
        map_[pre_start:t_event] -= 20 * pre_t**1.2 + rng.normal(0, 1.0, t_event - pre_start)
        spo2[pre_start:t_event] -= 3 * pre_t**1.3 + rng.normal(0, 0.1, t_event - pre_start)
        etco2[pre_start:t_event] -= 8 * pre_t**1.1 + rng.normal(0, 0.4, t_event - pre_start)
        hr[pre_start:t_event] += 10 * pre_t - 15 * (pre_t**4) + rng.normal(0, 0.8, t_event - pre_start)
        hr[t_event:event_end] = rng.normal(20, 5, event_end - t_event).clip(0, None)
        spo2[t_event:event_end] = rng.normal(85, 3, event_end - t_event).clip(50, 100)
        map_[t_event:event_end] = rng.normal(40, 6, event_end - t_event).clip(0, None)
        etco2[t_event:event_end] = rng.normal(20, 4, event_end - t_event).clip(0, None)

    hr = hr.clip(0, 220)
    spo2 = spo2.clip(50, 100)
    map_ = map_.clip(0, 180)
    etco2 = etco2.clip(0, 80)

    return pd.DataFrame(
        {
            "patient_id": patient_id,
            "t": np.round(t, 3),
            "fs": fs,
            "HR": hr,
            "SpO2": spo2,
            "MAP": map_,
            "EtCO2": etco2,
            "event": event,
        }
    )


def simulate_dataset(
    n_patients: int,
    duration_sec: int,
    fs: int,
    event_prob: float = 0.35,
    seed: int = 42,
) -> pd.DataFrame:
    """Concatena simulaciones de múltiples pacientes."""
    rng = np.random.default_rng(seed)
    frames = [
        simulate_patient_signals(
            patient_id=pid,
            duration_sec=duration_sec,
            fs=fs,
            event_prob=event_prob,
            rng=rng,
        )
        for pid in range(1, n_patients + 1)
    ]
    return pd.concat(frames, ignore_index=True)

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
    options=["Subir CSV", "Usar CSV de ejemplo", "Simular"],
    horizontal=True,
)

data = None

if source == "Simular":
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sim_duration = int(
            st.number_input("duration_sec", min_value=120, max_value=7200, value=1800, step=60)
        )
    with col2:
        sim_fs = int(st.number_input("fs", min_value=1, max_value=100, value=100, step=1))
    with col3:
        sim_n_patients = int(
            st.number_input("n_pacientes", min_value=5, max_value=500, value=60, step=5)
        )
    with col4:
        sim_event_prob = float(
            st.number_input("event_prob", min_value=0.05, max_value=0.95, value=0.35, step=0.05)
        )

    if st.button("Simular dataset"):
        try:
            sim_df = simulate_dataset(
                n_patients=sim_n_patients,
                duration_sec=sim_duration,
                fs=sim_fs,
                event_prob=sim_event_prob,
            )
            st.session_state["simulated_df"] = sim_df
            st.success(
                f"Simulación creada: {sim_n_patients} pacientes, {sim_duration}s, fs={sim_fs}Hz."
            )
        except Exception as exc:
            st.error(f"No se pudo simular el dataset: {exc}")
            st.stop()

    sim_df = st.session_state.get("simulated_df")
    if sim_df is not None:
        data = sim_df
    else:
        st.info("Configura duration_sec y fs, y pulsa Simular dataset.")
        st.stop()
elif source == "Subir CSV":
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
if "anomaly" in columns:
    default_target_index = columns.index("anomaly")
elif "event" in columns:
    default_target_index = columns.index("event")
else:
    default_target_index = 0
target_col = st.selectbox("Selecciona la columna objetivo", options=columns, index=default_target_index)

st.subheader("Visualización de señales")
time_column = None
if "second" in data.columns:
    time_column = "second"
elif "minute" in data.columns:
    time_column = "minute"
elif "t" in data.columns:
    time_column = "t"

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
            fs_ref = int(patient_df_full["fs"].iloc[0]) if "fs" in patient_df_full.columns else 1
            patient_df = patient_df_full.head(180 * fs_ref)
            st.caption("Paciente sin evento: se muestran hasta 180 segundos iniciales como referencia.")

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
dataset_fs = 1
if "fs" in data.columns:
    fs_values = pd.to_numeric(data["fs"], errors="coerce").dropna()
    if not fs_values.empty:
        dataset_fs = max(1, int(fs_values.iloc[0]))
default_stride = max(5, min(120, dataset_fs))

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
stride_samples = st.number_input(
    "stride_muestras (salto entre ventanas)",
    min_value=1,
    max_value=120,
    value=default_stride,
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

if len(data) > 400_000:
    st.caption(
        "Dataset grande detectado: para acelerar entrenamiento usa "
        "`stride_muestras` igual o mayor que `fs`."
    )

if st.button("Entrenar todos los modelos"):
    all_results: dict[str, dict] = {}
    train_errors: dict[str, str] = {}
    model_keys = list(MODEL_REGISTRY.keys())
    progress = st.progress(0.0)
    status_placeholder = st.empty()
    log_placeholder = st.empty()
    partial_table_placeholder = st.empty()
    logs = deque(maxlen=14)
    partial_rows: list[dict] = []
    global_start = time.perf_counter()

    def push_log(line: str):
        ts = time.strftime("%H:%M:%S")
        logs.append(f"[{ts}] {line}")
        log_placeholder.code("\n".join(logs), language="text")

    for idx, key in enumerate(model_keys, start=1):
        model_name = MODEL_REGISTRY[key].display_name
        model_start = time.perf_counter()
        elapsed_total = time.perf_counter() - global_start
        status_placeholder.info(
            f"Entrenando {model_name} ({idx}/{len(model_keys)}) · "
            f"transcurrido {elapsed_total:.1f}s"
        )
        push_log(f"Inicia entrenamiento de {model_name}.")
        try:
            result = train_and_evaluate(
                df=data,
                target_column=target_col,
                model_key=key,
                hyperparams=dict(MODEL_REGISTRY[key].defaults),
                window_size=int(window_size),
                prediction_horizon=int(prediction_horizon),
                stride=int(stride_samples),
                detection_threshold=float(detection_threshold),
                alarm_min_consecutive=int(alarm_min_consecutive),
                alarm_refractory=int(alarm_refractory),
            )
            all_results[key] = result
            metrics = result.get("patient_level_metrics", {})
            min_early = metrics.get("patient_min_early_detection") or {}
            max_early = metrics.get("patient_max_early_detection") or {}

            partial_rows.append(
                {
                    "Modelo": model_name,
                    "Pacientes detectados": metrics.get("patients_detected", 0),
                    "Detectados antes del evento": metrics.get("patients_detected_pre_event", 0),
                    "Tiempo medio detección temprana (s)": metrics.get("early_lead_time_mean_seconds"),
                    "Paciente menor tiempo detección": (
                        f"{min_early.get('patient_id')} ({min_early.get('lead_time_seconds')}s)"
                        if min_early
                        else "N/A"
                    ),
                    "Paciente mayor tiempo detección": (
                        f"{max_early.get('patient_id')} ({max_early.get('lead_time_seconds')}s)"
                        if max_early
                        else "N/A"
                    ),
                }
            )
            partial_table_placeholder.dataframe(
                pd.DataFrame(partial_rows),
                width="stretch",
                hide_index=True,
            )

            model_obj = result.get("model")
            technical_parts = []
            if hasattr(model_obj, "n_iter_"):
                technical_parts.append(f"n_iter={getattr(model_obj, 'n_iter_')}")
            if hasattr(model_obj, "loss_"):
                try:
                    technical_parts.append(f"loss_final={float(getattr(model_obj, 'loss_')):.6f}")
                except Exception:
                    technical_parts.append("loss_final=disponible")
            if hasattr(model_obj, "loss_curve_"):
                loss_curve = getattr(model_obj, "loss_curve_")
                if isinstance(loss_curve, list) and len(loss_curve) > 0:
                    technical_parts.append(f"loss_curve_pts={len(loss_curve)}")
            if hasattr(model_obj, "n_estimators_"):
                technical_parts.append(f"n_estimators={getattr(model_obj, 'n_estimators_')}")
            if hasattr(model_obj, "feature_importances_"):
                technical_parts.append("feature_importances=disponible")

            model_elapsed = time.perf_counter() - model_start
            if technical_parts:
                push_log(
                    f"{model_name} completado en {model_elapsed:.1f}s | "
                    + " | ".join(technical_parts)
                )
            else:
                push_log(f"{model_name} completado en {model_elapsed:.1f}s.")
        except Exception as exc:
            train_errors[key] = str(exc)
            model_elapsed = time.perf_counter() - model_start
            push_log(f"{model_name} falló en {model_elapsed:.1f}s: {exc}")

        elapsed = time.perf_counter() - global_start
        avg_per_model = elapsed / idx
        remaining_models = len(model_keys) - idx
        eta = avg_per_model * remaining_models
        progress.progress(
            idx / len(model_keys),
            text=f"Completado {idx}/{len(model_keys)} · transcurrido {elapsed:.1f}s · ETA {eta:.1f}s",
        )

    st.session_state["train_results_by_model"] = all_results
    st.session_state["train_errors_by_model"] = train_errors

    total_elapsed = time.perf_counter() - global_start
    status_placeholder.success(
        f"Entrenamiento multi-modelo finalizado en {total_elapsed:.1f}s "
        f"({len(all_results)} OK, {len(train_errors)} con error)."
    )
    push_log("Proceso finalizado.")

    if all_results:
        st.success(f"Entrenamiento completado en {len(all_results)} modelo(s).")
    if train_errors:
        st.warning("Algunos modelos fallaron en entrenamiento.")

train_results_by_model = st.session_state.get("train_results_by_model", {})
train_errors_by_model = st.session_state.get("train_errors_by_model", {})
train_result = None

if train_results_by_model:
    st.subheader("Comparativa de modelos")
    comparison_rows = []
    for key, result in train_results_by_model.items():
        metrics = result.get("patient_level_metrics", {})
        min_early = metrics.get("patient_min_early_detection") or {}
        max_early = metrics.get("patient_max_early_detection") or {}
        comparison_rows.append(
            {
                "Modelo": MODEL_REGISTRY[key].display_name,
                "Pacientes detectados": metrics.get("patients_detected", 0),
                "Detectados antes del evento": metrics.get("patients_detected_pre_event", 0),
                "Tiempo medio detección temprana (s)": metrics.get("early_lead_time_mean_seconds"),
                "Paciente menor tiempo detección": (
                    f"{min_early.get('patient_id')} ({min_early.get('lead_time_seconds')}s)"
                    if min_early
                    else "N/A"
                ),
                "Paciente mayor tiempo detección": (
                    f"{max_early.get('patient_id')} ({max_early.get('lead_time_seconds')}s)"
                    if max_early
                    else "N/A"
                ),
            }
        )
    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        by=["Detectados antes del evento", "Pacientes detectados"],
        ascending=[False, False],
    )
    st.dataframe(comparison_df, width="stretch", hide_index=True)

    selected_model_key = st.selectbox(
        "Modelo para visualizar predicción por paciente",
        options=list(train_results_by_model.keys()),
        format_func=lambda key: MODEL_REGISTRY[key].display_name,
    )
    train_result = train_results_by_model[selected_model_key]
    st.caption(f"Análisis detallado activo: {MODEL_REGISTRY[selected_model_key].display_name}")

if train_errors_by_model:
    error_lines = [
        f"- {MODEL_REGISTRY[key].display_name}: {msg}"
        for key, msg in train_errors_by_model.items()
    ]
    st.caption("Errores de entrenamiento:\n" + "\n".join(error_lines))

if train_result is not None:
    st.metric("Accuracy", f"{train_result['accuracy']:.4f}")
    st.metric("Ventanas usadas", f"{train_result['num_windows']}")
    st.metric("Tasa positiva", f"{train_result['positive_rate']:.2%}")
    st.metric("Umbral detección", f"{train_result.get('detection_threshold', 0.35):.2f}")
    st.metric("Stride", f"{train_result.get('stride', 1)}")
    st.text("Classification report")
    st.code(train_result["report"])

    patient_metrics = train_result.get("patient_level_metrics")
    if patient_metrics:
        st.subheader("Resultados por paciente")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pacientes con evento", f"{patient_metrics.get('patients_with_event', 0)}")
        c2.metric("Pacientes detectados", f"{patient_metrics.get('patients_detected', 0)}")
        sens_value = patient_metrics.get("sensitivity_by_patient")
        c3.metric(
            "Sensibilidad por paciente",
            f"{sens_value:.2%}" if sens_value is not None else "N/A",
        )
        c4.metric(
            "Detectados antes evento",
            f"{patient_metrics.get('patients_detected_pre_event', 0)}",
        )

        st.caption(
            f"Tiempo esperado de predicción (horizonte): "
            f"{patient_metrics.get('expected_prediction_horizon', prediction_horizon)}"
        )

        lead_times = patient_metrics.get("lead_times_seconds", [])
        if lead_times:
            st.write("Tiempos de predicción (segundos):")
            st.code(str(lead_times))
            st.write(
                "Lead time medio/mediano/min/máx (s): "
                f"{patient_metrics.get('lead_time_mean_seconds', 0):.2f} / "
                f"{patient_metrics.get('lead_time_median_seconds', 0):.2f} / "
                f"{patient_metrics.get('lead_time_min_seconds', 0)} / "
                f"{patient_metrics.get('lead_time_max_seconds', 0)}"
            )
            st.write(
                "% detectados con > horizonte/2: "
                f"{patient_metrics.get('pct_detected_gt_half_horizon', 0):.2f}%"
            )
            st.write(
                "% detectados con > horizonte: "
                f"{patient_metrics.get('pct_detected_gt_horizon', 0):.2f}%"
            )
            st.write(
                "Lead time temprano medio (solo anticipaciones positivas): "
                f"{patient_metrics.get('early_lead_time_mean_seconds', 0) or 0:.2f} s"
            )
        else:
            st.write("No hay detecciones con lead time calculable en este entrenamiento.")

        st.write(
            "Falsas alarmas promedio en pacientes sin evento: "
            f"{patient_metrics.get('false_alarms_avg_no_event_patients', 0):.2f}"
        )

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
                margin_before = 120 / 60.0
                margin_after = 30 / 60.0
            else:
                x_pred = pred_times
                x_alarm = alarm_times if len(alarm_times) else np.array([])
                xx = patient_df[time_column].to_numpy(dtype=float)
                xlabel = "Tiempo (segundos)"
                ev_start_x = ev_start_t
                ev_end_x = ev_end_t
                margin_before = 120
                margin_after = 30

            if ev_start_x is not None:
                x_min = max(float(np.min(xx)), ev_start_x - margin_before)
                x_max = min(float(np.max(xx)), ev_start_x + margin_after)
            else:
                x_min = float(np.min(xx))
                x_max = float(np.max(xx))

            threshold = float(train_result.get("detection_threshold", detection_threshold))
            full_prob_df = pd.DataFrame({"x": x_pred, "prob": pred_proba})
            prob_df = full_prob_df[(full_prob_df["x"] >= x_min) & (full_prob_df["x"] <= x_max)]

            signal_df = patient_df[[time_column] + signal_cols].copy()
            signal_df["x"] = xx
            signal_df = signal_df[(signal_df["x"] >= x_min) & (signal_df["x"] <= x_max)]

            # Ajuste fino del dominio X para evitar zonas vacías enormes.
            x_min_candidates = []
            x_max_candidates = []
            if not prob_df.empty:
                x_min_candidates.append(float(prob_df["x"].min()))
                x_max_candidates.append(float(prob_df["x"].max()))
            if not signal_df.empty:
                x_min_candidates.append(float(signal_df["x"].min()))
                x_max_candidates.append(float(signal_df["x"].max()))
            if x_min_candidates and x_max_candidates:
                x_min = min(x_min_candidates)
                x_max = max(x_max_candidates)

            # Fallback robusto: si la curva de probabilidad queda vacía o con un único punto,
            # mostramos los puntos más cercanos al evento para evitar gráfica "en blanco".
            if len(prob_df) < 2 and not full_prob_df.empty:
                if ev_start_x is not None:
                    nearest_idx = np.abs(full_prob_df["x"].to_numpy(dtype=float) - float(ev_start_x)).argsort()
                    keep_n = min(240, len(nearest_idx))
                    selected = np.sort(nearest_idx[:keep_n])
                    prob_df = full_prob_df.iloc[selected].sort_values("x")
                else:
                    prob_df = full_prob_df.copy()
                if not prob_df.empty:
                    x_min = min(float(signal_df["x"].min()) if not signal_df.empty else float(prob_df["x"].min()), float(prob_df["x"].min()))
                    x_max = max(float(signal_df["x"].max()) if not signal_df.empty else float(prob_df["x"].max()), float(prob_df["x"].max()))

            event_span_df = pd.DataFrame(columns=["x", "x2"])
            if ev_start_x is not None and ev_end_x is not None:
                ev_left = max(x_min, float(ev_start_x))
                ev_right = min(x_max, float(ev_end_x))
                if ev_left < ev_right:
                    event_span_df = pd.DataFrame([{"x": ev_left, "x2": ev_right}])

            alarm_df = pd.DataFrame({"x": x_alarm, "y": threshold})
            if not alarm_df.empty:
                alarm_df = alarm_df[(alarm_df["x"] >= x_min) & (alarm_df["x"] <= x_max)]

            prob_base = alt.Chart(prob_df).encode(
                x=alt.X("x:Q", title=xlabel, scale=alt.Scale(domain=[x_min, x_max]))
            )
            prob_area = prob_base.mark_area(opacity=0.22, color="#6aa7ff").encode(
                y=alt.Y("prob:Q", title="Probabilidad", scale=alt.Scale(domain=[0, 1]))
            )
            prob_line = prob_base.mark_line(color="#6aa7ff", strokeWidth=2).encode(
                y=alt.Y("prob:Q", title="Probabilidad", scale=alt.Scale(domain=[0, 1]))
            )
            prob_points = prob_base.mark_point(color="#6aa7ff", filled=True, size=38).encode(
                y=alt.Y("prob:Q", title="Probabilidad", scale=alt.Scale(domain=[0, 1]))
            )
            threshold_rule = alt.Chart(pd.DataFrame({"thr": [threshold]})).mark_rule(
                color="#f59e0b", strokeDash=[8, 6], strokeWidth=2
            ).encode(y=alt.Y("thr:Q", scale=alt.Scale(domain=[0, 1])))
            event_rect = alt.Chart(event_span_df).mark_rect(color="#2563eb", opacity=0.22).encode(
                x=alt.X("x:Q", scale=alt.Scale(domain=[x_min, x_max]), axis=None),
                x2="x2:Q",
            )
            alarm_lines = alt.Chart(alarm_df).mark_rule(color="#22c55e", opacity=0.28).encode(x="x:Q")
            alarm_points = alt.Chart(alarm_df).mark_point(
                color="#22c55e", size=95, shape="triangle-up", filled=True
            ).encode(x="x:Q", y="y:Q")
            prob_chart = (
                event_rect + prob_area + prob_line + prob_points + threshold_rule + alarm_lines + alarm_points
            ).properties(height=300, title="Streaming: probabilidad y alarmas").resolve_scale(x="shared")
            st.altair_chart(prob_chart, use_container_width=True)

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
            st.caption(" | ".join(metrics_lines))

            signal_palette = {
                "hr": "#60a5fa",
                "spo2": "#f59e0b",
                "map": "#34d399",
                "etco2": "#f87171",
                "HR": "#60a5fa",
                "SpO2": "#f59e0b",
                "MAP": "#34d399",
                "EtCO2": "#f87171",
            }
            signal_long = signal_df.melt(
                id_vars=["x"], value_vars=signal_cols, var_name="signal", value_name="value"
            )
            color_range = [signal_palette.get(col, "#9fe9ff") for col in signal_cols]
            signal_line = alt.Chart(signal_long).mark_line(strokeWidth=2).encode(
                x=alt.X("x:Q", title=xlabel, scale=alt.Scale(domain=[x_min, x_max])),
                y=alt.Y("value:Q", title="Valor (escalas distintas)"),
                color=alt.Color(
                    "signal:N",
                    scale=alt.Scale(domain=signal_cols, range=color_range),
                    title="Señales",
                ),
            )
            signal_event_rect = alt.Chart(event_span_df).mark_rect(color="#2563eb", opacity=0.22).encode(
                x=alt.X("x:Q", scale=alt.Scale(domain=[x_min, x_max]), axis=None),
                x2="x2:Q",
            )
            signal_chart = (signal_event_rect + signal_line).properties(
                height=320, title="Señales del paciente (inspección visual)"
            ).resolve_scale(x="shared")
            st.altair_chart(signal_chart, use_container_width=True)
