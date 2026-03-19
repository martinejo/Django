"""Interfaz web simple para entrenar modelos de anomalías desde CSV."""

from __future__ import annotations

import ast
import base64
from collections import deque
from pathlib import Path
import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.anomaly_web.config import MODEL_REGISTRY
from src.anomaly_web.training import read_csv, train_and_evaluate
from src.anomaly_web.user_store import (
    authenticate_user,
    list_user_datasets,
    list_user_models,
    load_model_artifact,
    register_user,
    save_user_dataset,
    save_user_dataset_named,
    save_user_model_artifact_named,
)

EXAMPLES_DIR = Path(__file__).parent / "data" / "examples"
BANNER_PATH = Path(__file__).parent / "Gemini_Generated_Image_vjzlxxvjzlxxvjzl.png"
LOGIN_BG_PATH = Path(__file__).parent / "56cba10a-428e-410d-ae50-7dc9d011b620.jpg"
LOGIN_BG_URL = (
    "https://raw.githubusercontent.com/martinejo/Django/"
    "codex/add-web-project-for-anomaly-prediction/56cba10a-428e-410d-ae50-7dc9d011b620.jpg"
)
APP_ROOT = Path(__file__).parent
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


def forecast_with_saved_model(
    model,
    df: pd.DataFrame,
    signal_columns: list[str],
    time_column: str,
    window_size_samples: int,
    stride_samples: int,
    threshold: float,
) -> pd.DataFrame:
    """Predicción streaming por paciente usando un modelo ya entrenado."""
    if "patient_id" not in df.columns:
        raise ValueError("El CSV para pronóstico debe tener columna 'patient_id'.")
    if time_column not in df.columns:
        raise ValueError(f"El CSV para pronóstico debe tener columna temporal '{time_column}'.")
    missing = [c for c in signal_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan señales requeridas por el modelo: {missing}")

    rows: list[dict] = []
    for pid, g in df.groupby("patient_id"):
        g = g.sort_values(time_column).reset_index(drop=True)
        if len(g) < window_size_samples:
            rows.append(
                {
                    "patient_id": pid,
                    "n_ventanas": 0,
                    "first_alarm_time": None,
                    "max_probability": None,
                }
            )
            continue

        x_signal = g[signal_columns].to_numpy(dtype=np.float32, copy=False)
        t_values = g[time_column].to_numpy(dtype=float)
        all_windows = np.lib.stride_tricks.sliding_window_view(
            x_signal, window_shape=window_size_samples, axis=0
        ).reshape(-1, window_size_samples * len(signal_columns))
        row_idx = np.arange(0, len(all_windows), max(1, stride_samples))
        x_windows = all_windows[row_idx]
        pred_times = t_values[row_idx + window_size_samples - 1]

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x_windows)[:, 1]
        else:
            probs = model.predict(x_windows).astype(float)

        alarm_times = pred_times[probs >= threshold]
        first_alarm = float(alarm_times[0]) if len(alarm_times) > 0 else None
        rows.append(
            {
                "patient_id": pid,
                "n_ventanas": int(len(x_windows)),
                "first_alarm_time": first_alarm,
                "max_probability": float(np.max(probs)) if len(probs) > 0 else None,
            }
        )

    return pd.DataFrame(rows)


def forecast_with_saved_model_details(
    model,
    df: pd.DataFrame,
    signal_columns: list[str],
    time_column: str,
    window_size_samples: int,
    stride_samples: int,
    threshold: float,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Pronóstico por paciente + payload detallado para visualización."""
    if "patient_id" not in df.columns:
        raise ValueError("El CSV para pronóstico debe tener columna 'patient_id'.")
    if time_column not in df.columns:
        raise ValueError(f"El CSV para pronóstico debe tener columna temporal '{time_column}'.")
    missing = [c for c in signal_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan señales requeridas por el modelo: {missing}")

    rows: list[dict] = []
    payloads: dict[str, dict] = {}

    for pid, g in df.groupby("patient_id"):
        g = g.sort_values(time_column).reset_index(drop=True)
        has_event = "event" in g.columns and pd.to_numeric(g["event"], errors="coerce").fillna(0).max() > 0
        event_start = None
        event_end = None
        if has_event:
            ev = pd.to_numeric(g["event"], errors="coerce").fillna(0)
            ev_times = g.loc[ev > 0, time_column].to_numpy(dtype=float)
            if len(ev_times) > 0:
                event_start = float(ev_times[0])
                event_end = float(ev_times[-1])

        if len(g) < window_size_samples:
            rows.append(
                {
                    "patient_id": pid,
                    "n_ventanas": 0,
                    "first_alarm_time": None,
                    "max_probability": None,
                    "has_event": bool(has_event),
                    "lead_time_s": None,
                }
            )
            payloads[str(pid)] = {
                "pred_times": [],
                "pred_proba": [],
                "alarm_times": [],
                "event_start_time": event_start,
                "event_end_time": event_end,
            }
            continue

        x_signal = g[signal_columns].to_numpy(dtype=np.float32, copy=False)
        t_values = g[time_column].to_numpy(dtype=float)
        all_windows = np.lib.stride_tricks.sliding_window_view(
            x_signal, window_shape=window_size_samples, axis=0
        ).reshape(-1, window_size_samples * len(signal_columns))
        row_idx = np.arange(0, len(all_windows), max(1, stride_samples))
        x_windows = all_windows[row_idx]
        pred_times = t_values[row_idx + window_size_samples - 1]

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x_windows)[:, 1]
        else:
            probs = model.predict(x_windows).astype(float)

        alarm_times = pred_times[probs >= threshold]
        first_alarm = float(alarm_times[0]) if len(alarm_times) > 0 else None
        lead_time = None
        if has_event and event_start is not None and first_alarm is not None:
            lead_time = float(event_start - first_alarm)

        rows.append(
            {
                "patient_id": pid,
                "n_ventanas": int(len(x_windows)),
                "first_alarm_time": first_alarm,
                "max_probability": float(np.max(probs)) if len(probs) > 0 else None,
                "has_event": bool(has_event),
                "lead_time_s": lead_time,
            }
        )
        payloads[str(pid)] = {
            "pred_times": [float(v) for v in pred_times.tolist()],
            "pred_proba": [float(v) for v in probs.tolist()],
            "alarm_times": [float(v) for v in alarm_times.tolist()],
            "event_start_time": event_start,
            "event_end_time": event_end,
        }

    return pd.DataFrame(rows), payloads


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

if "auth_user" not in st.session_state:
    st.session_state["auth_user"] = None

if st.session_state["auth_user"] is None:
    bg_source = LOGIN_BG_URL
    if LOGIN_BG_PATH.exists():
        encoded = base64.b64encode(LOGIN_BG_PATH.read_bytes()).decode("ascii")
        bg_source = f"data:image/jpeg;base64,{encoded}"

    login_css = f"""
        <style>
        .stApp {{
            background:
                linear-gradient(135deg, rgba(10,16,24,0.74), rgba(6,10,16,0.84)),
                url('{bg_source}') center center / cover no-repeat !important;
            min-height: 100vh;
        }}
        [data-testid="stSidebar"] {{ display: none !important; }}
        .block-container {{
            max-width: 1180px;
            padding-top: 8vh;
            padding-bottom: 8vh;
            position: relative;
            min-height: 100vh;
            box-sizing: border-box;
        }}
        .block-container::before {{
            content: "";
            position: absolute;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
            width: min(520px, 92vw);
            height: min(840px, 82vh);
            background: linear-gradient(165deg, rgba(8,49,89,0.58), rgba(4,29,58,0.58));
            border: 1.5px solid rgba(114, 202, 255, 0.65);
            border-radius: 24px;
            box-shadow: 0 24px 56px rgba(0,0,0,0.42);
            backdrop-filter: blur(1.5px);
            z-index: 0;
        }}
        .block-container > div {{
            position: relative;
            z-index: 1;
        }}
        .login-title {{
            text-align: center;
            color: #edf2f8 !important;
            letter-spacing: 0.24rem;
            font-size: 2.0rem;
            margin-bottom: 1.3rem;
            font-weight: 500;
        }}
        div[data-testid="stForm"] {{
            background: linear-gradient(160deg, rgba(9,54,95,0.97), rgba(5,33,62,0.97));
            border: 1px solid rgba(130, 202, 255, 0.35);
            border-radius: 18px;
            padding: 22px 20px 14px 20px;
            box-shadow: 0 24px 48px rgba(0,0,0,0.42);
        }}
        div[data-testid="stForm"] label,
        div[data-testid="stForm"] .stMarkdown,
        div[data-testid="stForm"] p {{
            color: #e4f4ff !important;
        }}
        div[data-testid="stForm"] .stTextInput input {{
            background: #ffffff !important;
            color: #0f172a !important;
            border: 1px solid #78b7e9 !important;
            border-radius: 10px !important;
            caret-color: #000000 !important;
        }}
        div[data-testid="stForm"] .stTextInput div[data-baseweb="input"] {{
            border-radius: 10px !important;
            overflow: hidden !important;
            align-items: stretch !important;
        }}
        div[data-testid="stForm"] .stTextInput button {{ display: none !important; }}
        div[data-testid="stForm"] button {{
            width: 100%;
            background: #2a90e3 !important;
            color: #ffffff !important;
            border-radius: 10px !important;
            border: none !important;
            font-weight: 600 !important;
        }}
        div[data-testid="stForm"] button:hover {{ background: #1b7bcc !important; }}
        .login-note {{
            text-align: center;
            color: #e7edf7 !important;
            margin-bottom: 0.8rem;
            font-size: 1.08rem;
        }}
        .login-badge {{
            text-align: center;
            color: #bfe7ff !important;
            margin-top: 0.45rem;
            margin-bottom: 0.6rem;
            font-size: 0.95rem;
        }}
        </style>
        """
    st.markdown(login_css, unsafe_allow_html=True)
    st.markdown("<div class='login-title'>ESTIMAMODELOS LOGIN</div>", unsafe_allow_html=True)
    col_l, col_c, col_r = st.columns([1.25, 1.1, 1.25])
    with col_c:
        if BANNER_PATH.exists():
            st.image(str(BANNER_PATH), width="stretch")
        st.markdown("<div class='login-badge'>Inteligencia artificial clínica para detección temprana</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='login-note'>Inicia sesión o crea tu cuenta para acceder a tu panel personal</div>",
            unsafe_allow_html=True,
        )
        auth_mode = st.radio(
            "Modo de acceso",
            ["Iniciar sesión", "Registrarse"],
            horizontal=True,
            label_visibility="collapsed",
        )
        with st.form("auth_form", clear_on_submit=False):
            login_user = st.text_input("Usuario", key="login_user")
            login_pass = st.text_input("Contraseña", type="password", key="login_pass")
            login_pass2 = None
            if auth_mode == "Registrarse":
                login_pass2 = st.text_input(
                    "Repite contraseña", type="password", key="login_pass_repeat"
                )
            submit = st.form_submit_button(
                "Entrar" if auth_mode == "Iniciar sesión" else "Crear cuenta"
            )

        if submit and auth_mode == "Iniciar sesión":
            if authenticate_user(APP_ROOT, login_user, login_pass):
                st.session_state["auth_user"] = login_user.strip()
                st.success("Sesión iniciada.")
                st.rerun()
            else:
                st.error("Usuario o contraseña incorrectos.")
        if submit and auth_mode == "Registrarse":
            if login_pass2 != login_pass:
                st.error("Las contraseñas no coinciden.")
            else:
                ok, msg = register_user(APP_ROOT, login_user, login_pass)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
    st.stop()

if BANNER_PATH.exists():
    st.image(str(BANNER_PATH), width="stretch")

st.title("Predicción temprana de anomalías")
st.caption("Panel clínico de soporte para detección anticipada de eventos perioperatorios.")

active_user = st.session_state["auth_user"]
st.sidebar.markdown(f"**Usuario activo:** `{active_user}`")


def clear_working_state() -> None:
    """Limpia estado de trabajo para arrancar desde cero con un nuevo dataset."""
    keys_to_clear = [
        "train_results_by_model",
        "train_errors_by_model",
        "simulated_df",
        "selected_example_file",
        "selected_example_name",
        "last_dataset_path",
        "last_saved_dataset_key",
        "forecast_file",
    ]
    for k in keys_to_clear:
        st.session_state.pop(k, None)


if st.sidebar.button("Cerrar sesión"):
    st.session_state["auth_user"] = None
    clear_working_state()
    st.session_state.pop("selected_user_dataset_path", None)
    st.rerun()

user_datasets = list_user_datasets(APP_ROOT, active_user)
st.sidebar.markdown("**Cargar dataset guardado**")
if user_datasets:
    selected_dataset_sidebar = st.sidebar.selectbox(
        "Tus datasets",
        options=user_datasets,
        format_func=lambda item: item["name"],
        key="sidebar_dataset_select",
    )
    if st.sidebar.button("Usar este dataset", key="btn_use_saved_dataset"):
        st.session_state["selected_user_dataset_path"] = selected_dataset_sidebar["path"]
        st.session_state["selected_user_dataset_name"] = selected_dataset_sidebar["name"]
        clear_working_state()
        st.session_state["selected_user_dataset_path"] = selected_dataset_sidebar["path"]
        st.session_state["selected_user_dataset_name"] = selected_dataset_sidebar["name"]
        st.success(f"Dataset seleccionado: {selected_dataset_sidebar['name']}")
        st.rerun()
else:
    st.sidebar.caption("No tienes datasets guardados todavía.")

source = st.radio(
    "Fuente de datos",
    options=["Mis datasets guardados", "Subir CSV", "Usar CSV de ejemplo", "Simular"],
    horizontal=True,
)

data = None
input_fs_csv = None

if source in {"Subir CSV", "Usar CSV de ejemplo"}:
    input_fs_csv = float(
        st.number_input(
            "Frecuencia de muestreo del CSV (fs, muestras/segundo)",
            min_value=0.1,
            max_value=500.0,
            value=1.0,
            step=0.1,
            help="Indica cuántas muestras por segundo contiene el CSV.",
        )
    )

if source == "Mis datasets guardados":
    selected_dataset_path = st.session_state.get("selected_user_dataset_path")
    selected_dataset_name = st.session_state.get("selected_user_dataset_name", "")
    if selected_dataset_path is None:
        st.info("Selecciona un dataset en la barra lateral y pulsa 'Usar este dataset'.")
        st.stop()
    try:
        data = read_csv(selected_dataset_path)
        if "fs" not in data.columns:
            data["fs"] = 1.0
        st.success(f"Dataset cargado desde tu almacenamiento: {selected_dataset_name}")
    except Exception as exc:
        st.error(f"No se pudo cargar el dataset guardado: {exc}")
        st.stop()
elif source == "Simular":
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
            st.info("Dataset simulado listo. Puedes guardarlo manualmente con un nombre.")
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
        data["fs"] = float(input_fs_csv)
        save_key = f"upload:{uploaded_file.name}:{len(data)}:{input_fs_csv}"
        if st.session_state.get("last_saved_dataset_key") != save_key:
            saved_path = save_user_dataset(APP_ROOT, active_user, data, source="subido")
            st.session_state["last_dataset_path"] = str(saved_path)
            st.session_state["last_saved_dataset_key"] = save_key
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
            data["fs"] = float(input_fs_csv)
            save_key = f"example:{selected_file}:{len(data)}:{input_fs_csv}"
            if st.session_state.get("last_saved_dataset_key") != save_key:
                saved_path = save_user_dataset(
                    APP_ROOT, active_user, data, source=f"ejemplo_{selected_file}"
                )
                st.session_state["last_dataset_path"] = str(saved_path)
                st.session_state["last_saved_dataset_key"] = save_key
            st.success(f"CSV cargado: {selected_name}")
        except Exception as exc:
            st.error(f"No se pudo cargar el CSV de ejemplo: {exc}")
            st.stop()
    else:
        st.info("Selecciona un CSV de ejemplo y pulsa el botón de carga.")
        st.stop()

if source == "Simular" and data is not None:
    st.subheader("Guardar dataset simulado")
    sim_dataset_name = st.text_input(
        "Nombre del dataset simulado",
        value=st.session_state.get("sim_dataset_name", "simulacion_anestesia"),
        key="sim_dataset_name",
    )
    if st.button("Guardar dataset simulado", key="btn_save_sim_dataset"):
        try:
            saved_path = save_user_dataset_named(APP_ROOT, active_user, data, sim_dataset_name)
            st.session_state["last_dataset_path"] = str(saved_path)
            st.success(f"Dataset guardado como: {saved_path.name}")
        except Exception as exc:
            st.error(f"No se pudo guardar el dataset simulado: {exc}")


st.subheader("Vista rápida del dataset")
st.dataframe(data.head(20), width="stretch")
with st.sidebar.expander("Mis datasets", expanded=False):
    if user_datasets:
        for item in user_datasets[:20]:
            st.caption(f"{item['name']} · {item['size_kb']} KB")
    else:
        st.caption("Aún no tienes datasets guardados.")

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
        dataset_fs = max(0.1, float(fs_values.iloc[0]))
default_stride = max(5, min(120, dataset_fs))

window_size_seconds = st.number_input(
    "tam_ventana (segundos por ventana)",
    min_value=1.0,
    max_value=600.0,
    value=5.0,
    step=1.0,
)
prediction_horizon_seconds = st.number_input(
    "horizonte_prediccion (segundos de antelación)",
    min_value=0.5,
    max_value=600.0,
    value=3.0,
    step=0.5,
)
stride_samples = st.number_input(
    "stride_muestras (salto entre ventanas)",
    min_value=1,
    max_value=120,
    value=max(1, int(round(default_stride))),
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

window_size_samples = max(2, int(round(float(window_size_seconds) * float(dataset_fs))))
prediction_horizon_samples = max(
    1, int(round(float(prediction_horizon_seconds) * float(dataset_fs)))
)
st.caption(
    f"Conversión actual con fs={dataset_fs:.3g} Hz: "
    f"tam_ventana={window_size_samples} muestras, "
    f"horizonte={prediction_horizon_samples} muestras."
)

st.subheader("Hiperparámetros por modelo")
st.caption("Modo pruebas activo: el entrenamiento se ejecuta solo con Random Forest.")
hyperparams_inputs: dict[str, str] = {}
for key, spec in MODEL_REGISTRY.items():
    st.markdown(f"**{spec.display_name}**")
    text_value = st.text_area(
        f"Hiperparámetros ({spec.display_name}) [dict Python]",
        value=st.session_state.get(f"hyperparams_text_{key}", str(spec.defaults)),
        key=f"hyperparams_text_{key}",
        height=140,
    )
    hyperparams_inputs[key] = text_value

if st.button("Entrenar todos los modelos"):
    all_results: dict[str, dict] = {}
    train_errors: dict[str, str] = {}
    model_keys = ["random_forest"]
    user_hyperparams: dict[str, dict] = {}
    parse_errors: list[str] = []
    for key in model_keys:
        model_name = MODEL_REGISTRY[key].display_name
        raw_text = hyperparams_inputs.get(key, "").strip()
        try:
            parsed = ast.literal_eval(raw_text)
            if not isinstance(parsed, dict):
                raise ValueError("Debe ser un diccionario Python.")
            user_hyperparams[key] = parsed
        except Exception as exc:
            parse_errors.append(f"{model_name}: {exc}")

    if parse_errors:
        st.error("Hiperparámetros inválidos en uno o más modelos:")
        st.code("\n".join(parse_errors))
        st.stop()

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
        progress_state = {"last_bucket": -1}

        def on_model_progress(fraction: float, detail: str):
            frac = min(1.0, max(0.0, float(fraction)))
            done_models = (idx - 1) + frac
            overall_frac = done_models / len(model_keys)
            elapsed_local = time.perf_counter() - global_start
            avg_per_model = elapsed_local / max(done_models, 1e-6)
            remaining_models = len(model_keys) - done_models
            eta_local = avg_per_model * max(remaining_models, 0.0)
            status_placeholder.info(
                f"Entrenando {model_name} ({idx}/{len(model_keys)}) · {detail} · "
                f"transcurrido {elapsed_local:.1f}s · ETA {eta_local:.1f}s"
            )
            progress.progress(
                overall_frac,
                text=(
                    f"Completado {done_models:.1f}/{len(model_keys)} · "
                    f"transcurrido {elapsed_local:.1f}s · ETA {eta_local:.1f}s"
                ),
            )
            bucket = int(frac * 10)
            if bucket > progress_state["last_bucket"]:
                progress_state["last_bucket"] = bucket
                push_log(f"{model_name}: {detail}.")

        try:
            result = train_and_evaluate(
                df=data,
                target_column=target_col,
                model_key=key,
                hyperparams=user_hyperparams.get(key, dict(MODEL_REGISTRY[key].defaults)),
                window_size=int(window_size_samples),
                prediction_horizon=int(prediction_horizon_samples),
                stride=int(stride_samples),
                detection_threshold=float(detection_threshold),
                alarm_min_consecutive=int(alarm_min_consecutive),
                alarm_refractory=int(alarm_refractory),
                sampling_frequency_hz=float(dataset_fs),
                progress_callback=on_model_progress,
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
                        f"{min_early.get('patient_id')} ({float(min_early.get('lead_time_seconds')):.2f}s)"
                        if min_early
                        else "N/A"
                    ),
                    "Paciente mayor tiempo detección": (
                        f"{max_early.get('patient_id')} ({float(max_early.get('lead_time_seconds')):.2f}s)"
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
                    f"{min_early.get('patient_id')} ({float(min_early.get('lead_time_seconds')):.2f}s)"
                    if min_early
                    else "N/A"
                ),
                "Paciente mayor tiempo detección": (
                    f"{max_early.get('patient_id')} ({float(max_early.get('lead_time_seconds')):.2f}s)"
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

if train_results_by_model:
    st.subheader("Guardar modelo entrenado")
    save_model_key = st.selectbox(
        "Modelo a guardar",
        options=list(train_results_by_model.keys()),
        format_func=lambda key: MODEL_REGISTRY[key].display_name,
        key="save_trained_model_key",
    )
    default_model_name = f"{MODEL_REGISTRY[save_model_key].display_name}_v1"
    save_model_name = st.text_input(
        "Nombre del modelo",
        value=st.session_state.get("save_model_name", default_model_name),
        key="save_model_name",
    )
    if st.button("Guardar modelo entrenado", key="btn_save_trained_model"):
        try:
            result = train_results_by_model[save_model_key]
            artifact = {
                "model_key": save_model_key,
                "display_name": MODEL_REGISTRY[save_model_key].display_name,
                "model": result.get("model"),
                "signal_columns": result.get("signal_columns", []),
                "window_size_samples": int(result.get("window_size_samples", result.get("window_size", 5))),
                "prediction_horizon_samples": int(
                    result.get("prediction_horizon_samples", result.get("prediction_horizon", 3))
                ),
                "stride_samples": int(result.get("stride", 1)),
                "detection_threshold": float(result.get("detection_threshold", 0.35)),
                "time_column": result.get("time_column", "t"),
                "sampling_frequency_hz": float(result.get("sampling_frequency_hz", dataset_fs)),
            }
            p = save_user_model_artifact_named(
                APP_ROOT, active_user, save_model_key, save_model_name, artifact
            )
            st.success(f"Modelo guardado como: {p.name}")
        except Exception as exc:
            st.error(f"No se pudo guardar el modelo: {exc}")

user_models = list_user_models(APP_ROOT, active_user)
with st.sidebar.expander("Mis modelos guardados", expanded=False):
    if user_models:
        for m in user_models[:20]:
            st.caption(f"{m['name']} · {m['size_kb']} KB")
    else:
        st.caption("Aún no tienes modelos guardados.")

if user_models:
    st.subheader("Pronóstico con modelo guardado")
    model_choice = st.selectbox(
        "Modelo guardado",
        options=user_models,
        format_func=lambda m: m["name"],
        key="saved_model_choice",
    )
    model_path = Path(model_choice["path"])
    with model_path.open("rb") as f:
        st.download_button(
            "Exportar modelo (.pkl)",
            data=f.read(),
            file_name=model_path.name,
            mime="application/octet-stream",
        )

    forecast_fs = float(
        st.number_input(
            "fs para CSV de pronóstico (muestras/segundo)",
            min_value=0.1,
            max_value=500.0,
            value=float(dataset_fs),
            step=0.1,
            key="forecast_fs",
        )
    )
    forecast_source = st.radio(
        "Dataset para pronóstico",
        options=["Usar dataset cargado actualmente", "Subir CSV nuevo"],
        horizontal=True,
        key="forecast_source_mode",
    )
    forecast_file = None
    if forecast_source == "Subir CSV nuevo":
        forecast_file = st.file_uploader(
            "CSV nuevo para pronóstico",
            type=["csv"],
            key="forecast_file",
        )
    if st.button("Ejecutar pronóstico con modelo guardado", key="btn_forecast_saved"):
        try:
            artifact = load_model_artifact(model_path)
            if forecast_source == "Usar dataset cargado actualmente":
                forecast_df = data.copy()
                forecast_df["fs"] = float(forecast_fs)
            else:
                if forecast_file is None:
                    st.error("Sube un CSV para pronóstico.")
                    forecast_df = None
                else:
                    forecast_df = read_csv(forecast_file)
                    forecast_df["fs"] = float(forecast_fs)
            if forecast_df is not None:
                out_df, payloads = forecast_with_saved_model_details(
                    model=artifact["model"],
                    df=forecast_df,
                    signal_columns=list(artifact.get("signal_columns", [])),
                    time_column=str(artifact.get("time_column", "t")),
                    window_size_samples=int(artifact.get("window_size_samples", 5)),
                    stride_samples=int(artifact.get("stride_samples", 1)),
                    threshold=float(artifact.get("detection_threshold", 0.35)),
                )
                st.dataframe(out_df, width="stretch")
                st.download_button(
                    "Descargar pronóstico CSV",
                    data=out_df.to_csv(index=False).encode("utf-8"),
                    file_name="pronostico_modelo_guardado.csv",
                    mime="text/csv",
                )
                st.session_state["saved_forecast_table"] = out_df
                st.session_state["saved_forecast_payloads"] = payloads
                st.session_state["saved_forecast_df"] = forecast_df
                st.session_state["saved_forecast_time_column"] = str(
                    artifact.get("time_column", "t")
                )
                st.session_state["saved_forecast_signal_columns"] = list(
                    artifact.get("signal_columns", [])
                )
                st.session_state["saved_forecast_threshold"] = float(
                    artifact.get("detection_threshold", 0.35)
                )
        except Exception as exc:
            st.error(f"No se pudo ejecutar pronóstico: {exc}")

saved_forecast_table = st.session_state.get("saved_forecast_table")
saved_forecast_payloads = st.session_state.get("saved_forecast_payloads")
saved_forecast_df = st.session_state.get("saved_forecast_df")
saved_forecast_time_column = st.session_state.get("saved_forecast_time_column")
saved_forecast_signal_columns = st.session_state.get("saved_forecast_signal_columns", [])
saved_forecast_threshold = float(st.session_state.get("saved_forecast_threshold", 0.35))

if (
    isinstance(saved_forecast_table, pd.DataFrame)
    and isinstance(saved_forecast_payloads, dict)
    and isinstance(saved_forecast_df, pd.DataFrame)
    and saved_forecast_time_column in saved_forecast_df.columns
):
    st.subheader("Visualización final del pronóstico")
    view_rows = saved_forecast_table.copy()
    patient_opts = view_rows["patient_id"].tolist()
    selected_pid = st.selectbox(
        "Paciente para visualizar (pronóstico)",
        options=patient_opts,
        format_func=lambda pid: f"Paciente {pid}",
        key="forecast_selected_patient",
    )
    payload = saved_forecast_payloads.get(str(selected_pid), {})
    pred_times = np.asarray(payload.get("pred_times", []), dtype=float)
    pred_proba = np.asarray(payload.get("pred_proba", []), dtype=float)
    alarm_times = np.asarray(payload.get("alarm_times", []), dtype=float)
    ev_start_t = payload.get("event_start_time")
    ev_end_t = payload.get("event_end_time")

    patient_df = saved_forecast_df[
        saved_forecast_df["patient_id"].astype(str) == str(selected_pid)
    ].sort_values(saved_forecast_time_column)
    signal_cols = [c for c in saved_forecast_signal_columns if c in patient_df.columns]
    if not signal_cols:
        signal_cols = [
            c
            for c in patient_df.columns
            if c not in {"patient_id", saved_forecast_time_column, "event", "anomaly", "fs"}
            and pd.api.types.is_numeric_dtype(patient_df[c])
        ][:4]

    if len(pred_times) > 0 and len(signal_cols) > 0:
        if saved_forecast_time_column == "minute":
            x_pred = pred_times
            x_alarm = alarm_times if len(alarm_times) else np.array([])
            xx = patient_df[saved_forecast_time_column].to_numpy(dtype=float)
            xlabel = "Tiempo (minutos)"
            ev_start_x = ev_start_t
            ev_end_x = ev_end_t
            margin_before = 2.0
            margin_after = 0.5
        else:
            x_pred = pred_times / 60.0
            x_alarm = alarm_times / 60.0 if len(alarm_times) else np.array([])
            xx = patient_df[saved_forecast_time_column].to_numpy(dtype=float) / 60.0
            xlabel = "Tiempo (minutos)"
            ev_start_x = None if ev_start_t is None else float(ev_start_t) / 60.0
            ev_end_x = None if ev_end_t is None else float(ev_end_t) / 60.0
            margin_before = 2.0
            margin_after = 0.5

        if ev_start_x is not None:
            x_min = max(float(np.min(xx)), float(ev_start_x) - margin_before)
            x_max = min(float(np.max(xx)), float(ev_start_x) + margin_after)
        else:
            x_min = float(np.min(xx))
            x_max = float(np.max(xx))

        prob_df = pd.DataFrame({"x": x_pred, "prob": pred_proba})
        prob_df = prob_df[(prob_df["x"] >= x_min) & (prob_df["x"] <= x_max)]
        signal_df = patient_df[[saved_forecast_time_column] + signal_cols].copy()
        signal_df["x"] = xx
        signal_df = signal_df[(signal_df["x"] >= x_min) & (signal_df["x"] <= x_max)]
        alarm_df = pd.DataFrame({"x": x_alarm, "y": saved_forecast_threshold})
        if not alarm_df.empty:
            alarm_df = alarm_df[(alarm_df["x"] >= x_min) & (alarm_df["x"] <= x_max)]

        event_span_df = pd.DataFrame(columns=["x", "x2"])
        if ev_start_x is not None and ev_end_x is not None:
            ev_left = max(x_min, float(ev_start_x))
            ev_right = min(x_max, float(ev_end_x))
            if ev_left < ev_right:
                event_span_df = pd.DataFrame([{"x": ev_left, "x2": ev_right}])

        prob_base = alt.Chart(prob_df).encode(
            x=alt.X("x:Q", title=xlabel, scale=alt.Scale(domain=[x_min, x_max]))
        )
        prob_area = prob_base.mark_area(opacity=0.22, color="#6aa7ff").encode(
            y=alt.Y("prob:Q", title="Probabilidad", scale=alt.Scale(domain=[0, 1]))
        )
        prob_line = prob_base.mark_line(color="#6aa7ff", strokeWidth=2).encode(
            y=alt.Y("prob:Q", title="Probabilidad", scale=alt.Scale(domain=[0, 1]))
        )
        threshold_rule = alt.Chart(pd.DataFrame({"thr": [saved_forecast_threshold]})).mark_rule(
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
            event_rect + prob_area + prob_line + threshold_rule + alarm_lines + alarm_points
        ).properties(height=300, title="Streaming: probabilidad y alarmas (modelo guardado)").resolve_scale(x="shared")
        st.altair_chart(prob_chart, use_container_width=True)

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
            height=320, title="Señales del paciente (pronóstico)"
        ).resolve_scale(x="shared")
        st.altair_chart(signal_chart, use_container_width=True)

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
            "Tiempo esperado de predicción (horizonte): "
            f"{patient_metrics.get('expected_prediction_horizon_seconds', prediction_horizon_seconds):.2f}s "
            f"({patient_metrics.get('expected_prediction_horizon_samples', prediction_horizon_samples)} muestras)"
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
                    f"lead={float(case['lead_time_seconds']):.2f}s"
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
