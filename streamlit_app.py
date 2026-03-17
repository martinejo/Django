"""Interfaz web simple para entrenar modelos de anomalías desde CSV."""

import ast

import streamlit as st

from src.anomaly_web.config import MODEL_REGISTRY
from src.anomaly_web.training import read_csv, train_and_evaluate


st.set_page_config(page_title="Predicción temprana de anomalías", layout="wide")
st.title("Predicción temprana de anomalías")
st.caption("Carga CSV, elige modelo e hiperparámetros, y entrena en minutos.")

uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])

if uploaded_file is None:
    st.info("Esperando archivo CSV...")
    st.stop()

try:
    data = read_csv(uploaded_file)
except Exception as exc:
    st.error(f"No se pudo leer el CSV: {exc}")
    st.stop()

st.subheader("Vista rápida del dataset")
st.dataframe(data.head(20), use_container_width=True)

columns = list(data.columns)
target_col = st.selectbox("Selecciona la columna objetivo", options=columns)

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
            result = train_and_evaluate(data, target_col, model_key, user_params)
        except Exception as exc:
            st.error(f"Error durante entrenamiento: {exc}")
            st.stop()

    st.success("Entrenamiento completado")
    st.metric("Accuracy", f"{result['accuracy']:.4f}")
    st.text("Classification report")
    st.code(result["report"])
