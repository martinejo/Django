"""Funciones de entrenamiento y evaluación para la app de anomalías."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np

from .config import get_model_spec, merge_hyperparameters


def _load_class(path: str):
    module_name, class_name = path.rsplit(".", 1)
    sklearn_module = import_module(f"sklearn.{module_name}")
    return getattr(sklearn_module, class_name)


def build_model(model_key: str, user_params: dict[str, Any] | None = None):
    """Crea una instancia del modelo solicitado con hiperparámetros finales."""
    spec = get_model_spec(model_key)
    params = merge_hyperparameters(model_key, user_params)
    model_class = _load_class(spec.sklearn_class)
    return model_class(**params)


def read_csv(file_obj):
    """Lee un CSV desde un path o file-like object."""
    import pandas as pd

    return pd.read_csv(file_obj)


def _build_window_dataset(
    df,
    target_column: str,
    window_size: int,
    prediction_horizon: int,
    patient_id_column: str = "patient_id",
    time_column: str = "minute",
):
    """Transforma series por paciente en ejemplos de ventana deslizante."""
    if target_column not in df.columns:
        raise ValueError(f"La columna objetivo '{target_column}' no existe en el CSV.")
    if patient_id_column not in df.columns:
        raise ValueError(
            f"Falta la columna '{patient_id_column}' para construir ventanas por paciente."
        )
    if time_column not in df.columns:
        raise ValueError(f"Falta la columna temporal '{time_column}' para ordenar las ventanas.")

    if window_size < 2:
        raise ValueError("tam_ventana debe ser >= 2.")
    if prediction_horizon < 1:
        raise ValueError("horizonte_prediccion debe ser >= 1.")

    excluded = {patient_id_column, time_column, target_column}
    signal_columns = [
        col for col in df.columns if col not in excluded and np.issubdtype(df[col].dtype, np.number)
    ]
    if not signal_columns:
        raise ValueError("No se detectaron señales numéricas para entrenar.")

    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []

    for _, group in df.groupby(patient_id_column):
        group_sorted = group.sort_values(time_column).reset_index(drop=True)
        y_event = group_sorted[target_column].to_numpy(dtype=int)
        x_signal = group_sorted[signal_columns].to_numpy(dtype=float)

        if len(group_sorted) < window_size + 1:
            continue

        event_positions = np.flatnonzero(y_event == 1)

        for end_idx in range(window_size - 1, len(group_sorted)):
            start_idx = end_idx - window_size + 1
            window_x = x_signal[start_idx : end_idx + 1]
            feature_vector = window_x.reshape(-1)

            is_positive = 0
            if event_positions.size > 0:
                future_distances = event_positions - end_idx
                valid_future = future_distances[(future_distances > 0) & (future_distances <= prediction_horizon)]
                if valid_future.size > 0:
                    is_positive = 1

            x_rows.append(feature_vector)
            y_rows.append(is_positive)

    if not x_rows:
        raise ValueError("No se pudieron construir ventanas. Revisa tamaño de ventana y datos.")

    x = np.vstack(x_rows)
    y = np.asarray(y_rows, dtype=int)

    if len(np.unique(y)) < 2:
        raise ValueError(
            "Las ventanas generadas quedaron con una sola clase. "
            "Ajusta tam_ventana/horizonte_prediccion o revisa eventos."
        )

    return x, y, signal_columns


def train_and_evaluate(
    df,
    target_column: str,
    model_key: str,
    hyperparams: dict[str, Any] | None = None,
    window_size: int = 5,
    prediction_horizon: int = 3,
):
    """Entrena un modelo por ventanas y devuelve métricas de clasificación."""
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split

    x, y, signal_columns = _build_window_dataset(
        df=df,
        target_column=target_column,
        window_size=window_size,
        prediction_horizon=prediction_horizon,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model(model_key, hyperparams)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    return {
        "model": model,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "report": classification_report(y_test, y_pred, zero_division=0),
        "num_windows": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "signal_columns": signal_columns,
    }
