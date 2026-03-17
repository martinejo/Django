"""Funciones de entrenamiento y evaluación para la app de anomalías."""

from importlib import import_module
from typing import Any

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


def train_and_evaluate(df, target_column: str, model_key: str, hyperparams: dict[str, Any] | None = None):
    """Entrena un modelo y devuelve métricas básicas para comparación inicial."""
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split

    if target_column not in df.columns:
        raise ValueError(f"La columna objetivo '{target_column}' no existe en el CSV.")

    x = df.drop(columns=[target_column])
    y = df[target_column]

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
    }
