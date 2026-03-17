"""Configuración y validación de modelos para entrenamiento de anomalías."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    """Especificación de un modelo entrenable desde la app."""

    key: str
    display_name: str
    sklearn_class: str
    defaults: dict[str, Any]


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "random_forest": ModelSpec(
        key="random_forest",
        display_name="Random Forest",
        sklearn_class="ensemble.RandomForestClassifier",
        defaults={"n_estimators": 200, "max_depth": 10, "random_state": 42},
    ),
    "gradient_boosting": ModelSpec(
        key="gradient_boosting",
        display_name="Gradient Boosting",
        sklearn_class="ensemble.GradientBoostingClassifier",
        defaults={"n_estimators": 150, "learning_rate": 0.05, "random_state": 42},
    ),
    "mlp": ModelSpec(
        key="mlp",
        display_name="MLP Classifier",
        sklearn_class="neural_network.MLPClassifier",
        defaults={
            "hidden_layer_sizes": (64, 32),
            "max_iter": 300,
            "learning_rate_init": 1e-3,
            "random_state": 42,
        },
    ),
}


def get_model_spec(model_key: str) -> ModelSpec:
    """Devuelve la definición del modelo o lanza un error descriptivo."""
    if model_key not in MODEL_REGISTRY:
        allowed = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Modelo no soportado: '{model_key}'. Opciones: {allowed}")
    return MODEL_REGISTRY[model_key]


def merge_hyperparameters(model_key: str, user_params: dict[str, Any] | None) -> dict[str, Any]:
    """Combina defaults del modelo con hiperparámetros del usuario."""
    spec = get_model_spec(model_key)
    merged = dict(spec.defaults)
    if user_params:
        merged.update(user_params)
    return merged
