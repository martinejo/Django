"""Funciones de entrenamiento y evaluación para la app de anomalías."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable

import numpy as np
import pandas as pd

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


def _fit_model_with_progress(
    model,
    model_key: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    progress_callback: Callable[[float, str], None] | None = None,
):
    """Ajusta el modelo con callback de progreso cuando es posible."""
    if progress_callback is None:
        model.fit(x_train, y_train)
        return

    if model_key == "gradient_boosting" and hasattr(model, "n_estimators"):
        total_estimators = max(1, int(getattr(model, "n_estimators", 100)))
        chunk = max(5, min(25, total_estimators // 10 if total_estimators >= 10 else 1))
        try:
            model.set_params(warm_start=True)
        except Exception:
            model.fit(x_train, y_train)
            progress_callback(1.0, "entrenamiento completado")
            return

        done = 0
        while done < total_estimators:
            done = min(total_estimators, done + chunk)
            model.set_params(n_estimators=done)
            model.fit(x_train, y_train)
            progress_callback(done / total_estimators, f"árboles {done}/{total_estimators}")
        return

    model.fit(x_train, y_train)
    progress_callback(1.0, "entrenamiento completado")


def read_csv(file_obj):
    """Lee un CSV desde un path o file-like object."""
    return pd.read_csv(file_obj)


def _resolve_time_column(df, time_column: str | None) -> str:
    if time_column is None:
        if "second" in df.columns:
            return "second"
        if "minute" in df.columns:
            return "minute"
        if "t" in df.columns:
            return "t"
        raise ValueError(
            "Falta columna temporal. Se esperaba 'second', 'minute' o 't' para ordenar las ventanas."
        )
    if time_column not in df.columns:
        raise ValueError(f"Falta la columna temporal '{time_column}' para ordenar las ventanas.")
    return time_column


def _time_to_seconds_factor(time_column: str) -> float:
    """Factor para convertir la unidad temporal del dataset a segundos."""
    if time_column == "minute":
        return 60.0
    return 1.0


def _patient_window_arrays(
    group_sorted: pd.DataFrame,
    signal_columns: list[str],
    target_column: str,
    time_column: str,
    window_size: int,
    prediction_horizon: int,
    stride: int = 1,
    include_windows: bool = True,
):
    """Construye ventanas de un paciente y sus etiquetas."""
    y_event = group_sorted[target_column].to_numpy(dtype=np.int8)
    x_signal = group_sorted[signal_columns].to_numpy(dtype=np.float32, copy=False)
    time_values = group_sorted[time_column].to_numpy()

    n = len(group_sorted)
    if n < window_size + 1:
        return None

    window_end_idx = np.arange(window_size - 1, n, stride)

    next_event_idx = np.full(n, np.inf)
    next_pos = np.inf
    for i in range(n - 1, -1, -1):
        if y_event[i] == 1:
            next_pos = i
        next_event_idx[i] = next_pos

    distances = next_event_idx[window_end_idx] - window_end_idx
    labels = ((distances >= 1) & (distances <= prediction_horizon)).astype(np.int8)

    windows = None
    if include_windows:
        all_windows = np.lib.stride_tricks.sliding_window_view(
            x_signal, window_shape=window_size, axis=0
        ).reshape(-1, window_size * len(signal_columns))
        row_idx = window_end_idx - (window_size - 1)
        windows = all_windows[row_idx]
    window_end_time = time_values[window_end_idx]

    return windows, labels, window_end_time


def _build_window_dataset(
    df,
    target_column: str,
    window_size: int,
    prediction_horizon: int,
    patient_id_column: str = "patient_id",
    time_column: str | None = None,
    max_windows: int = 250_000,
    random_state: int = 42,
    stride: int = 1,
):
    """Transforma series por paciente en ejemplos de ventana deslizante con muestreo."""
    if target_column not in df.columns:
        raise ValueError(f"La columna objetivo '{target_column}' no existe en el CSV.")
    if patient_id_column not in df.columns:
        raise ValueError(
            f"Falta la columna '{patient_id_column}' para construir ventanas por paciente."
        )
    time_column = _resolve_time_column(df, time_column)

    if window_size < 2:
        raise ValueError("tam_ventana debe ser >= 2.")
    if prediction_horizon < 1:
        raise ValueError("horizonte_prediccion debe ser >= 1.")
    if stride < 1:
        raise ValueError("stride_muestras debe ser >= 1.")

    excluded = {patient_id_column, time_column, target_column}
    signal_columns = [
        col for col in df.columns if col not in excluded and np.issubdtype(df[col].dtype, np.number)
    ]
    if not signal_columns:
        raise ValueError("No se detectaron señales numéricas para entrenar.")

    grouped = list(df.groupby(patient_id_column))
    total_windows = 0
    total_positives = 0

    for _, group in grouped:
        group_sorted = group.sort_values(time_column).reset_index(drop=True)
        result = _patient_window_arrays(
            group_sorted=group_sorted,
            signal_columns=signal_columns,
            target_column=target_column,
            time_column=time_column,
            window_size=window_size,
            prediction_horizon=prediction_horizon,
            stride=stride,
            include_windows=False,
        )
        if result is None:
            continue
        _, labels, _ = result
        total_windows += len(labels)
        total_positives += int(labels.sum())

    if total_windows == 0:
        raise ValueError("No se pudieron construir ventanas. Revisa tamaño de ventana y datos.")

    total_negatives = total_windows - total_positives
    if total_negatives <= 0:
        raise ValueError("Las ventanas generadas quedaron con una sola clase positiva.")

    desired_total = max(total_positives + 1, min(max_windows, total_windows))
    desired_negatives = max(1, desired_total - total_positives)
    negative_keep_prob = min(1.0, desired_negatives / total_negatives)

    rng = np.random.default_rng(random_state)
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    meta_rows: list[pd.DataFrame] = []

    for patient_id, group in grouped:
        group_sorted = group.sort_values(time_column).reset_index(drop=True)
        result = _patient_window_arrays(
            group_sorted=group_sorted,
            signal_columns=signal_columns,
            target_column=target_column,
            time_column=time_column,
            window_size=window_size,
            prediction_horizon=prediction_horizon,
            stride=stride,
        )
        if result is None:
            continue
        windows, labels, window_end_time = result

        pos_mask = labels == 1
        neg_mask = ~pos_mask

        sampled_neg_mask = rng.random(len(labels)) < negative_keep_prob
        keep_mask = pos_mask | (neg_mask & sampled_neg_mask)
        if not np.any(keep_mask):
            continue

        kept_idx = np.flatnonzero(keep_mask)
        x_rows.append(windows[kept_idx].astype(np.float32, copy=False))
        y_rows.append(labels[kept_idx].astype(np.int8, copy=False))
        meta_rows.append(
            pd.DataFrame(
                {
                    "patient_id": patient_id,
                    "window_end_time": window_end_time[kept_idx],
                }
            )
        )

    if not x_rows:
        raise ValueError("No se pudieron construir ventanas. Revisa tamaño de ventana y datos.")

    x = np.vstack(x_rows).astype(np.float32, copy=False)
    y = np.concatenate(y_rows).astype(np.int8, copy=False)
    meta = pd.concat(meta_rows, ignore_index=True)

    if len(np.unique(y)) < 2:
        raise ValueError(
            "Las ventanas generadas quedaron con una sola clase. "
            "Ajusta tam_ventana/horizonte_prediccion o revisa eventos."
        )

    return x, y, signal_columns, meta, time_column


def train_and_evaluate(
    df,
    target_column: str,
    model_key: str,
    hyperparams: dict[str, Any] | None = None,
    window_size: int = 5,
    prediction_horizon: int = 3,
    detection_threshold: float = 0.35,
    alarm_min_consecutive: int = 2,
    alarm_refractory: int = 10,
    stride: int = 1,
    sampling_frequency_hz: float = 1.0,
    progress_callback: Callable[[float, str], None] | None = None,
):
    """Entrena un modelo por ventanas y devuelve métricas de clasificación."""
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split

    if "patient_id" not in df.columns:
        raise ValueError("Se requiere columna 'patient_id' para dividir train/test por paciente.")

    patient_summary = (
        df.groupby("patient_id")[target_column].max().rename("has_event").reset_index()
    )
    stratify_labels = (
        patient_summary["has_event"] if patient_summary["has_event"].nunique() > 1 else None
    )
    train_patients, test_patients = train_test_split(
        patient_summary["patient_id"],
        test_size=0.2,
        random_state=42,
        stratify=stratify_labels,
    )
    train_df = df[df["patient_id"].isin(train_patients)].copy()
    test_df = df[df["patient_id"].isin(test_patients)].copy()

    train_max_windows = 220_000
    test_max_windows = 80_000
    if model_key == "random_forest":
        # RandomForest con ventanas largas puede volverse muy pesado en VPS.
        # Reducimos muestra para mantener tiempos de entrenamiento razonables.
        train_max_windows = 70_000
        test_max_windows = 25_000

    x_train, y_train, signal_columns, _, time_column = _build_window_dataset(
        df=train_df,
        target_column=target_column,
        window_size=window_size,
        prediction_horizon=prediction_horizon,
        max_windows=train_max_windows,
        random_state=42,
        stride=stride,
    )
    x_test, y_test, _, _, _ = _build_window_dataset(
        df=test_df,
        target_column=target_column,
        window_size=window_size,
        prediction_horizon=prediction_horizon,
        max_windows=test_max_windows,
        random_state=43,
        stride=stride,
    )
    model = build_model(model_key, hyperparams)
    _fit_model_with_progress(
        model=model,
        model_key=model_key,
        x_train=x_train,
        y_train=y_train,
        progress_callback=progress_callback,
    )
    time_factor = _time_to_seconds_factor(time_column)

    y_pred = model.predict(x_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(x_test)[:, 1]
    else:
        y_prob = y_pred.astype(float)

    detection_threshold = float(detection_threshold)
    detection_threshold = min(0.99, max(0.01, detection_threshold))

    event_case_summaries = []
    event_case_timelines: dict[str, list[dict[str, Any]]] = {}
    test_patient_payloads: dict[str, dict[str, Any]] = {}
    test_patient_ids = test_df["patient_id"].drop_duplicates().tolist()

    for patient_id in test_patient_ids:
        patient_group = test_df[test_df["patient_id"] == patient_id].sort_values(time_column)
        event_times = patient_group.loc[patient_group[target_column] == 1, time_column].to_numpy(dtype=float)
        has_event = len(event_times) > 0
        event_start_time = float(event_times[0]) if has_event else None
        event_end_time = float(event_times[-1]) if has_event else None

        patient_windows = _patient_window_arrays(
            group_sorted=patient_group,
            signal_columns=signal_columns,
            target_column=target_column,
            time_column=time_column,
            window_size=window_size,
            prediction_horizon=prediction_horizon,
            stride=stride,
        )
        if patient_windows is None:
            continue
        patient_x, _, patient_window_end_time = patient_windows
        if hasattr(model, "predict_proba"):
            patient_prob = model.predict_proba(patient_x)[:, 1]
        else:
            patient_prob = model.predict(patient_x).astype(float)

        above = patient_prob >= detection_threshold
        alarm_min_consecutive = max(1, int(alarm_min_consecutive))
        alarm_refractory = max(0, int(alarm_refractory))

        alarm_times: list[float] = []
        consecutive = 0
        last_alarm_t = float(-10**9)
        for i, t_curr in enumerate(patient_window_end_time):
            t_curr = float(t_curr)
            if t_curr - last_alarm_t < alarm_refractory:
                consecutive = 0
                continue

            if above[i]:
                consecutive += 1
            else:
                consecutive = 0

            if consecutive >= alarm_min_consecutive:
                alarm_times.append(t_curr)
                last_alarm_t = t_curr
                consecutive = 0

        first_alarm_time = float(alarm_times[0]) if len(alarm_times) > 0 else None
        lead_time_seconds = None
        false_alarm_count = 0

        if has_event:
            alarms_np = np.asarray(alarm_times, dtype=float)
            if first_alarm_time is not None:
                lead_time_seconds = float((event_start_time - first_alarm_time) * time_factor)
            if len(alarms_np) > 0:
                valid = (alarms_np < event_start_time) | (
                    (alarms_np >= event_start_time) & (alarms_np <= event_end_time)
                )
                false_alarm_count = int((~valid).sum())
        else:
            false_alarm_count = int(len(alarm_times))

        summary = {
            "patient_id": int(patient_id) if str(patient_id).isdigit() else str(patient_id),
            "is_test_patient": True,
            "has_event": bool(has_event),
            "event_start_time": event_start_time,
            "event_end_time": event_end_time,
            "first_alarm_time": first_alarm_time,
            "lead_time_seconds": lead_time_seconds,
            "false_alarm_count": false_alarm_count,
        }
        event_case_summaries.append(summary)

        event_case_timelines[str(patient_id)] = [
            {
                "time": float(t),
                "pred_prob": float(prob),
                "pred_positive": int(prob >= detection_threshold),
                "true_event": int(
                    has_event and event_start_time is not None and event_end_time is not None
                    and event_start_time <= float(t) <= event_end_time
                ),
            }
            for t, prob in zip(patient_window_end_time, patient_prob)
        ]
        test_patient_payloads[str(patient_id)] = {
            "pred_times": [float(t) for t in patient_window_end_time.tolist()],
            "pred_proba": [float(p) for p in patient_prob.tolist()],
            "alarm_times": [float(t) for t in alarm_times],
            "event_start_time": event_start_time,
            "event_end_time": event_end_time,
        }

    event_case_summaries.sort(key=lambda x: (not x["has_event"], x["patient_id"]))

    patients_with_event = sum(1 for case in event_case_summaries if case["has_event"])
    detected_event_cases = [
        case
        for case in event_case_summaries
        if case["has_event"] and case["first_alarm_time"] is not None
    ]
    patients_detected = len(detected_event_cases)
    lead_times_seconds = [
        float(case["lead_time_seconds"])
        for case in detected_event_cases
        if case["lead_time_seconds"] is not None
    ]
    early_detected_event_cases = [
        case
        for case in detected_event_cases
        if case["lead_time_seconds"] is not None and float(case["lead_time_seconds"]) > 0
    ]
    early_lead_times_seconds = [float(case["lead_time_seconds"]) for case in early_detected_event_cases]
    false_alarm_counts_no_event = [
        int(case["false_alarm_count"])
        for case in event_case_summaries
        if not case["has_event"]
    ]

    sensitivity_by_patient = (
        float(patients_detected / patients_with_event) if patients_with_event > 0 else None
    )
    lead_times_np = (
        np.asarray(lead_times_seconds, dtype=np.float32)
        if len(lead_times_seconds) > 0
        else np.asarray([], dtype=np.float32)
    )
    early_lead_times_np = (
        np.asarray(early_lead_times_seconds, dtype=np.float32)
        if len(early_lead_times_seconds) > 0
        else np.asarray([], dtype=np.float32)
    )
    false_alarm_avg_no_event = (
        float(np.mean(false_alarm_counts_no_event))
        if len(false_alarm_counts_no_event) > 0
        else 0.0
    )

    patient_level_metrics = {
        "patients_with_event": int(patients_with_event),
        "patients_detected": int(patients_detected),
        "patients_detected_pre_event": int(len(early_detected_event_cases)),
        "sensitivity_by_patient": sensitivity_by_patient,
        "expected_prediction_horizon": int(prediction_horizon),
        "expected_prediction_horizon_samples": int(prediction_horizon),
        "expected_prediction_horizon_seconds": float(
            prediction_horizon / max(float(sampling_frequency_hz), 1e-6)
        ),
        "lead_times_seconds": [float(v) for v in lead_times_seconds],
        "early_lead_times_seconds": [float(v) for v in early_lead_times_seconds],
        "lead_time_mean_seconds": float(lead_times_np.mean()) if lead_times_np.size else None,
        "early_lead_time_mean_seconds": (
            float(early_lead_times_np.mean()) if early_lead_times_np.size else None
        ),
        "lead_time_median_seconds": float(np.median(lead_times_np)) if lead_times_np.size else None,
        "lead_time_min_seconds": int(lead_times_np.min()) if lead_times_np.size else None,
        "lead_time_max_seconds": int(lead_times_np.max()) if lead_times_np.size else None,
        "early_lead_time_min_seconds": (
            int(early_lead_times_np.min()) if early_lead_times_np.size else None
        ),
        "early_lead_time_max_seconds": (
            int(early_lead_times_np.max()) if early_lead_times_np.size else None
        ),
        "patient_min_early_detection": (
            {
                "patient_id": early_detected_event_cases[
                    int(np.argmin(np.asarray(early_lead_times_seconds, dtype=np.float32)))
                ]["patient_id"],
                "lead_time_seconds": float(min(early_lead_times_seconds)),
            }
            if early_lead_times_seconds
            else None
        ),
        "patient_max_early_detection": (
            {
                "patient_id": early_detected_event_cases[
                    int(np.argmax(np.asarray(early_lead_times_seconds, dtype=np.float32)))
                ]["patient_id"],
                "lead_time_seconds": float(max(early_lead_times_seconds)),
            }
            if early_lead_times_seconds
            else None
        ),
        "pct_detected_gt_half_horizon": (
            float(np.mean(lead_times_np > (prediction_horizon / 2.0)) * 100.0)
            if lead_times_np.size
            else None
        ),
        "pct_detected_gt_horizon": (
            float(np.mean(lead_times_np > float(prediction_horizon)) * 100.0)
            if lead_times_np.size
            else None
        ),
        "false_alarms_avg_no_event_patients": float(false_alarm_avg_no_event),
    }

    return {
        "model": model,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "report": classification_report(y_test, y_pred, zero_division=0),
        "num_windows": int(len(y_train) + len(y_test)),
        "positive_rate": float(np.mean(np.concatenate([y_train, y_test]))),
        "signal_columns": signal_columns,
        "event_case_summaries": event_case_summaries,
        "event_case_timelines": event_case_timelines,
        "window_size": int(window_size),
        "window_size_samples": int(window_size),
        "window_size_seconds": float(window_size / max(float(sampling_frequency_hz), 1e-6)),
        "prediction_horizon": int(prediction_horizon),
        "prediction_horizon_samples": int(prediction_horizon),
        "prediction_horizon_seconds": float(
            prediction_horizon / max(float(sampling_frequency_hz), 1e-6)
        ),
        "detection_threshold": detection_threshold,
        "alarm_min_consecutive": int(alarm_min_consecutive),
        "alarm_refractory": int(alarm_refractory),
        "stride": int(stride),
        "sampling_frequency_hz": float(sampling_frequency_hz),
        "time_column": time_column,
        "test_patient_summaries": event_case_summaries,
        "test_patient_payloads": test_patient_payloads,
        "patient_level_metrics": patient_level_metrics,
    }
