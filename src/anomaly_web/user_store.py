"""Autenticación simple y almacenamiento por usuario para la app Streamlit."""

from __future__ import annotations

import hashlib
import json
import pickle
import re
import secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_.-]{3,32}$")


@dataclass(frozen=True)
class UserPaths:
    root: Path
    datasets: Path
    models: Path


def _users_file(base_dir: Path) -> Path:
    return base_dir / "data" / "users" / "users.json"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_users(base_dir: Path) -> dict[str, dict[str, Any]]:
    users_path = _users_file(base_dir)
    if not users_path.exists():
        return {}
    try:
        return json.loads(users_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_users(base_dir: Path, users: dict[str, dict[str, Any]]) -> None:
    users_path = _users_file(base_dir)
    _ensure_parent(users_path)
    users_path.write_text(json.dumps(users, ensure_ascii=True, indent=2), encoding="utf-8")


def validate_username(username: str) -> bool:
    return bool(_USERNAME_RE.match(username))


def _hash_password(password: str, salt_hex: str | None = None) -> str:
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 150_000)
    return f"{salt.hex()}:{digest.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, _ = stored.split(":", 1)
        return _hash_password(password, salt_hex=salt_hex) == stored
    except Exception:
        return False


def register_user(base_dir: Path, username: str, password: str) -> tuple[bool, str]:
    username = username.strip()
    if not validate_username(username):
        return False, "Usuario inválido. Usa 3-32 caracteres: letras, números, . _ -"
    if len(password) < 6:
        return False, "La contraseña debe tener al menos 6 caracteres."

    users = _load_users(base_dir)
    if username in users:
        return False, "El usuario ya existe."

    users[username] = {
        "password_hash": _hash_password(password),
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    _save_users(base_dir, users)
    ensure_user_dirs(base_dir, username)
    return True, "Usuario registrado correctamente."


def authenticate_user(base_dir: Path, username: str, password: str) -> bool:
    users = _load_users(base_dir)
    user = users.get(username.strip())
    if not user:
        return False
    return _verify_password(password, user.get("password_hash", ""))


def ensure_user_dirs(base_dir: Path, username: str) -> UserPaths:
    root = base_dir / "data" / "users" / username
    datasets = root / "datasets"
    models = root / "models"
    datasets.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    return UserPaths(root=root, datasets=datasets, models=models)


def save_user_dataset(base_dir: Path, username: str, df: pd.DataFrame, source: str) -> Path:
    paths = ensure_user_dirs(base_dir, username)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_source = re.sub(r"[^a-zA-Z0-9_-]+", "_", source.strip().lower()) or "dataset"
    out = paths.datasets / f"{stamp}_{safe_source}.csv"
    df.to_csv(out, index=False)
    return out


def list_user_datasets(base_dir: Path, username: str) -> list[dict[str, Any]]:
    paths = ensure_user_dirs(base_dir, username)
    items = []
    for p in sorted(paths.datasets.glob("*.csv"), reverse=True):
        items.append(
            {
                "name": p.name,
                "path": str(p),
                "size_kb": round(p.stat().st_size / 1024.0, 1),
                "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
            }
        )
    return items


def save_user_model_artifact(
    base_dir: Path,
    username: str,
    model_key: str,
    artifact: dict[str, Any],
) -> Path:
    paths = ensure_user_dirs(base_dir, username)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = paths.models / f"{stamp}_{model_key}.pkl"
    with out.open("wb") as f:
        pickle.dump(artifact, f)
    return out


def list_user_models(base_dir: Path, username: str) -> list[dict[str, Any]]:
    paths = ensure_user_dirs(base_dir, username)
    items = []
    for p in sorted(paths.models.glob("*.pkl"), reverse=True):
        items.append(
            {
                "name": p.name,
                "path": str(p),
                "size_kb": round(p.stat().st_size / 1024.0, 1),
                "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
            }
        )
    return items


def load_model_artifact(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)
