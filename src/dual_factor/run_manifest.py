"""Immutable run directories, source fingerprints and JSON without fabricated values."""
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
import hashlib
import json
import math
import platform
import numpy as np


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hash(root):
    digest = hashlib.sha256()
    root = Path(root)
    paths = sorted((root / "src" / "dual_factor").rglob("*.py"))
    paths += sorted((root / "scripts").glob("*.py"))
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(data), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def create_run(root):
    path = Path(root) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    return path


def require_teacher_acceptance(approval, report):
    value = json.loads(Path(approval).read_text())
    if value.get("accepted") is not True or not value.get("source") or value.get("report_sha256") != file_hash(report):
        raise ValueError("Real teacher acceptance tied to the exact report is required")
    return value
