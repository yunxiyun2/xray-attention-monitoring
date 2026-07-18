from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TaskRecord:
    subject_id: str
    difficulty: str
    label_name: str
    label: int
    errors: np.ndarray


def discover_records(
    data_root: Path, labels: dict[str, int], difficulties: list[str]
) -> list[TaskRecord]:
    records: list[TaskRecord] = []
    for difficulty in difficulties:
        for subject_dir in sorted((data_root / difficulty).iterdir()):
            if not subject_dir.is_dir():
                continue
            for label_name, label in labels.items():
                sequence_path = subject_dir / label_name / "all_errors.txt"
                if not sequence_path.is_file():
                    raise FileNotFoundError(f"Missing required sequence: {sequence_path}")
                errors = np.loadtxt(sequence_path, dtype=float, ndmin=1)
                if errors.size == 0 or not np.isfinite(errors).all():
                    raise ValueError(f"Invalid error sequence: {sequence_path}")
                records.append(
                    TaskRecord(subject_dir.name, difficulty, label_name, label, errors)
                )
    return records
