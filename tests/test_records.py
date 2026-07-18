from pathlib import Path

import numpy as np
import pytest

from xray_attention.data.records import discover_records


def test_discover_records_rejects_missing_sequence(tmp_path: Path) -> None:
    (tmp_path / "easy" / "01" / "alert").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="all_errors.txt"):
        discover_records(tmp_path, {"alert": 1, "sleepy": 0}, ["easy"])


def _write_sequence(root: Path, label: str, values: str) -> None:
    target = root / "easy" / "01" / label
    target.mkdir(parents=True)
    (target / "all_errors.txt").write_text(values, encoding="utf-8")


def test_discover_records_loads_all_labelled_sequences(tmp_path: Path) -> None:
    _write_sequence(tmp_path, "alert", "10\n20\n")
    _write_sequence(tmp_path, "sleepy", "30\n40\n")
    records = discover_records(tmp_path, {"alert": 1, "sleepy": 0}, ["easy"])
    assert [(item.subject_id, item.label) for item in records] == [("01", 1), ("01", 0)]
    assert np.array_equal(records[0].errors, np.array([10.0, 20.0]))
