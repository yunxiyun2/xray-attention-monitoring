"""Fixed protocol constants for the v7 predictive-validity analysis."""

RETAINED = ("FD", "PSV", "SA", "FN", "SE", "GTE", "SV")
Z_COLUMNS = tuple(f"{column}_z" for column in RETAINED)

F1_COMPONENTS = {
    "FD": 1.0,
    "PSV": -1.0,
    "SA": -1.0,
    "FN": -1.0,
}

F2_COMPONENTS = {
    "SE": 1.0,
    "GTE": 1.0,
    "SV": 1.0,
}

MODELS = {
    "global": ("global",),
    "f1_only": ("f1",),
    "f2_only": ("f2",),
    "two_factor": ("f1", "f2"),
    "seven_feature": Z_COLUMNS,
}


def require_columns(columns, required):
    missing = sorted(set(required).difference(columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
