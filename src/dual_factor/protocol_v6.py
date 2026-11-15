"""v6 scientific definitions. 4/3 structure: SV added to F2 for identifiability."""
RETAINED = ("FD", "PSV", "SA", "FN", "SE", "GTE", "SV")
ALL_INDICATORS = RETAINED + ("SDR",)
GROUPS = {"f1": RETAINED[:4], "f2": RETAINED[4:]}
DIRECTIONS = {"FD": 1, "PSV": -1, "SA": -1, "FN": -1,
              "SE": 1, "GTE": 1, "SV": 1, "SDR": -1}
CRITERIA = {"variance_min": .60, "loading_min": .50,
            "cross_max": .30, "correlation_max": .70}
STABILITY = {"valid_fraction": .95, "assignment_probability": .80,
             "congruence_min": .90, "joint_fraction": .80}


def require_columns(columns, expected):
    missing = set(expected) - set(columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
