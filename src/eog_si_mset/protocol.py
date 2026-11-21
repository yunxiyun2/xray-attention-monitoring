"""Fixed constants for SI-MSET."""

PROTOCOL = "si-mset-v1"
VARIANTS = ("full", "single_scale", "no_adversarial", "binary_only")
CHANNELS = ("eog_h", "eog_v", "velocity")
LABEL_NAMES = ("alert", "drowsy")


def variant_flags(variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown model variant: {variant}")
    return {
        "use_multiscale": variant != "single_scale",
        "use_adversarial": variant != "no_adversarial",
        "use_auxiliary": variant != "binary_only",
    }
