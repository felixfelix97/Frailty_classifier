"""L5 MLOps & Security: safe-state fallback controller (Art. 15).

Distinct from the L4 ODD gate's per-case safe-state (a single input has
an out-of-range/missing value - see frailty_pipeline.py::validate_odd).
This controller triggers a SYSTEM- or PROFILE-level safe-state instead:
either the model's operating environment has drifted (drift_detector.py)
- meaning predictions may not be trustworthy even for individually valid
inputs - or a specific input was flagged as adversarial/statistically
implausible by adversarial_sanitizer.py despite passing every per-field
ODD bound. In either case, inference is bypassed entirely: no RYG, no
SHAP, no prediction is shown.
"""

from enum import Enum


class SafeStateReason(Enum):
    NONE = "NONE"
    DRIFT = "DRIFT"                                 # system-level: population has shifted
    ADVERSARIAL_REJECTED = "ADVERSARIAL_REJECTED"    # sanitizer rejected the input outright
    ADVERSARIAL_NOVEL = "ADVERSARIAL_NOVEL"          # sanitizer flagged statistically implausible


def check_safe_state(sanitizer_result, drift_status):
    """Decide whether to bypass inference entirely.

    sanitizer_result: dict from adversarial_sanitizer.sanitize_and_score()
    drift_status: str from drift_detector.overall_drift_status()
      ('SAFE' / 'MONITOR' / 'DRIFT')

    Returns {safe_state: bool, reason: SafeStateReason, detail: str}.
    Drift takes priority over the sanitizer result: it's a system-wide
    signal, and moot to report a single input's novelty if the whole
    population has already shifted.
    """
    if drift_status == "DRIFT":
        return {
            "safe_state": True,
            "reason": SafeStateReason.DRIFT,
            "detail": (
                "Population-level distribution shift detected (L5 drift monitor). "
                "Model predictions are not considered reliable until this is reviewed."
            ),
        }
    if sanitizer_result["status"] == "REJECTED":
        return {
            "safe_state": True,
            "reason": SafeStateReason.ADVERSARIAL_REJECTED,
            "detail": f"Input rejected by the adversarial sanitizer: {'; '.join(sanitizer_result['problems'])}",
        }
    if sanitizer_result["status"] == "NOVEL":
        return {
            "safe_state": True,
            "reason": SafeStateReason.ADVERSARIAL_NOVEL,
            "detail": (
                f"Input flagged as statistically implausible (novelty score "
                f"{sanitizer_result['novelty_score']:.1f}), despite passing every per-field ODD bound."
            ),
        }
    return {"safe_state": False, "reason": SafeStateReason.NONE, "detail": ""}
