"""Full pipeline orchestration: input -> L5 sanitizer -> L5 safe-state ->
L4 ODD -> L2 inference -> L1 RYG.

A single function used by both streamlit_app.py and the integration test,
so the live app and the test exercise exactly the same code path rather
than two implementations that could silently drift apart.
"""

from adversarial_sanitizer import sanitize_and_score
from frailty_pipeline import classify_case
from safe_state import check_safe_state


def process_case(row, model, reference_df, novelty_thresh, drift_status="SAFE"):
    """Run one case through the full pipeline.

    Returns a dict:
    - stage: 'SAFE_STATE' (L5 bypass - drift or adversarial input),
             'ODD_INVALID' (L4 bypass - out-of-range/missing field), or
             'SCORED' (reached inference; classification has odd_status/ryg/p_frail).
    - sanitizer: raw result from adversarial_sanitizer.sanitize_and_score()
    - safe_state: raw result from safe_state.check_safe_state()
    - classification: raw result from frailty_pipeline.classify_case(), or
      None if the case never reached that stage (SAFE_STATE bypass).
    """
    sanitizer_result = sanitize_and_score(row, reference_df, novelty_thresh)
    safe_state_result = check_safe_state(sanitizer_result, drift_status)
    if safe_state_result["safe_state"]:
        return {
            "stage": "SAFE_STATE",
            "sanitizer": sanitizer_result,
            "safe_state": safe_state_result,
            "classification": None,
        }

    classification = classify_case(row, model)
    stage = "ODD_INVALID" if classification["odd_status"] == "INVALID" else "SCORED"
    return {
        "stage": stage,
        "sanitizer": sanitizer_result,
        "safe_state": safe_state_result,
        "classification": classification,
    }
