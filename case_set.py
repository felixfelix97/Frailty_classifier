"""Week 7-8: the real participant case set (expose Section 5.3) - built once,
reused for every participant, with a HIDDEN true label attached per case so
catch rate can be scored later. Distinct from rubberstamp_simulation.py,
which is a synthetic dry run of the *analysis method*, not the real case set
participants will actually see.

~30-40% of cases are real AI errors (model prediction != true label, both on
real ELSA data, no synthetic cases), stratified into three confidence bands
so the set spans "clearly to subtly incorrect predictions" (Section 5.3):
- 'clear':    AI confidently wrong (|p_frail - 0.5| large)   - easiest to catch
- 'moderate': in between
- 'subtle':   AI wrong but close to the decision boundary    - hardest to catch
The remaining cases are AI-correct, so a participant who always agrees with
the AI doesn't trivially "win."

Every case is pre-filtered to ODD-VALID *and* below the L5 novelty threshold,
so every case in the fixed set reliably reaches the SCORED pipeline stage in
the experiment app - no participant should ever hit a safe-state bypass
mid-session purely because of which row happened to get sampled.

The case set itself is FIXED (same N cases, same true labels, same
case_number 1..N) across every participant - only the PRESENTATION ORDER is
randomized per participant (Section 5.3), via a seed derived from the
participant ID so the order is reproducible without needing to store it.
"""

import hashlib

import numpy as np
import pandas as pd

from adversarial_sanitizer import novelty_score, novelty_threshold
from frailty_pipeline import ODD_RANGES, classify_all, load_data, train_model

# 20, not 30: set during the design phase per expose Section 5.2 ("the exact
# number of cases per participant will be determined during experiment design
# phase, depending on dataset availability and session constraints"). A
# researcher dry run of the 30-case version found it fatiguing enough to risk
# inducing rubberstamping from exhaustion rather than from genuine
# disengagement - which would contaminate the very behaviour this study
# measures. 20 cases yields 7 AI-error cases per participant (the unit catch
# rate is computed over), i.e. 84-105 error-case decisions at the target
# n=12-15 participants - enough for the mixed-effects model, at roughly
# two-thirds the session length.
DEFAULT_N_CASES = 20
DEFAULT_AI_ERROR_FRACTION = 0.35  # per Section 5.3: ~30-40%
DEFAULT_SEED = 42
ERROR_BANDS = ["subtle", "moderate", "clear"]


def build_case_set(n_cases=DEFAULT_N_CASES, ai_error_fraction=DEFAULT_AI_ERROR_FRACTION,
                    seed=DEFAULT_SEED):
    """Build the fixed participant case set.

    Returns (case_set_df, X, model):
    - case_set_df: one row per case -
        case_number   1..n_cases, the case's permanent identity
        row_index     row label into X/y
        true_label    1=Frail / 0=Robust  (HIDDEN from participants)
        ai_p_frail    model's p(Frail) on this case
        ai_predicted  1/0, the AI's own call
        is_ai_error   bool                (HIDDEN from participants)
        error_band    'subtle'/'moderate'/'clear'/'n/a'  (HIDDEN)
    - X: full feature matrix (row_index indexes into this)
    - model: the model used to build the set (reuse this in the experiment
      app so displayed predictions match exactly - retraining again with the
      same seed would give the same model, but reusing avoids paying for it
      twice and any incidental nondeterminism).
    """
    rng = np.random.default_rng(seed)
    X, y = load_data()
    model = train_model(X, y, random_state=seed)

    flags = classify_all(X, model, ODD_RANGES)
    p_frail = flags["p_frail"]
    ai_predicted = (p_frail >= 0.5).astype("Int64")

    odd_valid = flags["odd_status"] == "VALID"
    novelty_thresh = novelty_threshold(X)
    not_novel = X.apply(lambda r: novelty_score(r, X), axis=1) <= novelty_thresh
    eligible = odd_valid & not_novel

    is_ai_error = eligible & (ai_predicted != y)
    is_ai_correct = eligible & (ai_predicted == y)

    error_idx = X.index[is_ai_error]
    correct_idx = X.index[is_ai_correct]

    n_error = int(round(n_cases * ai_error_fraction))
    n_correct = n_cases - n_error
    if n_error > len(error_idx) or n_correct > len(correct_idx):
        raise ValueError(
            f"Not enough eligible cases: need {n_error} AI-error / {n_correct} "
            f"AI-correct, have {len(error_idx)} / {len(correct_idx)} available."
        )

    chosen_error, chosen_bands = _sample_error_bands(error_idx, p_frail, n_error, rng)
    chosen_correct = rng.choice(correct_idx, size=n_correct, replace=False)

    rows = []
    for row_idx, band in zip(chosen_error, chosen_bands):
        rows.append(_case_row(row_idx, y, p_frail, ai_predicted, is_ai_error=True, error_band=band))
    for row_idx in chosen_correct:
        rows.append(_case_row(row_idx, y, p_frail, ai_predicted, is_ai_error=False, error_band="n/a"))

    case_set = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    case_set.insert(0, "case_number", range(1, len(case_set) + 1))
    return case_set, X, model


def _case_row(row_idx, y, p_frail, ai_predicted, is_ai_error, error_band):
    return {
        "row_index": row_idx,
        "true_label": int(y.loc[row_idx]),
        "ai_p_frail": float(p_frail.loc[row_idx]),
        "ai_predicted": int(ai_predicted.loc[row_idx]),
        "is_ai_error": bool(is_ai_error),
        "error_band": error_band,
    }


def _sample_error_bands(error_idx, p_frail, n_error, rng):
    """Split AI-error cases into 3 confidence bands (by |p_frail - 0.5|,
    ascending) and sample as evenly as possible across them, so the error
    set spans the full "clearly to subtly incorrect" spectrum rather than
    clustering at one extreme."""
    confidence = (p_frail.loc[error_idx] - 0.5).abs().sort_values()
    bands = np.array_split(confidence.index.to_numpy(), 3)

    per_band = n_error // 3
    remainder = n_error - per_band * 3
    chosen, chosen_bands = [], []
    for i, band_pool in enumerate(bands):
        take = min(per_band + (1 if i < remainder else 0), len(band_pool))
        picked = rng.choice(band_pool, size=take, replace=False)
        chosen.extend(picked)
        chosen_bands.extend([ERROR_BANDS[i]] * take)

    shortfall = n_error - len(chosen)
    if shortfall > 0:
        remaining = np.setdiff1d(error_idx.to_numpy(), np.array(chosen))
        extra = rng.choice(remaining, size=shortfall, replace=False)
        chosen.extend(extra)
        chosen_bands.extend(["moderate"] * shortfall)
    return chosen, chosen_bands


def presentation_order(participant_id, case_set):
    """Deterministic per-participant shuffle of case_number (Section 5.3:
    randomized order per participant, same underlying case set). Seeded from
    a SHA-256 digest of the participant ID rather than Python's built-in
    hash(), which is randomized per-process (PYTHONHASHSEED) and would give
    a different order every time the app restarts for the same participant.
    """
    digest = hashlib.sha256(str(participant_id).encode("utf-8")).hexdigest()
    seed = int(digest, 16) % (2**32)
    rng = np.random.default_rng(seed)
    order = case_set["case_number"].to_numpy().copy()
    rng.shuffle(order)
    return order.tolist()


def score_decisions(decisions_df, case_set):
    """Join a participant's decisions (case_number, decision) against the
    hidden case_set labels and add per-decision scoring columns:
    - caught_error: True if this was a real AI-error case AND the
      participant overrode it (a genuine catch)
      - unnecessary_override: True if the AI was actually correct AND the
      participant overrode it anyway
    Returns the merged DataFrame - the long-format table the mixed-effects
    regression (Chapter 8) consumes."""
    merged = decisions_df.merge(
        case_set[["case_number", "true_label", "is_ai_error", "error_band", "ai_p_frail"]],
        left_on="case_id", right_on="case_number", how="left",
    )
    # A left join that matched nothing leaves NaN here, which silently turns
    # is_ai_error into a float column and makes the boolean scoring below
    # fail with an unreadable "bad operand type for unary ~" further down.
    # Fail here instead, naming the actual unmatched rows.
    unmatched = merged["is_ai_error"].isna()
    if unmatched.any():
        bad = merged.loc[unmatched, "case_id"].tolist()
        raise ValueError(
            f"{unmatched.sum()} decision(s) reference case_ids not present in the "
            f"case set: {bad}. Either non-decision records (consent/feedback) leaked "
            f"into decisions_df, or these decisions came from a different case set."
        )
    merged["is_ai_error"] = merged["is_ai_error"].astype(bool)
    overrode = merged["decision"].str.lower() == "override"
    merged["caught_error"] = merged["is_ai_error"] & overrode
    merged["unnecessary_override"] = (~merged["is_ai_error"]) & overrode
    return merged


def catch_rate_matrix(scored_df):
    """The 2x2 catch-rate matrix (expose Table 5, Section 5.5):
    rows = AI was actually right/wrong, columns = participant accepted/overrode."""
    table = pd.crosstab(
        scored_df["is_ai_error"].map({True: "AI wrong", False: "AI correct"}),
        scored_df["decision"].str.capitalize(),
    )
    return table.reindex(index=["AI wrong", "AI correct"], fill_value=0)
