"""L5 MLOps & Security: adversarial input sanitizer (Art. 15).

Runs BEFORE the L4 ODD gate. The ODD validator (frailty_pipeline.py)
assumes numeric-comparable input - `val < lo` on a non-numeric value
raises an exception rather than a clean rejection. Two lines of defense
here:

1. Type/format sanitization - reject non-numeric values, injection-like
   strings, Inf, or implausible magnitudes, before they ever reach a
   numeric comparison.
2. Multivariate novelty score - a row can pass every single-feature ODD
   bound and still be a statistically implausible combination (e.g.
   crafted to probe a decision-boundary blind spot). ODD only checks
   fields one at a time and can't see cross-feature implausibility; this
   is a distinct signal.
"""

import re

import numpy as np
import pandas as pd

_INJECTION_PATTERN = re.compile(r"[;'\"<>{}]|--|/\*|\bDROP\b|\bSELECT\b|\bscript\b", re.IGNORECASE)
_MAX_PLAUSIBLE_MAGNITUDE = 1e6  # beyond this, a value isn't a real clinical measurement


def sanitize_value(col, val):
    """Return (clean_value, problem). problem is None if the value is clean,
    else a short string naming exactly what's wrong with it."""
    if val is None:
        return np.nan, None
    if isinstance(val, str):
        if _INJECTION_PATTERN.search(val):
            return None, f"{col}: rejected suspicious string pattern"
        try:
            val = float(val)
        except ValueError:
            return None, f"{col}: not numeric ('{val}')"
    try:
        val = float(val)
    except (TypeError, ValueError):
        return None, f"{col}: not numeric ({val!r})"
    if np.isinf(val):
        return None, f"{col}: infinite value"
    if abs(val) > _MAX_PLAUSIBLE_MAGNITUDE:
        return None, f"{col}: implausible magnitude ({val})"
    return val, None


def sanitize_row(row):
    """Sanitize every field in a row (pandas Series). Returns
    (clean_row_or_None, problems) - each problem names the exact field and
    why. If ANY field fails, the whole row is rejected: a partially
    sanitized row isn't a safe thing to hand to a clinical classifier."""
    clean = {}
    problems = []
    for col, val in row.items():
        clean_val, problem = sanitize_value(col, val)
        if problem:
            problems.append(problem)
        else:
            clean[col] = clean_val
    if problems:
        return None, problems
    return pd.Series(clean), []


def novelty_score(row, reference_df):
    """Sum of absolute z-scores across features, relative to the reference
    (training) distribution. A crude but effective multivariate novelty
    signal, independent of per-feature ODD bounds."""
    means = reference_df.mean()
    stds = reference_df.std().replace(0, np.nan)
    z = ((row[reference_df.columns] - means) / stds).abs()
    return float(z.fillna(0).sum())


def novelty_threshold(reference_df, percentile=99.5):
    """Calibrate the 'too novel' cutoff from the reference data itself: the
    novelty score at the given percentile of the training set."""
    scores = reference_df.apply(lambda r: novelty_score(r, reference_df), axis=1)
    return float(np.percentile(scores, percentile))


def sanitize_and_score(row, reference_df, threshold):
    """Full pipeline: type/format sanitization, then novelty scoring.
    Returns a dict: {status: 'CLEAN'|'REJECTED'|'NOVEL', problems, novelty_score}.
    """
    clean_row, problems = sanitize_row(row)
    if clean_row is None:
        return {"status": "REJECTED", "problems": problems, "novelty_score": None}
    score = novelty_score(clean_row, reference_df)
    if score > threshold:
        return {"status": "NOVEL", "problems": [], "novelty_score": score}
    return {"status": "CLEAN", "problems": [], "novelty_score": score}
