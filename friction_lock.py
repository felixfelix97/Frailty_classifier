"""L1 Interface: positive friction lock, engagement timer, override justification.

Per the expose (Section 4.3.1): when the AI classifies a case as high-risk
(Red or Yellow on the RYG scale), the interface locks until the participant
engages with the explainability panel. The session-state timer captures
`time_on_friction_screen` - the engagement proxy this thesis measures (the
time between the lock appearing and the review checkbox being ticked).
"""

from datetime import datetime, timezone

# Standardized override justification codes (predefined dropdown, Section
# 4.3.1). Provisional set - not specified in the expose itself, so flagged
# here as adjustable pending supervisor input, not a fixed requirement.
#
# These English strings are the canonical, STORED values (audit log + CSV),
# regardless of the participant's display language - so the Chapter 8
# analysis sees one consistent code set either way. JUSTIFICATION_LABELS_DE
# supplies only the on-screen German text; justification_label() below is
# what the UI should call to display a code, never the raw code itself.
JUSTIFICATION_CODES = [
    "Clinical judgment contradicts AI output",
    "Suspected data quality issue (incomplete or inaccurate input)",
    "Patient-specific context not captured by the model (e.g. recent acute event)",
    "Known model limitation for this patient subgroup",
    "AI explanation unconvincing / evidence insufficient",
    "Other (see notes)",
]

JUSTIFICATION_LABELS_DE = {
    "Clinical judgment contradicts AI output": "Klinische Einschätzung widerspricht der KI-Ausgabe",
    "Suspected data quality issue (incomplete or inaccurate input)": "Vermutetes Datenqualitätsproblem (unvollständige oder ungenaue Eingabe)",
    "Patient-specific context not captured by the model (e.g. recent acute event)": "Patientenspezifischer Kontext, den das Modell nicht erfasst (z. B. kürzliches akutes Ereignis)",
    "Known model limitation for this patient subgroup": "Bekannte Modellgrenze für diese Patientengruppe",
    "AI explanation unconvincing / evidence insufficient": "KI-Erklärung nicht überzeugend / Beweislage unzureichend",
    "Other (see notes)": "Sonstiges (siehe Notizen)",
}


def justification_label(code, lang="en"):
    """On-screen text for a justification code. The code itself (English)
    is always what gets stored - this is display only."""
    if lang == "de":
        return JUSTIFICATION_LABELS_DE.get(code, code)
    return code


def requires_lock(ryg):
    """Yellow or Red locks the interface; Green does not (Section 4.3.1)."""
    return ryg in ("YELLOW", "RED")


def build_decision_record(case_id, ryg, p_frail, engagement_seconds, decision,
                           justification_code=None, justification_text=""):
    """One record per participant decision - the unit the L3 logger and the
    eventual experiment CSV export (Week 7-8) will both consume."""
    if decision == "override" and not justification_code:
        raise ValueError("justification_code is required when decision is 'override'")
    return {
        "case_id": case_id,
        "ryg": ryg,
        "p_frail": p_frail,
        "engagement_seconds": round(engagement_seconds, 3) if engagement_seconds is not None else None,
        "decision": decision,
        "justification_code": justification_code,
        "justification_text": justification_text,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
