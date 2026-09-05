"""Plain-language rendering of the model's SHAP explanation, for participants.

Companion to case_narrative.py: that module makes the PATIENT readable, this
one makes the MODEL'S REASONING readable. Both exist for the same reason -
participants are Human Factors Master's students with no ML or gerontology
background, and an explanation they cannot parse cannot be engaged with.

This matters more than presentation polish. The friction lock (L1, Art. 14)
holds the interface until the explanation panel is reviewed, and the study's
dependent variable is the time spent on that panel ("time-on-XAI-panel",
expose Section 5.1). If the panel shows only signed floats labelled
"shap_value", the lock produces a delay without producing understanding -
which manufactures oversight theater rather than measuring it. Art. 13 also
requires the reasoning be intelligible to the user, which a raw SHAP table
arguably is not for this population.

Nothing here changes the model, the SHAP computation, or the timer. The
underlying values are unchanged and remain visible in a collapsible table;
this module only converts them into sentences.

Strength is expressed RELATIVE to the strongest factor in the same case,
not against fixed absolute cutoffs: SHAP values are in log-odds units whose
absolute scale is not meaningful to a non-specialist, whereas "this factor
mattered most, that one barely mattered" is.

Bilingual (English/German): every function takes a lang argument.
"""

RYG_MEANING = {
    "en": {"GREEN": "low risk", "YELLOW": "moderate risk", "RED": "high risk"},
    "de": {"GREEN": "niedriges Risiko", "YELLOW": "mittleres Risiko", "RED": "hohes Risiko"},
}

_CONFIDENCE_BANDS_EN = [
    (0.80, "The tool is **highly confident** this patient is frail"),
    (0.60, "The tool considers this patient **likely frail**"),
    (0.50, "The tool leans toward **frail, but is close to undecided**"),
    (0.35, "The tool leans toward **robust, but is close to undecided**"),
    (0.20, "The tool considers this patient **likely robust**"),
    (0.0, "The tool is **confident this patient is robust**"),
]
_CONFIDENCE_BANDS_DE = [
    (0.80, "Das Tool ist **sehr sicher**, dass diese Person gebrechlich ist"),
    (0.60, "Das Tool hält diese Person **wahrscheinlich für gebrechlich**"),
    (0.50, "Das Tool tendiert zu **gebrechlich, ist sich aber nicht sicher**"),
    (0.35, "Das Tool tendiert zu **robust, ist sich aber nicht sicher**"),
    (0.20, "Das Tool hält diese Person **wahrscheinlich für robust**"),
    (0.0, "Das Tool ist **sicher, dass diese Person robust ist**"),
]


def confidence_sentence(p_frail, ryg, lang="en"):
    """One sentence putting the raw probability into words."""
    de = lang == "de"
    band = RYG_MEANING.get(lang, RYG_MEANING["en"]).get(ryg, ryg.lower())
    bands = _CONFIDENCE_BANDS_DE if de else _CONFIDENCE_BANDS_EN
    strength = next(text for cutoff, text in bands if p_frail >= cutoff)
    # Avoid rendering a near-zero or near-one probability as a flat "0%" /
    # "100%", which reads as false certainty rather than a rounded estimate.
    if p_frail < 0.01:
        pct = "unter 1%" if de else "under 1%"
    elif p_frail > 0.99:
        pct = "über 99%" if de else "over 99%"
    else:
        pct = f"{p_frail:.0%}"
    if de:
        return (
            f"{strength}. Es schätzt die Wahrscheinlichkeit für Gebrechlichkeit auf "
            f"**{pct}**, was es als **{band}** einstuft."
        )
    return (
        f"{strength}. It puts the chance of frailty at **{pct}**, "
        f"which it classifies as **{band}**."
    )


def _strength_word(shap_value, max_abs, lang="en"):
    de = lang == "de"
    if max_abs <= 0:
        return "minimal" if de else "minimally"
    ratio = abs(shap_value) / max_abs
    if ratio >= 0.75:
        return "stark" if de else "strongly"
    if ratio >= 0.40:
        return "mäßig" if de else "moderately"
    if ratio >= 0.15:
        return "leicht" if de else "slightly"
    return "minimal" if de else "minimally"


def explanation_sentences(shap_df, top_n=5, lang="en"):
    """Convert a SHAP DataFrame into readable statements.

    Expects the columns produced by frailty_pipeline.explain_case() after
    the app has added readable labels: 'feature' (plain-English name),
    'what it means' (translated value), and 'shap_value'.

    Returns (top_statements, remaining_count).
    """
    if shap_df.empty:
        return [], 0
    de = lang == "de"
    max_abs = shap_df["shap_value"].abs().max()
    statements = []
    for _, r in shap_df.head(top_n).iterrows():
        direction = ("gebrechlich" if de else "frail") if r["shap_value"] > 0 else ("robust" if de else "robust")
        strength = _strength_word(r["shap_value"], max_abs, lang=lang)
        if de:
            statements.append(
                f"**{r['feature']}: {r['what it means']}**. Wirkt sich {strength} auf **{direction}** aus"
            )
        else:
            statements.append(
                f"**{r['feature']}: {r['what it means']}**. Pushes {strength} toward **{direction}**"
            )
    return statements, max(0, len(shap_df) - top_n)


PANEL_INTRO = {
    "en": (
        "The tool reached its assessment by weighing all 17 patient factors. "
        "The factors below are the ones that influenced this particular case the most, "
        "listed from most to least influential. Each line states which way that factor "
        "pushed the assessment, and how much."
    ),
    "de": (
        "Das Tool ist zu seiner Einschätzung gekommen, indem es alle 17 "
        "Patientenfaktoren abgewogen hat. Die unten aufgeführten Faktoren haben "
        "diesen Fall am stärksten beeinflusst, sortiert vom einflussreichsten zum "
        "am wenigsten einflussreichen. Jede Zeile zeigt, in welche Richtung dieser "
        "Faktor die Einschätzung beeinflusst hat und wie stark."
    ),
}


def reliability_note(recall_frail, precision_frail, lang="en"):
    """Plain-language version of the class metrics the expose's interface
    diagram requires the panel to show ("feature attribution, probability,
    class metrics").

    Computed from the model's own measured performance rather than written
    as fixed text, so the claim on screen cannot drift away from the model
    if it is retrained or retuned.
    """
    if lang == "de":
        return (
            f"**Wie zuverlässig ist dieses Tool?** Über den gesamten Datensatz "
            f"erkennt es korrekt **{recall_frail:.0%} der tatsächlich gebrechlichen "
            f"Patient:innen**. Es löst auch eine beachtliche Anzahl an Fehlalarmen "
            f"aus: Wenn es *gebrechlich* meldet, liegt es in etwa "
            f"**{precision_frail:.0%}** der Fälle richtig. Es ist bewusst so "
            f"eingestellt, dass es gebrechliche Patient:innen möglichst nicht "
            f"übersieht, auf Kosten häufigerer Fehlalarme."
        )
    return (
        f"**How reliable is this tool?** Across the whole dataset it correctly identifies "
        f"**{recall_frail:.0%} of genuinely frail patients**. It also raises a fair number of false "
        f"alarms: when it says *frail*, it is right about **{precision_frail:.0%}** of the time. It is "
        f"deliberately set up to avoid missing frail patients, at the cost of over-flagging."
    )


CLASS_METRICS_INTRO = {
    "en": (
        "These figures describe the tool's accuracy across the whole dataset, not this individual "
        "patient. They are shown so the reviewer knows how much weight the assessment deserves."
    ),
    "de": (
        "Diese Kennzahlen beschreiben die Genauigkeit des Tools über den gesamten "
        "Datensatz, nicht bei dieser einzelnen Person. Sie werden gezeigt, damit "
        "die prüfende Person weiß, wie viel Gewicht der Einschätzung zukommt."
    ),
}
