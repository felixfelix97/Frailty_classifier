"""Turns a row of 17 ELSA features into a readable patient vignette.

Why this exists, beyond feature_labels.py: that module makes each individual
value readable ("r6shlt=4" -> "Self-rated health: Bad"). This module makes
the row as a whole mean something - a list of 17 translated ratings still
does not communicate a patient, and a participant who cannot form their own
impression of the patient has no basis on which to disagree with the AI.
That is a construct-validity problem, not a presentation one: catch rate
(the study's key measure) is only meaningful if the participant can reach an
independent judgement to compare against the model's.

Participants are Human Factors Master's students (TU Berlin) with no
gerontology background, so the vignette groups the raw features into the
three clinical domains frailty is actually assessed on, states counts in
plain language, and names what each scale covers.

Deliberate constraints, so the vignette informs without deciding for them:
- Strictly factual. No evaluative words ("severe", "concerning", "frail"),
  no summary judgement, no hint of the true label or the model's output.
- Population anchors are given per domain (what a typical respondent in this
  dataset reports) so a count has a reference point. Anchors describe the
  cohort, never this patient.
- Every statement is derived from the row itself; nothing is invented.

Scale contents are the standard Harmonized ELSA constructs already verified
against the codebook for this project (see dict_lookup.txt).

Bilingual (English/German): narrative_for() takes a lang argument. Section
headings are returned as stable English keys (SECTION_KEYS) rather than
translated text, so callers (experiment_app.py's icon lookup) can key off
something that doesn't change with language - use section_label(key, lang)
to get the on-screen heading text.
"""

import pandas as pd

from feature_labels import describe_value

SECTION_KEYS = ["patient", "health", "selfcare", "independent", "mobility"]

SECTION_LABELS = {
    "en": {
        "patient": "Patient",
        "health": "How they describe their own health",
        "selfcare": "Basic self-care (activities of daily living)",
        "independent": "Living independently (managing a household)",
        "mobility": "Mobility and physical function",
    },
    "de": {
        "patient": "Patient:in",
        "health": "Wie die Person ihre eigene Gesundheit einschätzt",
        "selfcare": "Grundlegende Selbstversorgung (Aktivitäten des täglichen Lebens)",
        "independent": "Selbstständig leben (Haushaltsführung)",
        "mobility": "Beweglichkeit und körperliche Funktion",
    },
}


def section_label(key, lang="en"):
    return SECTION_LABELS.get(lang, SECTION_LABELS["en"]).get(key, key)


# What each summed index actually covers - so "3 of 5" means something.
SCALE_CONTENT = {
    "en": {
        "r6adla": "bathing, dressing, eating, getting in or out of bed, and walking across a room",
        "r6adlwa": "bathing, dressing, and eating",
        "r6iadla": "using a telephone, managing money, and taking medications",
        "r6iadlza": "using a telephone, managing money, taking medications, shopping, and preparing meals",
        "r6grossa": "walking one block, walking across a room, climbing one flight of stairs, bathing, and getting in or out of bed",
        "r6lowermob": "walking several blocks, walking one block, sitting for two hours, and getting up from a chair",
        "r6uppermob": "reaching or extending the arms, pushing or pulling large objects, and lifting weights",
    },
    "de": {
        "r6adla": "Baden, Anziehen, Essen, Aufstehen aus dem Bett und Gehen durch einen Raum",
        "r6adlwa": "Baden, Anziehen und Essen",
        "r6iadla": "Telefonieren, Geld verwalten und Medikamente einnehmen",
        "r6iadlza": "Telefonieren, Geld verwalten, Medikamente einnehmen, Einkaufen und Kochen",
        "r6grossa": "einen Häuserblock gehen, durch einen Raum gehen, eine Treppe steigen, baden und aus dem Bett aufstehen",
        "r6lowermob": "mehrere Häuserblocks gehen, einen Häuserblock gehen, zwei Stunden sitzen und von einem Stuhl aufstehen",
        "r6uppermob": "die Arme ausstrecken, große Gegenstände schieben oder ziehen und Gewichte heben",
    },
}

# Item count each scale is out of - shared by _count_phrase() and the
# combined mobility sentence in narrative_for().
SCALE_MAX = {"r6adla": 5, "r6adlwa": 3, "r6iadla": 3, "r6iadlza": 5,
             "r6grossa": 5, "r6lowermob": 4, "r6uppermob": 3}

MOBILITY_AREAS = {
    "en": {"r6grossa": "general movement", "r6lowermob": "walking/standing", "r6uppermob": "reaching/lifting"},
    "de": {"r6grossa": "allgemeine Bewegung", "r6lowermob": "Gehen/Stehen", "r6uppermob": "Greifen/Heben"},
}


def _count_phrase(row, col, lang="en"):
    """'difficulty with 3 of the 5 ...' - or the zero case, stated plainly."""
    val = row[col]
    if pd.isna(val):
        return None
    n = int(val)
    covers = SCALE_CONTENT.get(lang, SCALE_CONTENT["en"]).get(col, "")
    total = SCALE_MAX[col]
    if lang == "de":
        if n == 0:
            return f"Berichtet **keine Schwierigkeiten** bei den {total} Aufgaben: {covers}."
        return f"Berichtet Schwierigkeiten bei **{n} von {total}** Aufgaben: {covers}."
    if n == 0:
        return f"Reports **no difficulty** with any of the {total} items covering {covers}."
    return f"Reports difficulty with **{n} of the {total}** items covering {covers}."


def narrative_for(row, reference_df=None, lang="en"):
    """Build the vignette. Returns a list of (section_key, [statements]).

    reference_df, when given, is used only for cohort-level anchors (medians
    across the dataset) - never to describe this patient. Use section_label()
    to turn a section_key into the on-screen heading for the given lang.
    """
    sections = []
    de = lang == "de"

    # ---- who ----
    who = []
    age = row.get("r6agey")
    if pd.notna(age):
        who.append(f"**{int(age)} Jahre alt.**" if de else f"**{int(age)} years old.**")
    edu = describe_value("raeducl", row.get("raeducl"), lang=lang)
    if edu and edu != "None":
        who.append(f"Höchster Bildungsabschluss: {edu}." if de else f"Highest education level: {edu}.")
    sections.append(("patient", who))

    # ---- how they rate their own health ----
    # Cohort anchor only shown when it adds information: skip it when this
    # patient's own rating already matches the typical one, since restating
    # the norm back at the reader is filler, not context - and at 20 cases
    # per session, filler compounds into real fatigue.
    health = []
    shlt_val = row.get("r6shlt")
    shlt = describe_value("r6shlt", shlt_val, lang=lang)
    if shlt:
        health.append(
            f"Beschreibt die eigene allgemeine Gesundheit als **{shlt.lower()}**." if de else
            f"Describes their own general health as **{shlt.lower()}**."
        )
    lim = row.get("r6hlthlm")
    if pd.notna(lim):
        if de:
            health.append(
                "Gibt an, dass ein gesundheitliches Problem Art oder Umfang der "
                "möglichen Arbeit **einschränkt**."
                if int(lim) == 1 else
                "Gibt an, dass ein gesundheitliches Problem Art oder Umfang der "
                "möglichen Arbeit **nicht einschränkt**."
            )
        else:
            health.append(
                "Says a health problem **does** limit the kind or amount of work they can do."
                if int(lim) == 1 else
                "Says a health problem **does not** limit the kind or amount of work they can do."
            )
    if reference_df is not None and "r6shlt" in reference_df and pd.notna(shlt_val):
        med = reference_df["r6shlt"].median()
        if int(shlt_val) != round(med):
            med_label = describe_value("r6shlt", med, lang=lang).lower()
            health.append(
                f"_Typische Einschätzung in diesem Datensatz: „{med_label}\"._" if de else
                f"_Typical rating in this dataset: '{med_label}'._"
            )
    sections.append(("health", health))

    # ---- basic self-care ----
    # Only the 5-item scale is narrated: the 3-item short version is a strict
    # subset of it, so stating both repeats the same fact in two forms and
    # reads as clutter. Both remain visible in the full data table.
    adl = [s for s in (_count_phrase(row, "r6adla", lang=lang),) if s]
    eat = row.get("r6eata")
    if pd.notna(eat) and int(eat) == 1:
        adl.append(
            "Konkret: hat **einige Schwierigkeiten beim Essen** ohne Hilfe." if de else
            "Specifically, has **some difficulty eating** without help."
        )
    adl_val = row.get("r6adla")
    if reference_df is not None and "r6adla" in reference_df and pd.notna(adl_val) and int(adl_val) > 0:
        share = (reference_df["r6adla"] == 0).mean()
        adl.append(
            f"_{share:.0%} der Befragten berichten keinerlei Schwierigkeiten bei der Selbstversorgung._" if de else
            f"_{share:.0%} of respondents report no self-care difficulty at all._"
        )
    sections.append(("selfcare", adl))

    # ---- independent living ----
    iadl = [s for s in (_count_phrase(row, "r6iadlza", lang=lang),) if s]
    iadl_val = row.get("r6iadlza")
    if reference_df is not None and "r6iadlza" in reference_df and pd.notna(iadl_val) and int(iadl_val) > 0:
        share = (reference_df["r6iadlza"] == 0).mean()
        iadl.append(
            f"_{share:.0%} der Befragten berichten hier keine Schwierigkeiten._" if de else
            f"_{share:.0%} of respondents report no difficulty here._"
        )
    sections.append(("independent", iadl))

    # ---- mobility ----
    # Three related scales (gross motor, lower-body, upper-body) collapsed
    # into one combined sentence instead of three near-identical ones - same
    # information, a third of the reading.
    areas = MOBILITY_AREAS.get(lang, MOBILITY_AREAS["en"])
    parts = []
    for col in ("r6grossa", "r6lowermob", "r6uppermob"):
        v = row.get(col)
        if pd.notna(v):
            of_word = "von" if de else "of"
            parts.append(f"{areas[col]} ({int(v)} {of_word} {SCALE_MAX[col]})")
    mob = []
    if parts:
        mob.append(
            f"Berichtete Schwierigkeiten nach Bereich: {'; '.join(parts)}." if de else
            f"Reported difficulties, by area: {'; '.join(parts)}."
        )
    lowermob_val = row.get("r6lowermob")
    if reference_df is not None and "r6lowermob" in reference_df and pd.notna(lowermob_val) and int(lowermob_val) > 0:
        share = (reference_df["r6lowermob"] == 0).mean()
        mob.append(
            f"_{share:.0%} der Befragten berichten keine Schwierigkeiten beim Gehen/Stehen._" if de else
            f"_{share:.0%} of respondents report no walking/standing difficulty._"
        )
    sections.append(("mobility", mob))

    return sections


def total_difficulty_count(row):
    """Sum of reported difficulties across self-care, independent living, and
    mobility. A single orienting number, stated without interpretation."""
    cols = ["r6adla", "r6iadlza", "r6grossa", "r6lowermob", "r6uppermob"]
    total = 0
    max_total = 5 + 5 + 5 + 4 + 3
    for c in cols:
        v = row.get(c)
        if pd.notna(v):
            total += int(v)
    return total, max_total
