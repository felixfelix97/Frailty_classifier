"""Human-readable labels and value translations for the 17 ELSA model features.

Built from the same authoritative Harmonized ELSA codebook citations already
verified in dict_lookup.txt / read_frailty_data.ipynb - this module just
surfaces that work in the UI, so a participant without gerontology/ELSA
background can read "Self-rated health: Bad (4/5)" instead of "r6shlt: 4".

Raw column names remain the data/model contract everywhere else in the
codebase (frailty_pipeline.py, ODD_RANGES, etc.) - this module is purely a
display-layer translation, not a data transformation.

Bilingual (English/German, per the experiment's TU Berlin participant pool):
label/desc/unit/values are all nested under "en"/"de". Every accessor takes
a lang argument defaulting to "en", so existing non-experiment callers
(streamlit_app.py) are unaffected.
"""

import pandas as pd

FEATURE_META = {
    "r6agey": {
        "kind": "numeric",
        "label": {"en": "Age", "de": "Alter"},
        "unit": {"en": "years", "de": "Jahre"},
        "desc": {
            "en": "The patient's age in years.",
            "de": "Das Alter der Patientin bzw. des Patienten in Jahren.",
        },
    },
    "raracem": {
        "kind": "categorical",
        "label": {"en": "Race", "de": "Ethnische Zugehörigkeit"},
        "values": {
            "en": {1: "White", 4: "Non-white"},
            "de": {1: "Weiß", 4: "Nicht-weiß"},
        },
        "desc": {
            "en": "Race, as self-reported by the patient.",
            "de": "Ethnische Zugehörigkeit, wie von der Person selbst angegeben.",
        },
    },
    "raeduc_e": {
        "kind": "categorical",
        "label": {"en": "Education level", "de": "Bildungsniveau"},
        "values": {
            "en": {1: "Less than high school", 3: "High school graduate",
                   4: "Some college", 5: "College and above"},
            "de": {1: "Ohne Abschluss (unterhalb Highschool)", 3: "Highschool-Abschluss",
                   4: "Teilweise Hochschulbildung", 5: "Hochschulabschluss oder höher"},
        },
        "desc": {
            "en": "Highest education level completed, on a US-style scale.",
            "de": "Höchster erreichter Bildungsabschluss, nach US-amerikanischer Einteilung.",
        },
    },
    "raeducl": {
        "kind": "categorical",
        "label": {"en": "Education (harmonized)", "de": "Bildung (harmonisiert)"},
        "values": {
            "en": {1: "Less than upper secondary", 2: "Upper secondary / vocational",
                   3: "Tertiary"},
            "de": {1: "Unterhalb Sekundarstufe II", 2: "Sekundarstufe II / Berufsausbildung",
                   3: "Tertiärbildung (Hochschule)"},
        },
        "desc": {
            "en": "Highest education level completed, on a 3-tier scale comparable "
                  "across countries.",
            "de": "Höchster erreichter Bildungsabschluss, auf einer international "
                  "vergleichbaren 3-stufigen Skala.",
        },
    },
    "rarelig_e": {
        "kind": "categorical",
        "label": {"en": "Religion", "de": "Religion"},
        "values": {
            "en": {1: "Christian", 2: "Buddhist", 3: "Hindu", 4: "Jewish",
                   5: "Muslim", 6: "Sikh", 7: "Other non-Christian", 8: "None"},
            "de": {1: "Christlich", 2: "Buddhistisch", 3: "Hinduistisch", 4: "Jüdisch",
                   5: "Muslimisch", 6: "Sikh", 7: "Andere nicht-christliche Religion", 8: "Keine"},
        },
        "desc": {
            "en": "Religious affiliation, as self-reported by the patient.",
            "de": "Religionszugehörigkeit, wie von der Person selbst angegeben.",
        },
    },
    "r6shlt": {
        "kind": "categorical",
        "label": {"en": "Self-rated health", "de": "Selbsteingeschätzter Gesundheitszustand"},
        "values": {
            "en": {1: "Very good", 2: "Good", 3: "Fair", 4: "Bad", 5: "Very bad"},
            "de": {1: "Sehr gut", 2: "Gut", 3: "Mittelmäßig", 4: "Schlecht", 5: "Sehr schlecht"},
        },
        "desc": {
            "en": "The patient's own rating of their general health, from Very good "
                  "to Very bad.",
            "de": "Die eigene Einschätzung des allgemeinen Gesundheitszustands, von "
                  "Sehr gut bis Sehr schlecht.",
        },
    },
    "r6hlthlm": {
        "kind": "binary",
        "label": {"en": "Health limits work", "de": "Gesundheit schränkt Arbeit ein"},
        "values": {"en": {0: "No", 1: "Yes"}, "de": {0: "Nein", 1: "Ja"}},
        "desc": {
            "en": "Whether a health problem limits the kind or amount of work the "
                  "patient can do.",
            "de": "Ob ein gesundheitliches Problem Art oder Umfang der möglichen "
                  "Arbeit einschränkt.",
        },
    },
    "r6eata": {
        "kind": "binary",
        "label": {"en": "Difficulty eating", "de": "Schwierigkeiten beim Essen"},
        "values": {
            "en": {0: "No difficulty", 1: "Some difficulty"},
            "de": {0: "Keine Schwierigkeiten", 1: "Einige Schwierigkeiten"},
        },
        "desc": {
            "en": "Whether the patient has difficulty eating without help.",
            "de": "Ob die Person Schwierigkeiten hat, ohne Hilfe zu essen.",
        },
    },
    "r6adlwa": {
        "kind": "count", "max": 3,
        "label": {
            "en": "Basic daily-living difficulties (short version, 3 tasks)",
            "de": "Schwierigkeiten bei grundlegenden Alltagsaktivitäten (Kurzversion, 3 Aufgaben)",
        },
        "desc": {
            "en": "Number of these 3 basic self-care tasks the patient has difficulty "
                  "with: bathing, dressing, and eating.",
            "de": "Anzahl dieser 3 grundlegenden Alltagsaufgaben, bei denen die Person "
                  "Schwierigkeiten hat: Baden, Anziehen und Essen.",
        },
    },
    "r6adlwaa": {
        "kind": "binary",
        "label": {
            "en": "Any basic daily-living difficulty (short version)",
            "de": "Irgendeine grundlegende Alltagsschwierigkeit (Kurzversion)",
        },
        "values": {"en": {0: "No", 1: "Yes"}, "de": {0: "Nein", 1: "Ja"}},
        "desc": {
            "en": "Whether the patient has difficulty with any of those 3 basic "
                  "self-care tasks at all.",
            "de": "Ob die Person bei irgendeiner dieser 3 grundlegenden "
                  "Alltagsaufgaben überhaupt Schwierigkeiten hat.",
        },
    },
    "r6adla": {
        "kind": "count", "max": 5,
        "label": {
            "en": "Basic daily-living difficulties (full version, 5 tasks)",
            "de": "Schwierigkeiten bei grundlegenden Alltagsaktivitäten (Vollversion, 5 Aufgaben)",
        },
        "desc": {
            "en": "Number of these 5 basic self-care tasks the patient has difficulty "
                  "with: bathing, dressing, eating, getting in or out of bed, and "
                  "walking across a room.",
            "de": "Anzahl dieser 5 grundlegenden Alltagsaufgaben, bei denen die Person "
                  "Schwierigkeiten hat: Baden, Anziehen, Essen, Aufstehen aus dem Bett "
                  "und Gehen durch einen Raum.",
        },
    },
    "r6iadla": {
        "kind": "count", "max": 3,
        "label": {
            "en": "Independent-living difficulties (short version, 3 tasks)",
            "de": "Schwierigkeiten bei selbstständiger Lebensführung (Kurzversion, 3 Aufgaben)",
        },
        "desc": {
            "en": "Number of these 3 independent-living tasks the patient has "
                  "difficulty with: using a telephone, managing money, and taking "
                  "medications.",
            "de": "Anzahl dieser 3 Aufgaben der selbstständigen Lebensführung, bei "
                  "denen die Person Schwierigkeiten hat: Telefonieren, Geld verwalten "
                  "und Medikamente einnehmen.",
        },
    },
    "r6iadlza": {
        "kind": "count", "max": 5,
        "label": {
            "en": "Independent-living difficulties (full version, 5 tasks)",
            "de": "Schwierigkeiten bei selbstständiger Lebensführung (Vollversion, 5 Aufgaben)",
        },
        "desc": {
            "en": "Number of these 5 independent-living tasks the patient has "
                  "difficulty with: using a telephone, managing money, taking "
                  "medications, shopping, and preparing meals.",
            "de": "Anzahl dieser 5 Aufgaben der selbstständigen Lebensführung, bei "
                  "denen die Person Schwierigkeiten hat: Telefonieren, Geld "
                  "verwalten, Medikamente einnehmen, Einkaufen und Kochen.",
        },
    },
    "r6iadlzaa": {
        "kind": "binary",
        "label": {
            "en": "Any independent-living difficulty",
            "de": "Irgendeine Schwierigkeit bei selbstständiger Lebensführung",
        },
        "values": {"en": {0: "No", 1: "Yes"}, "de": {0: "Nein", 1: "Ja"}},
        "desc": {
            "en": "Whether the patient has difficulty with any of those 5 "
                  "independent-living tasks at all.",
            "de": "Ob die Person bei irgendeiner dieser 5 Aufgaben der "
                  "selbstständigen Lebensführung überhaupt Schwierigkeiten hat.",
        },
    },
    "r6grossa": {
        "kind": "count", "max": 5,
        "label": {
            "en": "Everyday movement difficulties (walk / climb / bathe)",
            "de": "Schwierigkeiten bei alltäglichen Bewegungen (Gehen / Treppensteigen / Baden)",
        },
        "desc": {
            "en": "Number of these 5 everyday movement tasks the patient has "
                  "difficulty with: walking one block, walking across a room, "
                  "climbing one flight of stairs, bathing, and getting in or out of "
                  "bed.",
            "de": "Anzahl dieser 5 alltäglichen Bewegungsaufgaben, bei denen die "
                  "Person Schwierigkeiten hat: einen Häuserblock gehen, durch einen "
                  "Raum gehen, eine Treppe steigen, baden und aus dem Bett aufstehen.",
        },
    },
    "r6lowermob": {
        "kind": "count", "max": 4,
        "label": {
            "en": "Lower-body mobility difficulties",
            "de": "Beweglichkeitsschwierigkeiten der unteren Körperhälfte",
        },
        "desc": {
            "en": "Number of these 4 lower-body tasks the patient has difficulty "
                  "with: walking several blocks, walking one block, sitting for two "
                  "hours, and getting up from a chair.",
            "de": "Anzahl dieser 4 Aufgaben der unteren Körperhälfte, bei denen die "
                  "Person Schwierigkeiten hat: mehrere Häuserblocks gehen, einen "
                  "Häuserblock gehen, zwei Stunden sitzen und von einem Stuhl "
                  "aufstehen.",
        },
    },
    "r6uppermob": {
        "kind": "count", "max": 3,
        "label": {
            "en": "Upper-body mobility difficulties",
            "de": "Beweglichkeitsschwierigkeiten der oberen Körperhälfte",
        },
        "desc": {
            "en": "Number of these 3 upper-body tasks the patient has difficulty "
                  "with: reaching or extending the arms, pushing or pulling large "
                  "objects, and lifting weights.",
            "de": "Anzahl dieser 3 Aufgaben der oberen Körperhälfte, bei denen die "
                  "Person Schwierigkeiten hat: die Arme ausstrecken, große "
                  "Gegenstände schieben oder ziehen und Gewichte heben.",
        },
    },
}


def label_for(col, lang="en"):
    """Plain-English (or German) feature name, falling back to the raw
    column name."""
    meta = FEATURE_META.get(col)
    if not meta:
        return col
    return meta["label"].get(lang, meta["label"]["en"])


def describe_feature(col, lang="en"):
    """One-sentence plain-language explanation of what this factor measures,
    for the primer's full-factor-list expander - not what this patient's
    value is, but what the factor itself covers."""
    meta = FEATURE_META.get(col)
    if not meta:
        return ""
    return meta["desc"].get(lang, meta["desc"]["en"])


def describe_value(col, val, lang="en"):
    """Human-readable rendering of one (column, value) pair."""
    meta = FEATURE_META.get(col)
    if meta is None or val is None:
        return str(val)
    if pd.isna(val):
        return "missing" if lang == "en" else "fehlt"

    kind = meta["kind"]
    if kind in ("categorical", "binary"):
        try:
            code = int(val)
        except (TypeError, ValueError):
            return f"code {val}"
        values = meta["values"].get(lang, meta["values"]["en"])
        return values.get(code, f"code {val}")
    if kind == "count":
        of_word = "of" if lang == "en" else "von"
        try:
            return f"{int(val)} {of_word} {meta['max']}"
        except (TypeError, ValueError):
            return str(val)
    if kind == "numeric":
        unit = meta.get("unit", {}).get(lang, "")
        try:
            return f"{val:.0f} {unit}".strip()
        except (TypeError, ValueError):
            return str(val)
    return str(val)
