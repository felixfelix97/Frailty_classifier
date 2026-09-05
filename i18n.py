"""Static UI strings for experiment_app.py, in English and German.

Only the app CHROME lives here (buttons, labels, headings, fixed copy).
Content that is generated per-patient or per-case (the vignette, the SHAP
explanation, feature labels/values, justification codes) is translated in
its own module instead (case_narrative.py, explanation_text.py,
feature_labels.py, friction_lock.py) - each already owns that content in
English, so the German pair lives right next to it rather than duplicated
here.

Stored/logged values (participant answers, justification codes, experience
levels) are always a stable, language-independent code. Only the on-screen
label is translated - see EXPERIENCE_LEVELS below for the pattern.
"""

LANGUAGES = {"en": "🇬🇧 English", "de": "🇩🇪 Deutsch"}

# Stable codes for the two screening questions, stored as-is in the audit
# log/CSV regardless of display language - so the Chapter 8 regression sees
# one consistent value set no matter which language a participant used.
EXPERIENCE_LEVELS = ["none", "some", "extensive"]

EXPERIENCE_LABELS = {
    "en": {"none": "None", "some": "Some", "extensive": "Extensive"},
    "de": {"none": "Keine", "some": "Etwas", "extensive": "Umfangreich"},
}

# On-screen text for the RYG pill. The underlying code (GREEN/YELLOW/RED)
# stays the CSS class name and the stored/logged value either way.
RYG_LABELS = {
    "en": {"GREEN": "GREEN", "YELLOW": "YELLOW", "RED": "RED"},
    "de": {"GREEN": "GRÜN", "YELLOW": "GELB", "RED": "ROT"},
}

UI = {
    "en": {
        "hero_kicker": "Frailty Risk Screening",
        "hero_title": "🩺 Reviewing an AI Assistant, Together",
        "hero_subtitle": (
            "You'll walk through a set of real patient cases alongside an AI screening "
            "tool. One case at a time, at your own pace. No way to skip ahead or go back "
            "once you've confirmed a decision."
        ),
        "consent_heading": "Before you start",
        "consent_body": (
            "This study records only your task performance: your Accept/Override "
            "decisions, how long you spend reviewing each explanation, and the "
            "clinical and AI-experience answers you provide below. No other personal "
            "data is collected, and no real name is requested. Your participant ID "
            "is a code you choose yourself."
            "<br><br>"
            "Participation is voluntary. You may stop at any time, for any reason, "
            "without any disadvantage to you."
            "<br><br>"
            "Questions about the study can be directed to Felix "
            "(f.felix@campus.tu-berlin.de)."
        ),
        "consent_checkbox_label": (
            "I have read the information above and I voluntarily agree to take part "
            "in this study."
        ),
        "consent_required_note": "Check the box above to continue.",
        "session_advice_heading": "Please do this in one sitting",
        "session_advice_body": (
            "The study takes about 30 to 45 minutes. Please try not to step away "
            "from your device partway through, since long pauses affect the data."
            "\n\n"
            "If you do get disconnected, nothing is lost: enter the **same "
            "Participant ID** again and you will continue exactly where you left "
            "off."
        ),
        "participant_id_label": "Participant ID",
        "participant_id_placeholder": "e.g. P07",
        "clinical_exp_label": "Clinical experience",
        "clinical_exp_help": "Any experience in a clinical, nursing, or care setting (training or work).",
        "ai_exp_label": "AI / machine learning experience",
        "ai_exp_help": "Any experience building, studying, or working with AI or machine learning systems.",
        "exp_detail_placeholder": "Briefly describe (optional)",
        "start_button": "Start session",
        "participant_caption": "Participant",
        "primer_time_note": "~10 minutes · required before case 1",
        "primer_heading": "Before you begin",
        "primer_card1_label": "What is frailty?",
        "primer_card1_text": (
            "Reduced physiological reserve and increased vulnerability to a health "
            "stressor, distinct from normal aging. Patients here are classified as "
            "<b>Frail</b> or <b>Robust</b> from English Longitudinal Study of Ageing "
            "(ELSA) data."
        ),
        "primer_card2_label": "What data does the tool use?",
        "primer_card2_text": (
            "17 factors covering self-rated health, daily-living independence, and "
            "mobility. Full list is below. Its output is a risk color: "
            "<b>Green</b> (low risk of frailty), <b>Yellow</b> (moderate risk), or "
            "<b>Red</b> (high risk). You'll use this same Green/Yellow/Red scale for "
            "your own read of each patient, before the tool's is revealed."
        ),
        "primer_card3_label": "Your task",
        "primer_card3_intro": "You will review {total} patient cases, one at a time, in a fixed random order.",
        "primer_card3_li1": "The tool shows a risk color (Green / Yellow / Red) from its own prediction.",
        "primer_card3_li2": (
            "Yellow and Red cases lock the screen until you review the tool's explanation "
            "for that specific patient."
        ),
        "primer_card3_li3": (
            "You then Accept or Override the tool's assessment. Override requires "
            "selecting a reason from a fixed list."
        ),
        "primer_warn_text": (
            "There is no way to skip a case or go back once you confirm a decision. The "
            "tool's prediction is sometimes wrong. Your task is to use the explanation to "
            "catch it when it is."
        ),
        "primer_factor_list_label": "Full list of the 17 factors the model uses",
        "factor_table_col_factor": "Factor",
        "factor_table_col_desc": "What it measures",
        "primer_begin_button": "I have read this - begin case 1",
        "case_progress": "Case {n} of {total}",
        "judgment_heading": "Your assessment",
        "judgment_caption": "Before the tool's result is revealed, give your own quick read based on the case above: Green if you think they're likely robust (low risk), Yellow if you're unsure or think it's moderate risk, Red if you think they're likely frail (high risk).",
        "judgment_label": "How would you classify this patient?",
        "judgment_submit": "See the tool's assessment",
        "your_read_label": "Your read",
        "ai_read_label": "AI's assessment",
        "primer_card3_li0": "First, you give your own quick Green / Yellow / Red read of the patient, before seeing the tool's assessment.",
        "patient_case_subheader": "Patient case",
        "difficulty_caption": (
            "Reported difficulties across self-care, independent living, and mobility: "
            "**{count} of {max}** possible items."
        ),
        "show_all_fields_label": "Show all 17 underlying data fields",
        "field_table_col_feature": "Feature",
        "field_table_col_value": "Value",
        "explanation_subheader": "Why the tool reached this assessment",
        "explanation_remaining": "The remaining {n} factors had smaller effects on this case.",
        "shap_expander_label": "Show the full SHAP breakdown and model performance",
        "shap_section_title": "SHAP feature attribution, all 17 factors",
        "shap_caption": (
            "Bars to the right push the assessment toward frail; bars to the left push it "
            "toward robust. Longer bars had more influence on this case."
        ),
        "prob_section_title": "Predicted probability for this patient",
        "prob_metric_label": "Probability of frailty",
        "prob_metric_help": "The model's raw output for this patient, before it is banded into a colour.",
        "class_metrics_title": "Class metrics, measured across the whole dataset",
        "recall_label": "Recall on Frail",
        "recall_help": "Of all genuinely frail patients, the share the model correctly identifies.",
        "precision_label": "Precision on Frail",
        "precision_help": "When the model says frail, how often it is right.",
        "rocauc_label": "ROC-AUC",
        "rocauc_help": "How well the model separates the two groups overall. 0.5 is chance, 1.0 is perfect.",
        "risk_signal_caption": "Risk signal: p(Frail) = {p:.3f}",
        "lock_warning": (
            "🔒 **Positive friction lock engaged.** This case is high-risk (Yellow/Red). "
            "Review the explanation below before you can accept or override the AI's "
            "assessment."
        ),
        "review_checkbox": "I have reviewed the explanation above",
        "lock_info": "Accept / Override unlocks once you've reviewed the explanation.",
        "reviewed_success": "✅ Reviewed. Engagement time: {secs:.1f}s",
        "low_risk_caption": "🟩 Low risk. No friction lock required.",
        "decision_label": "Accept or override the AI's assessment",
        "decision_accept": "Accept",
        "decision_override": "Override",
        "justification_label": "Justification code (required for override)",
        "notes_label": "Additional notes (optional)",
        "confirm_button": "Confirm decision",
        "complete_kicker": "Session Complete",
        "complete_title": "Thank you, {name} 🎉",
        "complete_subtitle": "You reviewed all {total} cases. Your decisions are recorded below.",
        "sona_credit_heading": "SONA participants: one last step to receive your VP-Stunde",
        "sona_credit_body": (
            "SONA cannot detect on its own that you finished this study, so credit "
            "has to be granted manually. To receive your **1 VP-Stunde**, please "
            "email a **screenshot of this page** together with your **Participant "
            "ID** to **f.felix@campus.tu-berlin.de**. Your credit will be granted "
            "as soon as the email arrives."
        ),
        "sona_credit_id_label": "Your Participant ID (include this in the email)",
        "stat_cases": "Cases reviewed",
        "stat_overrides": "Overrides",
        "stat_minutes": "Time reviewing explanations",
        "download_button": "⬇ Download session CSV",
        "csv_caption": (
            "This CSV is the raw unit the Chapter 8 mixed-effects regression consumes: "
            "one row per decision, with engagement time. Hidden ground truth (true label "
            "/ AI-error flag) is joined on separately by the researcher via "
            "`case_set.score_decisions()`. Never shown in-session, so it can't influence "
            "the participant's decisions."
        ),
        "feedback_label": "One last thing: how was the experience?",
        "feedback_help": "Anything about the tool, the explanations, or the task itself - what worked, what didn't, what was confusing or frustrating.",
        "feedback_placeholder": "Your feedback (optional, but genuinely useful to us)",
        "feedback_submit": "Submit feedback",
        "feedback_thanks": "Thanks, your feedback has been recorded.",
    },
    "de": {
        "hero_kicker": "Gebrechlichkeits-Risikoscreening",
        "hero_title": "🩺 Gemeinsam ein KI-Assistenzsystem prüfen",
        "hero_subtitle": (
            "Sie gehen eine Reihe echter Patientenfälle gemeinsam mit einem "
            "KI-Screening-Tool durch. Ein Fall nach dem anderen, in Ihrem eigenen Tempo. "
            "Sobald eine Entscheidung bestätigt ist, kann sie nicht mehr übersprungen "
            "oder rückgängig gemacht werden."
        ),
        "consent_heading": "Bevor Sie beginnen",
        "consent_body": (
            "Diese Studie erfasst ausschließlich Ihre Aufgabenleistung: Ihre "
            "Annahme-/Überstimmungsentscheidungen, die Zeit, die Sie für die Prüfung "
            "jeder Erklärung aufwenden, sowie die unten stehenden Antworten zu Ihrer "
            "klinischen und KI-Erfahrung. Es werden keine weiteren personenbezogenen "
            "Daten erhoben, und es wird kein echter Name abgefragt. Ihre "
            "Teilnehmer-ID ist ein von Ihnen selbst gewählter Code."
            "<br><br>"
            "Die Teilnahme ist freiwillig. Sie können jederzeit und ohne Angabe von "
            "Gründen abbrechen, ohne dass Ihnen dadurch Nachteile entstehen."
            "<br><br>"
            "Fragen zur Studie richten Sie bitte an Felix "
            "(f.felix@campus.tu-berlin.de)."
        ),
        "consent_checkbox_label": (
            "Ich habe die obenstehenden Informationen gelesen und nehme freiwillig "
            "an dieser Studie teil."
        ),
        "consent_required_note": "Aktivieren Sie das Kästchen oben, um fortzufahren.",
        "session_advice_heading": "Bitte in einem Durchgang bearbeiten",
        "session_advice_body": (
            "Die Studie dauert etwa 30 bis 45 Minuten. Bitte vermeiden Sie längere "
            "Pausen zwischendurch, da diese die Daten beeinflussen."
            "\n\n"
            "Falls die Verbindung unterbrochen wird, geht nichts verloren: Geben Sie "
            "einfach dieselbe **Teilnehmer-ID** erneut ein, dann machen Sie genau "
            "dort weiter, wo Sie aufgehört haben."
        ),
        "participant_id_label": "Teilnehmer-ID",
        "participant_id_placeholder": "z. B. P07",
        "clinical_exp_label": "Klinische Erfahrung",
        "clinical_exp_help": "Jegliche Erfahrung in einem klinischen, pflegerischen oder Betreuungsumfeld (Ausbildung oder Berufstätigkeit).",
        "ai_exp_label": "KI-/Machine-Learning-Erfahrung",
        "ai_exp_help": "Jegliche Erfahrung im Aufbau, Studium oder in der Arbeit mit KI- oder Machine-Learning-Systemen.",
        "exp_detail_placeholder": "Kurz beschreiben (optional)",
        "start_button": "Sitzung starten",
        "participant_caption": "Teilnehmer/in",
        "primer_time_note": "~10 Minuten · erforderlich vor Fall 1",
        "primer_heading": "Bevor Sie beginnen",
        "primer_card1_label": "Was ist Gebrechlichkeit?",
        "primer_card1_text": (
            "Verminderte physiologische Reserve und erhöhte Anfälligkeit gegenüber einem "
            "gesundheitlichen Stressfaktor, unterscheidbar von normaler Alterung. "
            "Patient:innen werden hier anhand von Daten der English Longitudinal Study "
            "of Ageing (ELSA) als <b>gebrechlich</b> oder <b>robust</b> eingestuft."
        ),
        "primer_card2_label": "Welche Daten nutzt das Tool?",
        "primer_card2_text": (
            "17 Faktoren zu selbsteingeschätzter Gesundheit, Selbstständigkeit im Alltag "
            "und Beweglichkeit. Die vollständige Liste finden Sie unten. Die Ausgabe ist "
            "eine Risikofarbe: <b>Grün</b> (niedriges Risiko für Gebrechlichkeit), "
            "<b>Gelb</b> (mittleres Risiko) oder <b>Rot</b> (hohes Risiko). Dieselbe "
            "Grün-/Gelb-/Rot-Skala verwenden Sie für Ihre eigene Einschätzung jedes "
            "Falls, bevor die Einschätzung des Tools angezeigt wird."
        ),
        "primer_card3_label": "Ihre Aufgabe",
        "primer_card3_intro": "Sie werden {total} Patientenfälle nacheinander in einer festgelegten, zufälligen Reihenfolge durchgehen.",
        "primer_card3_li1": "Das Tool zeigt anhand seiner eigenen Vorhersage eine Risikofarbe (Grün / Gelb / Rot) an.",
        "primer_card3_li2": (
            "Bei Gelb- und Rot-Fällen ist der Bildschirm gesperrt, bis Sie die Erklärung "
            "des Tools für diesen Fall geprüft haben."
        ),
        "primer_card3_li3": (
            "Anschließend akzeptieren oder überstimmen Sie die Einschätzung des Tools. "
            "Beim Überstimmen muss ein Grund aus einer festen Liste ausgewählt werden."
        ),
        "primer_warn_text": (
            "Es gibt keine Möglichkeit, einen Fall zu überspringen oder zurückzugehen, "
            "sobald Sie eine Entscheidung bestätigt haben. Die Vorhersage des Tools ist "
            "manchmal falsch. Ihre Aufgabe ist es, dies anhand der Erklärung zu erkennen."
        ),
        "primer_factor_list_label": "Vollständige Liste der 17 vom Modell genutzten Faktoren",
        "factor_table_col_factor": "Faktor",
        "factor_table_col_desc": "Was er misst",
        "primer_begin_button": "Gelesen - Fall 1 beginnen",
        "case_progress": "Fall {n} von {total}",
        "judgment_heading": "Ihre Einschätzung",
        "judgment_caption": "Bevor das Ergebnis des Tools angezeigt wird, geben Sie Ihre eigene schnelle Einschätzung anhand des obigen Falls ab: Grün, wenn Sie die Person für wahrscheinlich robust halten (niedriges Risiko), Gelb bei Unsicherheit oder mittlerem Risiko, Rot, wenn Sie die Person für wahrscheinlich gebrechlich halten (hohes Risiko).",
        "judgment_label": "Wie würden Sie diesen Fall einstufen?",
        "judgment_submit": "Einschätzung des Tools ansehen",
        "your_read_label": "Ihre Einschätzung",
        "ai_read_label": "Einschätzung der KI",
        "primer_card3_li0": "Zuerst geben Sie Ihre eigene schnelle Grün-/Gelb-/Rot-Einschätzung des Falls ab, bevor Sie die Einschätzung des Tools sehen.",
        "patient_case_subheader": "Patientenfall",
        "difficulty_caption": (
            "Berichtete Schwierigkeiten bei Selbstversorgung, selbstständiger "
            "Lebensführung und Beweglichkeit: **{count} von {max}** möglichen Punkten."
        ),
        "show_all_fields_label": "Alle 17 zugrunde liegenden Datenfelder anzeigen",
        "field_table_col_feature": "Merkmal",
        "field_table_col_value": "Wert",
        "explanation_subheader": "Warum das Tool zu dieser Einschätzung kam",
        "explanation_remaining": "Die übrigen {n} Faktoren hatten bei diesem Fall einen geringeren Einfluss.",
        "shap_expander_label": "Vollständige SHAP-Aufschlüsselung und Modellleistung anzeigen",
        "shap_section_title": "SHAP-Merkmalszuordnung, alle 17 Faktoren",
        "shap_caption": (
            "Balken nach rechts verschieben die Einschätzung Richtung gebrechlich; Balken "
            "nach links Richtung robust. Längere Balken hatten mehr Einfluss auf diesen Fall."
        ),
        "prob_section_title": "Vorhergesagte Wahrscheinlichkeit für diese Person",
        "prob_metric_label": "Wahrscheinlichkeit für Gebrechlichkeit",
        "prob_metric_help": "Die Rohausgabe des Modells für diese Person, bevor sie einer Farbe zugeordnet wird.",
        "class_metrics_title": "Klassenkennzahlen, gemessen über den gesamten Datensatz",
        "recall_label": "Recall bei Gebrechlich",
        "recall_help": "Von allen tatsächlich gebrechlichen Patient:innen der Anteil, den das Modell korrekt erkennt.",
        "precision_label": "Precision bei Gebrechlich",
        "precision_help": "Wenn das Modell „gebrechlich\" meldet, wie oft es damit richtig liegt.",
        "rocauc_label": "ROC-AUC",
        "rocauc_help": "Wie gut das Modell die beiden Gruppen insgesamt trennt. 0,5 ist Zufall, 1,0 ist perfekt.",
        "risk_signal_caption": "Risikosignal: p(Gebrechlich) = {p:.3f}",
        "lock_warning": (
            "🔒 **Positive Friktionssperre aktiv.** Dieser Fall ist hochriskant "
            "(Gelb/Rot). Prüfen Sie die Erklärung unten, bevor Sie die Einschätzung der "
            "KI akzeptieren oder überstimmen können."
        ),
        "review_checkbox": "Ich habe die obige Erklärung geprüft",
        "lock_info": "Akzeptieren/Überstimmen wird freigeschaltet, sobald Sie die Erklärung geprüft haben.",
        "reviewed_success": "✅ Geprüft. Bearbeitungszeit: {secs:.1f}s",
        "low_risk_caption": "🟩 Niedriges Risiko. Keine Friktionssperre erforderlich.",
        "decision_label": "Einschätzung der KI akzeptieren oder überstimmen",
        "decision_accept": "Akzeptieren",
        "decision_override": "Überstimmen",
        "justification_label": "Begründungscode (bei Überstimmen erforderlich)",
        "notes_label": "Zusätzliche Notizen (optional)",
        "confirm_button": "Entscheidung bestätigen",
        "complete_kicker": "Sitzung abgeschlossen",
        "complete_title": "Danke, {name} 🎉",
        "complete_subtitle": "Sie haben alle {total} Fälle geprüft. Ihre Entscheidungen sind unten aufgezeichnet.",
        "sona_credit_heading": "SONA-Teilnehmende: ein letzter Schritt für Ihre VP-Stunde",
        "sona_credit_body": (
            "SONA kann nicht selbst erkennen, dass Sie diese Studie abgeschlossen "
            "haben, daher muss die Gutschrift manuell erfolgen. Um Ihre "
            "**1 VP-Stunde** zu erhalten, senden Sie bitte einen **Screenshot "
            "dieser Seite** zusammen mit Ihrer **Teilnehmer-ID** per E-Mail an "
            "**f.felix@campus.tu-berlin.de**. Die Gutschrift erfolgt, sobald die "
            "E-Mail eingegangen ist."
        ),
        "sona_credit_id_label": "Ihre Teilnehmer-ID (bitte in der E-Mail angeben)",
        "stat_cases": "Geprüfte Fälle",
        "stat_overrides": "Überstimmungen",
        "stat_minutes": "Zeit mit Erklärungen verbracht",
        "download_button": "⬇ Sitzungs-CSV herunterladen",
        "csv_caption": (
            "Diese CSV ist die Rohdatenbasis für die gemischte Regression in Kapitel 8: "
            "eine Zeile pro Entscheidung, mit Bearbeitungszeit. Die verborgene Ground "
            "Truth (wahres Label / KI-Fehler-Kennzeichnung) wird separat von der "
            "Forschungsperson über `case_set.score_decisions()` verknüpft. Wird während "
            "der Sitzung nie angezeigt, damit sie die Entscheidungen der Teilnehmenden "
            "nicht beeinflussen kann."
        ),
        "feedback_label": "Noch eine letzte Sache: Wie war die Erfahrung?",
        "feedback_help": "Alles zum Tool, den Erklärungen oder der Aufgabe selbst - was gut funktioniert hat, was nicht, was verwirrend oder frustrierend war.",
        "feedback_placeholder": "Ihr Feedback (optional, aber für uns wirklich hilfreich)",
        "feedback_submit": "Feedback absenden",
        "feedback_thanks": "Danke, Ihr Feedback wurde aufgezeichnet.",
    },
}


def t(lang, key, **kwargs):
    """Look up one UI string in the given language and format it."""
    text = UI.get(lang, UI["en"]).get(key, UI["en"].get(key, key))
    return text.format(**kwargs) if kwargs else text
