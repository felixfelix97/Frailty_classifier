"""Week 7-8: the real participant experiment instrument (expose Section 5).

Distinct from streamlit_app.py (the free-browsing dev/QA scaffold, which
stays as-is for internal testing): this app shows a participant ONE case at
a time, in a fixed random order unique to them, with no way to go back or
skip ahead. It reuses the same L1-L5 pipeline, friction lock, and audit
logger as the scaffold - same underlying system, different presentation
mode - via case_set.py's fixed, hidden-ground-truth case set.

Flow: participant ID -> sequential cases (friction lock + SHAP panel on
Yellow/Red, same as the scaffold) -> completion screen with a CSV download
of every decision + engagement time, ready for the Chapter 8 mixed-effects
regression. The true label / AI-error flag is never shown during the
session - only case_set.py and the researcher's post-hoc scoring
(case_set.score_decisions) ever see it.
"""

import base64
import html
import io
import json
import os
import time

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit.errors import StreamlitSecretNotFoundError

from adversarial_sanitizer import novelty_threshold
from audit_logger import AuditLogger
from case_narrative import narrative_for, section_label, total_difficulty_count
from case_set import DEFAULT_AI_ERROR_FRACTION, DEFAULT_N_CASES, build_case_set, presentation_order
from explanation_text import (
    CLASS_METRICS_INTRO,
    PANEL_INTRO,
    confidence_sentence,
    explanation_sentences,
    reliability_note,
)
from feature_labels import describe_feature, describe_value, label_for
from friction_lock import JUSTIFICATION_CODES, build_decision_record, justification_label, requires_lock
from frailty_pipeline import cv_clinical_metrics, explain_case, load_data
from i18n import EXPERIENCE_LABELS, EXPERIENCE_LEVELS, LANGUAGES, RYG_LABELS, t
from pipeline_flow import process_case
from sheet_sync import append_session_rows

st.set_page_config(page_title="Frailty Risk Screening - Experiment", layout="wide")

# ---- Dataset materialization ----
# Felix_FrailtyData.csv is intentionally NOT committed to git (its
# redistribution license was never confirmed; see docs/DEPLOYMENT_CHECKLIST.md,
# Group 2). Locally it just sits on disk, untouched, and this is a no-op. On a
# deployment with no local copy, write it out from a Streamlit secret instead,
# so the dataset never has to live in the (possibly public) git repository.
# frailty_pipeline.py's load_data() still just reads the plain relative path,
# unaware of where the file actually came from - no change to that contract.
# Stored as a chunked TOML array (dataset_csv_b64_parts), not one long string:
# confirmed live that Streamlit mirrors every secret into an OS environment
# variable, and Windows caps a single env var at 32767 characters - the whole
# base64 dataset (~349KB) blew past that as one value. Chunking each piece
# well under that ceiling avoids depending on which OS the limit does or
# doesn't apply on.
# Checks file SIZE too, not just existence: an earlier deploy attempt could
# leave a zero-byte or truncated file behind (a crash mid-write, a stale
# artifact on the platform's persistent disk), which "exists" but is not
# valid data - self-heals by re-materializing whenever the file looks too
# small to be the real ~255KB dataset, rather than silently trusting it.
_DATASET_PATH = "Felix_FrailtyData.csv"
_dataset_present = os.path.exists(_DATASET_PATH) and os.path.getsize(_DATASET_PATH) > 10_000
try:
    _has_dataset_secret = not _dataset_present and "dataset_csv_b64_parts" in st.secrets
except StreamlitSecretNotFoundError:
    # Raised on ANY st.secrets access (not just a missing key) when zero
    # secrets files exist anywhere - the normal local-dev case, fine to
    # ignore. Deliberately NOT catching broader exceptions here: a malformed
    # secrets file should fail loudly rather than silently skip
    # materialization and resurface later as a confusing parse error on an
    # empty CSV.
    _has_dataset_secret = False
if _has_dataset_secret:
    with open(_DATASET_PATH, "wb") as _f:
        _f.write(base64.b64decode("".join(st.secrets["dataset_csv_b64_parts"])))

# ---- Visual chrome only, below this line. No logic, no timers, no content
# changes live here - the friction-lock explanation panel deliberately keeps
# its existing plain styling (see render_explainability_panel), since it is
# the one screen whose viewing time is the study's dependent variable, and
# adding visual interest there specifically would be a confound, not a
# feature. Everything else - the patient view, the decision moment, the
# session bookends - deliberately breaks from index.html/the supervisor
# deck's plain corporate palette: a participant sits through 20 near-
# identical cases in one sitting, so this screen needs to read as a game-like
# clinical chart (chunky cards, a teal/cream/coral palette) rather than a
# document, to work against the fatigue a dry run already surfaced.
SECTION_ICONS = {
    "patient": "🧑",
    "health": "🩺",
    "selfcare": "🧴",
    "independent": "🏠",
    "mobility": "🚶",
}


def _md_to_html(text):
    """Minimal **bold**/_italic_ -> HTML, for text embedded inside raw HTML
    cards (a markdown parser does not run inside raw HTML blocks, so
    case_narrative.py's markdown-formatted statements need converting)."""
    import re
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<em>\1</em>", text)
    return text

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap');

:root {
  --accent: #0E7C7B; --accent-soft: #DCF2EF;
  --cta: #E8672C; --cta-hover: #CB5620;
  --green: #2E7D5B; --green-soft: #E7F3EC;
  --amber: #B8790A; --amber-soft: #FDF1DA;
  --red: #C23B36; --red-soft: #FCEAE9;
  --ink: #1E2B2A; --ink-soft: #5C716E; --line: #BFE0DB; --paper: #FFFDF8;
}
html, body, [class*="css"] { font-family: 'Inter', -apple-system, "Segoe UI", sans-serif; }
h1, h2, h3 { font-family: 'Baloo 2', 'Segoe UI', sans-serif !important; letter-spacing: -0.01em; }
.stApp { background: linear-gradient(180deg, #EAF6F4 0%, #F5FBF9 340px); }

/* ---- chunky "game panel" card base: flat fill, bold border, offset shadow
   instead of a soft blur - the look this pivot is built around ---- */
.chunk {
  background: var(--paper); border: 2px solid var(--line); border-radius: 16px;
  box-shadow: 4px 4px 0 rgba(14,124,123,0.14);
}

/* ---- landing hero ---- */
.hero-wrap { text-align: center; padding: 48px 12px 12px 12px; }
.hero-wrap .kicker {
  font-size: 0.8rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--accent); margin-bottom: 10px;
}
.hero-wrap h1 { font-size: 2.5rem; margin: 0 0 14px 0; color: var(--ink); }
.hero-wrap p { font-size: 1.05rem; color: var(--ink-soft); max-width: 46ch; margin: 0 auto; }

/* ---- primer step cards, in a 2-up grid so the primer fits in one screen
   instead of a long vertical stack ---- */
.primer-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 14px; }
.primer-grid .full { grid-column: 1 / -1; }
.primer-card {
  background: var(--paper); border: 2px solid var(--line); border-left: 6px solid var(--accent);
  border-radius: 14px; padding: 18px 22px;
  box-shadow: 4px 4px 0 rgba(14,124,123,0.10);
}
.primer-card b.step-label {
  display: inline-block; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--accent); margin-bottom: 6px;
}

/* ---- your-read-vs-AI's-read comparison row ---- */
.compare-row { display: flex; gap: 20px; align-items: flex-start; flex-wrap: wrap; margin: 4px 0 4px 0; }
.compare-item { display: flex; flex-direction: column; gap: 6px; }
.compare-item .compare-lbl {
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--ink-soft);
}

/* ---- RYG badge, a real pill instead of a markdown header ---- */
.ryg-pill {
  display: inline-flex; align-items: center; gap: 10px;
  font-family: 'Baloo 2', sans-serif; font-weight: 700; font-size: 1.5rem;
  padding: 10px 22px; border-radius: 100px; margin: 6px 0 4px 0;
  border: 2px solid currentColor;
}
.ryg-pill.GREEN { background: var(--green-soft); color: var(--green); }
.ryg-pill.YELLOW { background: var(--amber-soft); color: var(--amber); }
.ryg-pill.RED { background: var(--red-soft); color: var(--red); }
.ryg-pill .dot { width: 14px; height: 14px; border-radius: 50%; background: currentColor; }

/* ---- vignette section cards: a 2-column chart-like grid, collapsing to
   one column on narrow/phone screens (see the media query below) ---- */
.vgrid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 8px; }
.vcard {
  background: var(--paper); border: 2px solid var(--line); border-radius: 14px;
  padding: 14px 18px; box-shadow: 4px 4px 0 rgba(14,124,123,0.10);
}
.vcard .vhead {
  display: flex; align-items: center; gap: 8px; font-weight: 700; color: var(--accent);
  margin-bottom: 6px; font-size: 1rem; font-family: 'Baloo 2', sans-serif;
}
.vcard .vhead .icon {
  font-size: 1.1rem; background: var(--accent-soft); border-radius: 8px;
  width: 30px; height: 30px; display: inline-flex; align-items: center; justify-content: center;
}
.vcard ul { margin: 0; padding-left: 20px; }
.vcard li { margin-bottom: 4px; color: var(--ink); }
.vcard li em { color: var(--ink-soft); }

/* ---- segmented Accept/Override control ---- */
div[data-testid="stRadio"] div[role="radiogroup"] {
  display: flex; gap: 10px; flex-wrap: wrap;
}
div[data-testid="stRadio"] div[role="radiogroup"] label {
  border: 2px solid var(--line); border-radius: 12px; padding: 10px 22px !important;
  background: var(--paper); transition: all 0.15s ease; cursor: pointer;
}
div[data-testid="stRadio"] div[role="radiogroup"] label:hover { border-color: var(--accent); }
div[data-testid="stRadio"] div[role="radiogroup"] label[data-selected="true"] {
  border-color: var(--accent); background: var(--accent-soft);
}

/* ---- primary buttons: chunky, coral, "3D press" feel on click ---- */
button[kind="primary"], button[kind="primaryFormSubmit"] {
  border-radius: 12px !important; font-weight: 700 !important;
  box-shadow: 0 4px 0 var(--cta-hover) !important; border: none !important;
  transition: transform 0.05s ease, box-shadow 0.05s ease !important;
}
button[kind="primary"]:active, button[kind="primaryFormSubmit"]:active {
  transform: translateY(3px) !important; box-shadow: 0 1px 0 var(--cta-hover) !important;
}

/* ---- completion stat chips ---- */
.stat-row { display: flex; gap: 16px; flex-wrap: wrap; margin: 18px 0; }
.stat-chip {
  background: var(--paper); border: 2px solid var(--line); border-radius: 14px;
  padding: 14px 22px; min-width: 140px; box-shadow: 4px 4px 0 rgba(14,124,123,0.10);
}
.stat-chip .val { font-family: 'Baloo 2', sans-serif; font-size: 1.8rem; font-weight: 700; color: var(--accent); }
.stat-chip .lbl { font-size: 0.78rem; color: var(--ink-soft); text-transform: uppercase; letter-spacing: 0.04em; }

/* ---- phone / narrow screens: single column everywhere, smaller hero ---- */
@media (max-width: 640px) {
  .vgrid, .primer-grid { grid-template-columns: 1fr; }
  .hero-wrap h1 { font-size: 1.9rem; }
  .hero-wrap { padding: 28px 4px 8px 4px; }
  .chunk, .primer-card, .vcard, .stat-chip { box-shadow: 3px 3px 0 rgba(14,124,123,0.10); }
}

@media (prefers-reduced-motion: reduce) {
  div[data-testid="stRadio"] div[role="radiogroup"] label,
  button[kind="primary"], button[kind="primaryFormSubmit"] { transition: none; }
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_case_set():
    """One fixed case set, shared by every participant on this server
    process (Section 5.3: same underlying cases, only presentation order
    varies per participant)."""
    return build_case_set(n_cases=DEFAULT_N_CASES, ai_error_fraction=DEFAULT_AI_ERROR_FRACTION)


@st.cache_data
def get_novelty_threshold(_X):
    return novelty_threshold(_X)


@st.cache_data
def get_labels():
    """The true labels, needed only to compute the class metrics shown in
    the explanation panel. build_case_set() does not return them, since the
    session itself must never see them."""
    _X, y = load_data()
    return y


@st.cache_data
def get_metrics(_X, _y):
    """Class metrics for the explanation panel, per the expose's interface
    diagram. Cached because this runs full cross-validation, and it is the
    same for every patient and every participant."""
    return cv_clinical_metrics(_X, _y)


@st.cache_resource
def get_audit_logger():
    return AuditLogger("experiment_audit_log.jsonl")


# ---- Admin data-download panel ----
# The audit log lives only on this server's own disk (Streamlit Cloud's free
# tier gives no file browser or SSH access), so this is the only way to
# actually retrieve session data without asking every participant to
# remember to send back their downloaded CSV. Gated behind a token in
# Streamlit secrets, not a hardcoded password, so it never appears in the
# (public) git repo. A participant visiting the plain app URL never sees
# this; only ?admin=<the real token> reaches it, and st.stop() below means
# nothing past this block ever renders for that request.
try:
    _admin_token = st.secrets["admin_token"] if "admin_token" in st.secrets else None
except StreamlitSecretNotFoundError:
    # Same as above: no secrets file at all is the normal local-dev case.
    _admin_token = None
if _admin_token and st.query_params.get("admin") == _admin_token:
    st.title("Admin: session data")
    _entries = get_audit_logger().entries
    _records = [e["record"] for e in _entries]

    # Three record types share one log: decisions (no record_type key),
    # plus "consent" and "session_feedback" markers written at the session
    # bookends. Split them so each download is a clean, flat table rather
    # than one ragged mixed one.
    _decisions = [r for r in _records if "record_type" not in r]
    _feedback = [r for r in _records if r.get("record_type") == "session_feedback"]
    _consents = [r for r in _records if r.get("record_type") == "consent"]
    _participants = sorted({r.get("participant_id") for r in _records if r.get("participant_id")})

    c1, c2, c3 = st.columns(3)
    c1.metric("Participants", len(_participants))
    c2.metric("Decisions logged", len(_decisions))
    c3.metric("Sessions finished", len(_feedback))

    _chain_ok, _problems = get_audit_logger().verify_chain()
    if _chain_ok:
        st.success("Audit chain verified: no tampering detected.")
    else:
        st.error(f"Audit chain verification FAILED: {_problems}")

    if _decisions:
        _dec_df = pd.DataFrame(_decisions)
        st.subheader("All decisions")
        st.dataframe(_dec_df, hide_index=True)
        _buf = io.StringIO()
        _dec_df.to_csv(_buf, index=False)
        st.download_button(
            "Download all decisions (.csv)",
            data=_buf.getvalue(),
            file_name="all_decisions.csv",
            mime="text/csv",
            type="primary",
        )
    else:
        st.info("No decisions recorded yet.")

    if _feedback:
        _fb_df = pd.DataFrame(_feedback)
        st.subheader("Closing feedback and screening answers")
        st.dataframe(_fb_df, hide_index=True)
        _fbuf = io.StringIO()
        _fb_df.to_csv(_fbuf, index=False)
        st.download_button(
            "Download feedback (.csv)",
            data=_fbuf.getvalue(),
            file_name="all_feedback.csv",
            mime="text/csv",
        )

    st.subheader("Raw log")
    st.caption(
        f"{len(_entries)} entries total ({len(_consents)} consents, "
        f"{len(_decisions)} decisions, {len(_feedback)} feedback). The .jsonl "
        "keeps the hash chain intact and is what compute_catch_rate.py reads."
    )
    st.download_button(
        "Download full audit log (.jsonl)",
        data="\n".join(json.dumps(e) for e in _entries),
        file_name="experiment_audit_log.jsonl",
        mime="application/x-ndjson",
    )
    st.stop()

case_set, X, model = get_case_set()
y = get_labels()
novelty_thresh = get_novelty_threshold(X)

if "participant_id" not in st.session_state:
    st.session_state.participant_id = None
if "language" not in st.session_state:
    st.session_state.language = "en"

if st.session_state.participant_id is None:
    _, lang_col, _ = st.columns([1, 1.4, 1])
    with lang_col:
        st.radio(
            "Language", list(LANGUAGES.keys()), format_func=lambda code: LANGUAGES[code],
            horizontal=True, key="language", label_visibility="collapsed",
        )
    lang = st.session_state.language

    st.markdown(
        f"""
        <div class="hero-wrap">
          <div class="kicker">{t(lang, "hero_kicker")}</div>
          <h1>{t(lang, "hero_title")}</h1>
          <p>{t(lang, "hero_subtitle")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        # Deliberately says nothing about how fast to answer any single case:
        # engagement time on the explanation panel is the study's dependent
        # variable, so telling participants to hurry would compress exactly
        # the measure being collected. It targets the real causes instead,
        # stepping away mid-session (idle time recorded as engagement) and
        # abandoning after a disconnect (not knowing the session resumes).
        # Added 2026-09-04 on Prof. Hughes's instruction, with the instrument
        # otherwise left unchanged and idle inflation handled at analysis.
        st.warning(
            f"**{t(lang, 'session_advice_heading')}**\n\n{t(lang, 'session_advice_body')}"
        )

        with st.container(border=True):
            st.markdown(f"**{t(lang, 'consent_heading')}**")
            st.markdown(t(lang, "consent_body"), unsafe_allow_html=True)
            st.checkbox(t(lang, "consent_checkbox_label"), key="consent_given")

        pid = st.text_input(t(lang, "participant_id_label"), placeholder=t(lang, "participant_id_placeholder"))

        st.markdown(f"**{t(lang, 'clinical_exp_label')}**")
        st.caption(t(lang, "clinical_exp_help"))
        st.radio(
            t(lang, "clinical_exp_label"), EXPERIENCE_LEVELS,
            format_func=lambda code: EXPERIENCE_LABELS[lang][code],
            horizontal=True, key="clinical_experience", label_visibility="collapsed",
        )
        st.text_area(t(lang, "exp_detail_placeholder"), key="clinical_experience_detail",
                     label_visibility="collapsed", placeholder=t(lang, "exp_detail_placeholder"))

        st.markdown(f"**{t(lang, 'ai_exp_label')}**")
        st.caption(t(lang, "ai_exp_help"))
        st.radio(
            t(lang, "ai_exp_label"), EXPERIENCE_LEVELS,
            format_func=lambda code: EXPERIENCE_LABELS[lang][code],
            horizontal=True, key="ai_experience", label_visibility="collapsed",
        )
        st.text_area(t(lang, "exp_detail_placeholder"), key="ai_experience_detail",
                     label_visibility="collapsed", placeholder=t(lang, "exp_detail_placeholder"))

    _, mid2, _ = st.columns([1, 1.4, 1])
    with mid2:
        consent_given = st.session_state.get("consent_given", False)
        if not consent_given and pid.strip():
            st.caption(t(lang, "consent_required_note"))
        if st.button(
            t(lang, "start_button"),
            disabled=not pid.strip() or not consent_given,
            use_container_width=True, type="primary",
        ):
            pid = pid.strip()
            st.session_state.participant_id = pid
            st.session_state.order = presentation_order(pid, case_set)
            get_audit_logger().append({
                "record_type": "consent",
                "participant_id": pid,
                "language": st.session_state.language,
                "consent_given": True,
            })

            # Snapshot into non-widget keys before leaving the landing screen:
            # once a widget (key="language" etc.) stops being rendered on later
            # reruns, Streamlit drops its session_state entry entirely (this is
            # documented Streamlit behavior, not a bug) - so reading
            # st.session_state.language on the case screen would silently fall
            # back to the .get() default instead of the participant's actual
            # choice. Caught live: a full German session rendered the case
            # screen in English because of exactly this.
            st.session_state.confirmed_language = st.session_state.language
            st.session_state.confirmed_clinical_experience = st.session_state.clinical_experience
            st.session_state.confirmed_clinical_experience_detail = st.session_state.clinical_experience_detail
            st.session_state.confirmed_ai_experience = st.session_state.ai_experience
            st.session_state.confirmed_ai_experience_detail = st.session_state.ai_experience_detail

            # Resume from the audit log rather than always starting at position
            # 0: a real participant's browser refresh/crash mid-session must not
            # silently restart them at case 1 (which would re-log already-
            # answered cases as duplicates - a real gap this exact fix closes).
            # Decisions only: the log also holds "consent" and
            # "session_feedback" markers for this same participant_id, and
            # counting those as answered cases makes a resuming participant
            # skip a case and finish one short (a real bug, seen in live data
            # where resumed sessions ended with 19 of 20 decisions).
            prior = [
                e["record"] for e in get_audit_logger().entries
                if e["record"].get("participant_id") == pid
                and "record_type" not in e["record"]
            ]
            st.session_state.position = len(prior)
            st.session_state.decisions = prior
            # Skip the primer on a resumed session (prior decisions already
            # exist) - only a genuinely fresh start needs it.
            st.session_state.primer_ack = len(prior) > 0
            st.rerun()
    st.stop()

participant_id = st.session_state.participant_id
lang = st.session_state.get("confirmed_language", "en")
order = st.session_state.order
position = st.session_state.position
total = len(order)

st.markdown(
    f'<span style="color:var(--ink-soft);font-size:0.92rem;">{t(lang, "participant_caption")} '
    f'<b style="color:var(--ink);">{html.escape(participant_id)}</b></span>',
    unsafe_allow_html=True,
)

# ---- Primer: required before case 1 (expose Section 5.3) ----
if not st.session_state.get("primer_ack", False):
    st.markdown(f'<div class="kicker" style="margin-top:6px;">{t(lang, "primer_time_note")}</div>',
                unsafe_allow_html=True)
    st.markdown(f"## {t(lang, 'primer_heading')}")

    st.markdown(
        f"""
        <div class="primer-grid">
          <div class="primer-card">
            <b class="step-label">{t(lang, "primer_card1_label")}</b><br>
            {t(lang, "primer_card1_text")}
          </div>
          <div class="primer-card">
            <b class="step-label">{t(lang, "primer_card2_label")}</b><br>
            {t(lang, "primer_card2_text")}
          </div>
          <div class="primer-card full">
            <b class="step-label">{t(lang, "primer_card3_label")}</b><br>
            {t(lang, "primer_card3_intro", total=total)}
            <ol style="margin:8px 0 0 0; padding-left:20px;">
              <li>{t(lang, "primer_card3_li0")}</li>
              <li>{t(lang, "primer_card3_li1")}</li>
              <li>{t(lang, "primer_card3_li2")}</li>
              <li>{t(lang, "primer_card3_li3")}</li>
            </ol>
          </div>
          <div class="primer-card full" style="border-left-color:var(--amber);">
            {t(lang, "primer_warn_text")}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander(t(lang, "primer_factor_list_label")):
        st.dataframe(
            pd.DataFrame({
                t(lang, "factor_table_col_factor"): [label_for(c, lang) for c in X.columns],
                t(lang, "factor_table_col_desc"): [describe_feature(c, lang) for c in X.columns],
            }),
            hide_index=True,
            use_container_width=True,
        )
    if st.button(t(lang, "primer_begin_button"), type="primary"):
        st.session_state.primer_ack = True
        st.rerun()
    st.stop()

# ---- Session complete: CSV export ----
if position >= total:
    decisions_df = pd.DataFrame(st.session_state.decisions)
    n_override = int((decisions_df["decision"] == "override").sum()) if len(decisions_df) else 0
    times = decisions_df["engagement_seconds"].dropna() if "engagement_seconds" in decisions_df else []
    total_minutes = (sum(times) / 60) if len(times) else 0.0

    if not st.session_state.get("_celebrated", False):
        st.balloons()
        st.session_state["_celebrated"] = True

    if not st.session_state.get("_sheet_synced", False):
        # External backup, independent of local storage surviving between
        # sessions (see sheet_sync.py docstring). Fires once per session,
        # right when the session completes, not gated on the optional
        # feedback step below.
        append_session_rows(st.session_state.decisions)
        st.session_state["_sheet_synced"] = True

    st.markdown(
        f"""
        <div class="hero-wrap" style="padding-top:24px;">
          <div class="kicker">{t(lang, "complete_kicker")}</div>
          <h1>{t(lang, "complete_title", name=html.escape(participant_id))}</h1>
          <p>{t(lang, "complete_subtitle", total=total)}</p>
        </div>
        <div class="stat-row" style="justify-content:center;">
          <div class="stat-chip"><div class="val">{total}</div><div class="lbl">{t(lang, "stat_cases")}</div></div>
          <div class="stat-chip"><div class="val">{n_override}</div><div class="lbl">{t(lang, "stat_overrides")}</div></div>
          <div class="stat-chip"><div class="val">{total_minutes:.0f} min</div><div class="lbl">{t(lang, "stat_minutes")}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # SONA cannot detect completion of an externally hosted study on its own,
    # and the automatic credit-granting endpoint is not wired up yet (see
    # docs/DEPLOYMENT_CHECKLIST.md, Group 3), so credit is granted manually
    # from an emailed screenshot. Placed directly under the completion header,
    # above everything else, since a participant who misses this does not get
    # paid for their time.
    with st.container(border=True):
        st.subheader(t(lang, "sona_credit_heading"))
        st.markdown(t(lang, "sona_credit_body"))
        st.text_input(
            t(lang, "sona_credit_id_label"),
            value=participant_id,
            disabled=True,
            key=f"sona_id_display_{participant_id}",
        )

    st.dataframe(decisions_df, hide_index=True)

    feedback_key = f"feedback_{participant_id}"
    feedback_text = st.text_area(
        t(lang, "feedback_label"), placeholder=t(lang, "feedback_placeholder"),
        help=t(lang, "feedback_help"), key=feedback_key,
    )
    if not st.session_state.get("_feedback_submitted", False):
        if st.button(t(lang, "feedback_submit")):
            get_audit_logger().append({
                "record_type": "session_feedback",
                "participant_id": participant_id,
                "language": lang,
                "clinical_experience": st.session_state.get("confirmed_clinical_experience"),
                "clinical_experience_detail": st.session_state.get("confirmed_clinical_experience_detail", ""),
                "ai_experience": st.session_state.get("confirmed_ai_experience"),
                "ai_experience_detail": st.session_state.get("confirmed_ai_experience_detail", ""),
                "feedback_text": feedback_text,
            })
            st.session_state["_feedback_submitted"] = True
            st.rerun()
    else:
        st.success(t(lang, "feedback_thanks"))

    decisions_df["feedback"] = feedback_text
    buf = io.StringIO()
    decisions_df.to_csv(buf, index=False)
    st.download_button(
        t(lang, "download_button"),
        data=buf.getvalue(),
        file_name=f"session_{participant_id}.csv",
        mime="text/csv",
        type="primary",
    )
    st.caption(t(lang, "csv_caption"))
    st.stop()

# ---- One case, sequential, no back navigation ----
steps = "".join(
    f'<span style="width:9px;height:9px;border-radius:50%;display:inline-block;'
    f'background:{"var(--accent)" if i < position else ("var(--accent-soft)" if i == position else "var(--line)")};"></span>'
    for i in range(total)
)
st.markdown(
    f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">'
    f'<span style="font-weight:700;color:var(--accent);">{t(lang, "case_progress", n=position + 1, total=total)}</span>'
    f'</div><div style="display:flex;gap:4px;flex-wrap:wrap;margin-bottom:18px;">{steps}</div>',
    unsafe_allow_html=True,
)

case_number = order[position]
case_meta = case_set.loc[case_set["case_number"] == case_number].iloc[0]
row = X.loc[case_meta["row_index"]]

# Scroll to the top on a genuine case change only, not on every rerun (a
# checkbox toggle or radio pick within the same case must not yank the
# participant's scroll position). window.scrollTo does nothing in Streamlit;
# the real scroll container is [data-testid="stMain"] - confirmed by DOM
# inspection (Playwright), not guessed - an earlier "section.main" guess
# silently scrolled nothing.
if st.session_state.get("_last_case_number") != case_number:
    st.session_state["_last_case_number"] = case_number
    components.html(
        "<script>"
        "var doc = window.parent.document;"
        "var main = doc.querySelector('[data-testid=\"stMain\"]');"
        "if (main) { main.scrollTo(0, 0); }"
        "</script>",
        height=0,
    )

pipeline_result = process_case(row, model, X, novelty_thresh, drift_status="SAFE")
classification = pipeline_result["classification"]
case_key = f"case_{participant_id}_{case_number}"

# Vignette first, raw table second: a list of 17 ratings does not communicate
# a patient, and a participant who cannot form their own impression has no
# basis to disagree with the model (see case_narrative.py docstring).
st.subheader(t(lang, "patient_case_subheader"))
total_diff, max_diff = total_difficulty_count(row)
st.caption(t(lang, "difficulty_caption", count=total_diff, max=max_diff))

cards_html = "".join(
    f'<div class="vcard"><div class="vhead"><span class="icon">{SECTION_ICONS.get(key, "•")}</span>{section_label(key, lang)}</div>'
    f'<ul>{"".join(f"<li>{_md_to_html(s)}</li>" for s in statements)}</ul></div>'
    for key, statements in narrative_for(row, reference_df=X, lang=lang) if statements
)
st.markdown(f'<div class="vgrid">{cards_html}</div>', unsafe_allow_html=True)

with st.expander(t(lang, "show_all_fields_label")):
    display_rows = [
        {t(lang, "field_table_col_feature"): label_for(col, lang),
         t(lang, "field_table_col_value"): describe_value(col, row[col], lang)}
        for col in row.index
    ]
    st.dataframe(pd.DataFrame(display_rows), hide_index=True)

# ---- Independent judgment: required before the AI's result is revealed.
# A clean, unanchored data point for the Chapter 8 analysis (their own read
# vs. the AI's, and vs. ground truth) - distinct from, and prior to, the
# accept/override decision that follows once they've seen the AI's
# reasoning. Gated with st.stop() the same way the primer/landing screens
# are, so nothing below (the AI's pill, explanation, decision UI) renders
# until this is answered.
judgment_key = f"judgment_{case_key}"
if judgment_key not in st.session_state:
    st.markdown(f"### {t(lang, 'judgment_heading')}")
    st.caption(t(lang, "judgment_caption"))
    judgment_choice = st.radio(
        t(lang, "judgment_label"), ["GREEN", "YELLOW", "RED"],
        format_func=lambda code: RYG_LABELS[lang][code],
        horizontal=True, key=f"judgment_radio_{case_key}", index=None,
    )
    if st.button(t(lang, "judgment_submit"), key=f"judgment_submit_{case_key}",
                 type="primary", disabled=judgment_choice is None):
        st.session_state[judgment_key] = judgment_choice
        st.rerun()
    st.stop()

your_ryg = st.session_state[judgment_key]


def render_explainability_panel():
    """The panel the friction lock holds on, and whose viewing time is this
    study's dependent variable. Rendered in plain language rather than as a
    raw SHAP table - see explanation_text.py for why that distinction
    matters to the validity of the measurement."""
    st.subheader(t(lang, "explanation_subheader"))

    st.markdown(confidence_sentence(classification["p_frail"], classification["ryg"], lang=lang))
    st.caption(PANEL_INTRO[lang])

    shap_df = explain_case(row, model)
    shap_df.insert(0, "what it means", [
        describe_value(f, v, lang) for f, v in zip(shap_df["feature"], shap_df["value"])
    ])
    shap_df["feature"] = shap_df["feature"].map(lambda f: label_for(f, lang))

    statements, remaining = explanation_sentences(shap_df, top_n=5, lang=lang)
    for s in statements:
        st.markdown(f"- {s}")
    if remaining:
        st.caption(t(lang, "explanation_remaining", n=remaining))

    metrics = get_metrics(X, y)
    recall_frail = metrics["report"]["Frail (1)"]["recall"]
    precision_frail = metrics["report"]["Frail (1)"]["precision"]
    st.info(reliability_note(recall_frail, precision_frail, lang=lang))

    # The expose's interface diagram specifies a "SHAP Explainability Panel"
    # carrying feature attribution, probability, and class metrics. All three
    # are present above in plain language; this expander holds the same
    # content in its technical form, so the requirement is demonstrably met
    # without forcing every participant through raw SHAP values.
    with st.expander(t(lang, "shap_expander_label")):
        st.markdown(f"**{t(lang, 'shap_section_title')}**")
        st.caption(t(lang, "shap_caption"))
        chart_df = shap_df.set_index("feature")["shap_value"].sort_values()
        st.bar_chart(chart_df, horizontal=True)
        st.dataframe(shap_df, hide_index=True)

        st.markdown(f"**{t(lang, 'prob_section_title')}**")
        st.metric(t(lang, "prob_metric_label"), f"{classification['p_frail']:.3f}",
                  help=t(lang, "prob_metric_help"))

        st.markdown(f"**{t(lang, 'class_metrics_title')}**")
        st.caption(CLASS_METRICS_INTRO[lang])
        c1, c2, c3 = st.columns(3)
        c1.metric(t(lang, "recall_label"), f"{recall_frail:.3f}", help=t(lang, "recall_help"))
        c2.metric(t(lang, "precision_label"), f"{precision_frail:.3f}", help=t(lang, "precision_help"))
        c3.metric(t(lang, "rocauc_label"), f"{metrics['roc_auc']:.3f}", help=t(lang, "rocauc_help"))


def render_decision_ui(ryg, p_frail, engagement_seconds):
    decision = st.radio(
        t(lang, "decision_label"), ["Accept", "Override"],
        format_func=lambda v: t(lang, "decision_accept") if v == "Accept" else t(lang, "decision_override"),
        horizontal=True, key=f"decision_{case_key}",
    )
    justification_code = None
    justification_text = ""
    if decision == "Override":
        justification_code = st.selectbox(
            t(lang, "justification_label"), JUSTIFICATION_CODES,
            format_func=lambda code: justification_label(code, lang), key=f"just_{case_key}",
        )
        justification_text = st.text_area(t(lang, "notes_label"), key=f"justtext_{case_key}")

    if st.button(t(lang, "confirm_button"), key=f"confirm_{case_key}", type="primary"):
        record = build_decision_record(
            case_id=case_number, ryg=ryg, p_frail=p_frail,
            engagement_seconds=engagement_seconds, decision=decision.lower(),
            justification_code=justification_code, justification_text=justification_text,
        )
        record["participant_id"] = participant_id
        record["initial_judgment"] = your_ryg
        record["language"] = lang
        record["clinical_experience"] = st.session_state.get("confirmed_clinical_experience")
        record["clinical_experience_detail"] = st.session_state.get("confirmed_clinical_experience_detail", "")
        record["ai_experience"] = st.session_state.get("confirmed_ai_experience")
        record["ai_experience_detail"] = st.session_state.get("confirmed_ai_experience_detail", "")
        get_audit_logger().append(record)
        st.session_state.decisions.append(record)
        st.session_state.position += 1
        st.rerun()


ryg = classification["ryg"]
st.markdown(
    f'<div class="compare-row">'
    f'<div class="compare-item"><span class="compare-lbl">{t(lang, "your_read_label")}</span>'
    f'<div class="ryg-pill {your_ryg}"><span class="dot"></span>{RYG_LABELS[lang][your_ryg]}</div></div>'
    f'<div class="compare-item"><span class="compare-lbl">{t(lang, "ai_read_label")}</span>'
    f'<div class="ryg-pill {ryg}"><span class="dot"></span>{RYG_LABELS[lang][ryg]}</div></div>'
    f'</div>',
    unsafe_allow_html=True,
)
st.caption(t(lang, "risk_signal_caption", p=classification["p_frail"]))

if requires_lock(ryg):
    lock_key = f"lock_start_{case_key}"
    reviewed_key = f"reviewed_{case_key}"
    elapsed_key = f"elapsed_{case_key}"

    if lock_key not in st.session_state:
        st.session_state[lock_key] = time.time()
        st.session_state[reviewed_key] = False

    st.warning(t(lang, "lock_warning"))
    render_explainability_panel()

    already_reviewed = st.session_state[reviewed_key]
    checked = st.checkbox(
        t(lang, "review_checkbox"),
        key=f"cb_{case_key}", value=already_reviewed, disabled=already_reviewed,
    )
    if checked and not already_reviewed:
        st.session_state[elapsed_key] = time.time() - st.session_state[lock_key]
        st.session_state[reviewed_key] = True
        st.rerun()

    if not st.session_state[reviewed_key]:
        st.info(t(lang, "lock_info"))
    else:
        st.success(t(lang, "reviewed_success", secs=st.session_state[elapsed_key]))
        render_decision_ui(ryg, classification["p_frail"], st.session_state[elapsed_key])
else:
    st.caption(t(lang, "low_risk_caption"))
    render_explainability_panel()
    render_decision_ui(ryg, classification["p_frail"], None)
