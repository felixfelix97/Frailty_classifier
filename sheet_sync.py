"""Best-effort external backup of completed sessions to a Google Sheet.

Why this exists: a Streamlit Cloud free-tier app can be restarted (going idle,
then waking back up) between sessions, and local file storage is not
guaranteed to survive that restart. `experiment_audit_log.jsonl` stays the
primary, tamper-evident record either way; this module is a second, external
copy written at the moment each session completes, so a storage reset between
participants cannot silently lose already-finished sessions.

Configuration lives in Streamlit's secrets, never in this file or in git:

    sheet_id = "the Google Sheet ID from its URL"

    [gcp_service_account]
    type = "service_account"
    project_id = "..."
    private_key_id = "..."
    private_key = "..."
    client_email = "..."
    client_id = "..."
    auth_uri = "https://accounts.google.com/o/oauth2/auth"
    token_uri = "https://oauth2.googleapis.com/token"
    auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
    client_x509_cert_url = "..."

Locally, this goes in `.streamlit/secrets.toml` (already gitignored). On
Streamlit Cloud, the same content goes into the app's Settings -> Secrets
panel in the dashboard, not into the repository.

If no secrets are configured (local dev, tests, or before setup is done),
`append_session_rows` is a silent no-op rather than an error, so nothing
about the experiment flow depends on this being set up.
"""

import streamlit as st

_SHEET_TITLE = "sessions"


def _get_worksheet():
    """Returns the target worksheet, or None if sync is not configured."""
    if "gcp_service_account" not in st.secrets or "sheet_id" not in st.secrets:
        return None
    import gspread

    creds = dict(st.secrets["gcp_service_account"])
    client = gspread.service_account_from_dict(creds)
    sheet = client.open_by_key(st.secrets["sheet_id"])
    try:
        return sheet.worksheet(_SHEET_TITLE)
    except gspread.WorksheetNotFound:
        return sheet.sheet1


def append_session_rows(rows):
    """Appends every decision row of one completed session to the sheet.

    `rows` is a list of dicts (one per case decision, same shape as the CSV
    export). Best-effort: any failure is caught and surfaced as a small
    warning in the UI, never raised, so a Sheets outage cannot break a
    participant's session or block the CSV download that already works
    without this.
    """
    if not rows:
        return
    try:
        ws = _get_worksheet()
        if ws is None:
            return  # not configured, nothing to do

        columns = list(rows[0].keys())
        if not ws.get_all_values():
            ws.append_row(columns, value_input_option="RAW")
        ws.append_rows(
            [[str(row.get(c, "")) for c in columns] for row in rows],
            value_input_option="RAW",
        )
    except Exception as e:  # noqa: BLE001 - best-effort backup, never fatal
        st.warning(
            f"Session data saved locally, but the external backup sync failed: {e}",
            icon="⚠️",
        )
