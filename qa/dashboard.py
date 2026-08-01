# Branded, light, government-portal-styled local dashboard for the Q&A
# engine -- deliberately modeled on the Indian Railways / gov.in visual
# language (navy blue masthead, saffron/white/green strip, clean white
# content cards) rather than a generic dark "tech" theme.
# python -m streamlit run qa/dashboard.py
#
# Same backend as qa/webapp.py (qa.retrieval + qa.answer, unmodified) --
# this file is purely presentation: masthead, live coverage stats,
# styled search + answer + citation cards (including a superseded/current
# status badge, sourced from extraction.resolve_supersession's work).
# qa/webapp.py stays as the minimal fallback UI.
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from dotenv import load_dotenv

from extraction import auto_update
from qa.answer import answer_question, page_label
from qa.retrieval import build_conversational_query, is_exhaustive_query, retrieve
from qa.web_fallback import search_basic_answer
from storage import db as storedb

load_dotenv()

DB_PATH = "data/ir_kb.sqlite3"
VECTOR_DIR = "data/chroma_db"

st.set_page_config(
    page_title="Traffic Commercial Intelligence — Ministry of Railways",
    page_icon="🚆",
    layout="wide",
    # "expanded", not "auto": "auto" left some desktop browsers loading
    # with the sidebar collapsed on first paint, and the collapse/expand
    # arrow that should reopen it is inside header[data-testid="stHeader"],
    # which is force-hidden below -- so a user who landed collapsed had no
    # way back in and the whole sidebar (New chat, Sources, etc.) looked
    # simply missing. Forcing "expanded" sidesteps relying on that hidden
    # control on first load; the CSS fix below also keeps the control
    # visible so manually collapsing still works.
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Noto+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --gov-navy: #0b3d63;
    --gov-navy-dark: #072944;
    --saffron: #ff9933;
    --india-green: #128807;
    --gov-red: #c8102e;
    --bg-page: #eef2f6;
    --bg-panel: #ffffff;
    --border-soft: #dde4ec;
    --text-primary: #14212e;
    --text-muted: #5b6b7c;
    --text-on-navy: #f4f8fb;
}

html, body, [class*="css"] { font-family: 'Noto Sans', sans-serif; }

.stApp {
    background: var(--bg-page);
    color: var(--text-primary);
}

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
/* The sidebar's reopen arrow lives inside the header element hidden
   above -- hiding the whole header took the arrow down with it, so a
   collapsed sidebar had no way to reopen. Force it back to visible so the
   toggle always works regardless of the header being hidden. Streamlit's
   test-id for this control has changed across versions
   (stSidebarCollapsedControl / collapsedControl / stExpandSidebarButton
   all seen in the wild), so cover all of them defensively. */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
[data-testid="stExpandSidebarButton"] {
    visibility: visible !important;
    opacity: 1 !important;
    height: auto !important;
    width: auto !important;
    display: flex !important;
}
.block-container { padding-top: 0; max-width: 1200px; }

/* ---- Government-portal tricolor strip ---- */
.gov-strip {
    height: 6px;
    width: 100%;
    background: linear-gradient(90deg, var(--saffron) 0 33%, #ffffff 33% 67%, var(--india-green) 67% 100%);
    border-radius: 0 0 3px 3px;
    margin-bottom: 1.5rem;
}
.gov-strip-caption {
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted) !important;
    margin: 0.35rem 0 1.1rem;
}

/* ---- Masthead / hero ---- */
.hero {
    display: flex;
    align-items: center;
    gap: 1.4rem;
    padding: 1.65rem 2rem;
    border-radius: 14px;
    background: linear-gradient(120deg, var(--gov-navy) 0%, var(--gov-navy-dark) 100%);
    box-shadow: 0 10px 30px rgba(11, 61, 99, 0.22);
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    /* faint rail-track motif along the bottom edge of the masthead */
    content: "";
    position: absolute;
    left: 0; right: 0; bottom: 0;
    height: 6px;
    background: repeating-linear-gradient(90deg, rgba(255,255,255,0.35) 0 14px, transparent 14px 26px);
}
.hero-mark {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 1.9rem;
    width: 62px; height: 62px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    background: var(--bg-panel);
    border: 3px solid var(--saffron);
    box-shadow: 0 0 0 3px rgba(255,255,255,0.15);
    flex-shrink: 0;
}
.hero-text { line-height: 1.28; }
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
}
.hero-title {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 1.85rem;
    margin: 0.1rem 0;
}
.hero-slogan {
    font-family: 'Noto Sans', sans-serif;
    font-weight: 500;
    font-size: 0.98rem;
    letter-spacing: 0.01em;
}
/* The broad dark-on-light catch-all further below matches these same
   elements (.stApp div/span) at equal or higher CSS specificity, so a
   single-class selector here would silently lose the color fight even with
   !important -- specificity, not source order, decides !important ties.
   Prefixing with ".stApp" turns each into a two-class selector, which
   reliably outranks the catch-all's one-class-plus-type selectors. */
.stApp .hero-mark { color: var(--gov-navy) !important; }
.stApp .hero-eyebrow { color: var(--saffron) !important; }
.stApp .hero-title { color: var(--text-on-navy) !important; }
.stApp .hero-slogan { color: #cfe0ee !important; }
.stApp .hero-slogan b { color: var(--saffron) !important; }
.stApp .hero-slogan .hindi { color: #cfe0ee !important; font-weight: 400; margin-left: 0.4rem; }

/* ---- Stat cards ---- */
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.9rem; margin-bottom: 1.75rem; }
.stat-card {
    background: var(--bg-panel);
    border: 1px solid var(--border-soft);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    box-shadow: 0 1px 3px rgba(11, 61, 99, 0.06);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}
.stat-card:hover { box-shadow: 0 6px 16px rgba(11, 61, 99, 0.12); transform: translateY(-1px); }
.stat-card.c-navy    { border-top: 3px solid var(--gov-navy); }
.stat-card.c-green   { border-top: 3px solid var(--india-green); }
.stat-card.c-saffron { border-top: 3px solid var(--saffron); }
.stat-card.c-red     { border-top: 3px solid var(--gov-red); }
.stat-card.c-slate   { border-top: 3px solid #64748b; }
.stat-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}
.stat-value {
    font-family: 'Poppins', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
}
.stApp .stat-label { color: var(--text-muted) !important; }
.stApp .stat-value { color: var(--text-primary) !important; }
.stat-value.navy    { color: var(--gov-navy) !important; }
.stat-value.green   { color: var(--india-green) !important; }
.stat-value.saffron { color: #b5691a !important; }
.stat-value.red     { color: var(--gov-red) !important; }
.stat-value.slate   { color: #475569 !important; }

/* ---- Answer panel ---- */
.answer-panel {
    background: var(--bg-panel);
    border: 1px solid var(--border-soft);
    border-left: 4px solid var(--gov-navy);
    border-radius: 12px;
    padding: 1.5rem 1.75rem;
    box-shadow: 0 1px 3px rgba(11, 61, 99, 0.06);
    margin-bottom: 1.25rem;
    line-height: 1.65;
}
.stApp .answer-panel { color: var(--text-primary) !important; }
.mode-chip {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    padding: 0.25rem 0.65rem;
    border-radius: 999px;
    background: rgba(11, 61, 99, 0.08);
    border: 1px solid rgba(11, 61, 99, 0.25);
    margin-right: 0.5rem;
}
.stApp .mode-chip { color: var(--gov-navy) !important; }

/* ---- General web fallback panel -- deliberately styled distinct from
   .answer-panel (dashed border, muted background, no navy accent bar) so it
   reads as "unverified general web result", never confusable with a
   circular-grounded, cited answer. ---- */
.web-answer-panel {
    background: #f7f5ef;
    border: 1px dashed #b9ad8f;
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 1.25rem;
    line-height: 1.6;
}
.stApp .web-answer-panel { color: var(--text-primary) !important; }
.web-answer-label {
    display: block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.stApp .web-answer-label { color: #8a7a4a !important; }

/* ---- Citation cards ---- */
.citation-card {
    background: var(--bg-panel);
    border: 1px solid var(--border-soft);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
    box-shadow: 0 1px 2px rgba(11, 61, 99, 0.05);
}
.citation-title { font-weight: 600; font-size: 0.95rem; }
.citation-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    margin-top: 0.25rem;
}
.citation-section { font-size: 0.8rem; margin-top: 0.15rem; }
.stApp .citation-title { color: var(--text-primary) !important; }
.stApp .citation-meta { color: #8a5a12 !important; }
.stApp .citation-section { color: var(--text-muted) !important; }
.status-badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0.15rem 0.55rem;
    border-radius: 999px;
    margin-top: 0.4rem;
}
.status-badge.superseded {
    background: rgba(200, 16, 46, 0.08);
    color: var(--gov-red) !important;
    border: 1px solid rgba(200, 16, 46, 0.3);
}
.status-badge.current {
    background: rgba(18, 136, 7, 0.08);
    color: var(--india-green) !important;
    border: 1px solid rgba(18, 136, 7, 0.3);
}
.status-badge.amended {
    background: rgba(255, 153, 51, 0.12);
    color: #b5691a !important;
    border: 1px solid rgba(255, 153, 51, 0.4);
}
.status-badge.ambiguous {
    background: rgba(100, 116, 139, 0.1);
    color: #475569 !important;
    border: 1px solid rgba(100, 116, 139, 0.35);
}

/* Streamlit widget restyle */
.stTextInput input {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border-soft) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
    font-size: 1rem !important;
}
.stButton button,
[data-testid="stFormSubmitButton"] button,
[data-testid^="stBaseButton"],
button[kind="primary"],
button[kind="secondary"],
button[kind="primaryFormSubmit"],
button[kind="secondaryFormSubmit"] {
    background: linear-gradient(135deg, var(--saffron), #f2711c) !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}
.stRadio label { color: var(--text-primary) !important; }
section[data-testid="stSidebar"] {
    background: var(--bg-panel);
    border-right: 1px solid var(--border-soft);
}

/* Catch-all: Streamlit's default widget/text colors need to be pinned to a
   dark-on-light palette everywhere, including inside the sidebar. More
   specific selectors above (buttons, chips, citation cards) already use
   !important and still win. */
.stApp, .stApp p, .stApp span, .stApp li, .stApp label, .stApp div,
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown strong, .stMarkdown em, .stMarkdown a,
h1, h2, h3, h4, h5, h6,
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
[data-testid="stExpander"], [data-testid="stExpander"] p, [data-testid="stExpander"] summary,
[data-testid="stExpanderDetails"], [data-testid="stExpanderDetails"] p,
.stAlert, .stAlert p,
section[data-testid="stSidebar"] * ,
pre, code {
    color: var(--text-primary) !important;
}
[data-testid="stExpander"] {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 10px !important;
}
pre {
    background: #f4f7fa !important;
    border: 1px solid var(--border-soft) !important;
}
[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, var(--saffron), #f2711c) !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}
a { color: var(--gov-navy) !important; }

/* The catch-all above recolors every nested span/div dark, which is
   already correct for these buttons' light gradient backgrounds -- but the
   button text should stay white regardless, so pin it explicitly (both
   class- and attribute-based selectors, since Streamlit's exact DOM/class
   names have changed across versions). Placed last so it wins same-
   specificity cascade ties. */
.stButton button, .stButton button *,
[data-testid="stDownloadButton"] button, [data-testid="stDownloadButton"] button *,
[data-testid="stFormSubmitButton"] button, [data-testid="stFormSubmitButton"] button *,
[data-testid^="stBaseButton"], [data-testid^="stBaseButton"] *,
button[kind="primary"], button[kind="primary"] *,
button[kind="secondary"], button[kind="secondary"] *,
button[kind="primaryFormSubmit"], button[kind="primaryFormSubmit"] *,
button[kind="secondaryFormSubmit"], button[kind="secondaryFormSubmit"] * {
    color: #ffffff !important;
}

/* Columns (source-link / download-button row on each citation card) should
   stack instead of squeezing side by side on a narrow phone screen. */
[data-testid="stHorizontalBlock"] { flex-wrap: wrap; }

/* ---- Mobile ---- */
@media (max-width: 640px) {
    .block-container { padding-left: 0.75rem !important; padding-right: 0.75rem !important; }
    .gov-strip-caption { font-size: 0.56rem; letter-spacing: 0.05em; padding: 0 0.5rem; }
    .hero {
        flex-direction: column;
        align-items: flex-start;
        padding: 1.1rem 1.1rem;
        gap: 0.6rem;
    }
    .hero-mark { width: 46px; height: 46px; font-size: 1.3rem; }
    .hero-eyebrow { font-size: 0.6rem; letter-spacing: 0.08em; }
    .hero-title { font-size: 1.3rem; }
    .hero-slogan { font-size: 0.85rem; }
    .hero-slogan .hindi { display: block; margin-left: 0; margin-top: 0.15rem; }
    .stat-grid { grid-template-columns: repeat(2, 1fr); gap: 0.6rem; }
    .stat-card { padding: 0.75rem 0.9rem; }
    .stat-value { font-size: 1.3rem; }
    .answer-panel { padding: 1.1rem 1.25rem; font-size: 0.95rem; }
    .citation-card { padding: 0.75rem 0.9rem; }
    .citation-title { font-size: 0.88rem; }
    [data-testid="stHorizontalBlock"] > div { width: 100% !important; flex: 1 1 100% !important; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Government-portal strip + masthead
# ---------------------------------------------------------------------------
st.markdown('<div class="gov-strip"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="gov-strip-caption">भारत सरकार · रेल मंत्रालय &nbsp;|&nbsp; Government of India · Ministry of Railways</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <div class="hero-mark">IR</div>
  <div class="hero-text">
    <div class="hero-eyebrow">Railway Board &middot; Traffic Commercial Directorate</div>
    <div class="hero-title">Traffic Commercial Intelligence</div>
    <div class="hero-slogan"><b>On the right track.</b> The current rule, always cited, always up to date.<span class="hindi">&mdash; सही नीति, सही समय पर</span></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Password gate -- only enforced when DASHBOARD_PASSWORD is set (see
# .env.example). Meant for when this dashboard is exposed via a tunnel
# (ngrok, etc.) to a link being shared with someone else: without a gate,
# anyone with the link could ask unlimited questions billed to your
# ANTHROPIC_API_KEY. Checked BEFORE connecting to the database/vector store
# below, so an unauthenticated visitor never even triggers backend I/O.
# Local-only use (no DASHBOARD_PASSWORD set) is unaffected.
# ---------------------------------------------------------------------------
DASHBOARD_PASSWORD = os.environ.get("DASHBOARD_PASSWORD")

if DASHBOARD_PASSWORD and not st.session_state.get("authenticated"):
    st.markdown(
        '<div class="answer-panel">This dashboard is password-protected. '
        'Enter the access password to continue.</div>',
        unsafe_allow_html=True,
    )
    with st.form("password_gate"):
        entered_password = st.text_input("Password", type="password")
        submitted_password = st.form_submit_button("Enter")
    if submitted_password:
        if entered_password == DASHBOARD_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


# ---------------------------------------------------------------------------
# Auto-update -- checks indianrailways.gov.in for new/changed circulars and
# indexes them automatically, at most once per day (extraction/auto_update.py
# has the full pipeline + gating logic). Runs in a background thread so the
# dashboard opens immediately with the existing index rather than blocking on
# a full site crawl. @st.cache_resource guarantees the thread is started
# exactly once per server process, no matter how many browser tabs/visitors
# hit the dashboard or how many times Streamlit reruns this script -- without
# it, every widget interaction would try to kick off another crawl.
# Placed AFTER the password gate above: an unauthenticated visitor should not
# be able to trigger a site crawl or paid OCR spend just by loading the page.
# ---------------------------------------------------------------------------
DOCS_DIR = "documents"


@st.cache_resource
def _start_auto_update():
    import threading

    from extraction.auto_update import run_if_due

    thread = threading.Thread(
        target=run_if_due,
        kwargs={"docs_dir": DOCS_DIR, "db_path": DB_PATH, "vector_dir": VECTOR_DIR},
        daemon=True,
    )
    thread.start()
    return thread


_start_auto_update()


# ---------------------------------------------------------------------------
# Backend resources + live coverage stats
# ---------------------------------------------------------------------------
@st.cache_resource
def get_resources():
    conn = storedb.connect(DB_PATH)
    vector_store = None
    embedder = None
    try:
        from storage.embeddings import get_default_provider
        from storage.vectorstore import VectorStore

        embedder = get_default_provider()
        vector_store = VectorStore(VECTOR_DIR)
    except Exception:
        pass
    return conn, vector_store, embedder


conn, vector_store, embedder = get_resources()


def _stats(conn):
    total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    flagged = conn.execute("SELECT COUNT(*) FROM documents WHERE needs_review = 1").fetchone()[0]
    sections = conn.execute("SELECT COUNT(DISTINCT section_path) FROM documents").fetchone()[0]
    chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    with_number = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE document_number IS NOT NULL AND document_number != ''"
    ).fetchone()[0]
    superseded = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE superseded_by_doc_id IS NOT NULL AND superseded_by_doc_id != ''"
    ).fetchone()[0]
    amended = conn.execute("SELECT COUNT(*) FROM chunks WHERE status = 'amended'").fetchone()[0]
    coverage_pct = round(100 * (total - flagged) / total, 1) if total else 0.0
    number_pct = round(100 * with_number / total, 1) if total else 0.0
    return total, flagged, sections, chunks, coverage_pct, number_pct, superseded, amended


(
    total_docs, flagged_docs, sections_count, chunk_count,
    coverage_pct, number_pct, superseded_count, amended_count,
) = _stats(conn)

st.markdown(
    f"""
<div class="stat-grid">
  <div class="stat-card c-navy">
    <div class="stat-label">Documents Indexed</div>
    <div class="stat-value navy">{total_docs:,}</div>
  </div>
  <div class="stat-card c-slate">
    <div class="stat-label">Searchable Chunks</div>
    <div class="stat-value slate">{chunk_count:,}</div>
  </div>
  <div class="stat-card c-green">
    <div class="stat-label">Sections Covered</div>
    <div class="stat-value green">{sections_count:,}</div>
  </div>
  <div class="stat-card c-saffron">
    <div class="stat-label">Extraction Coverage</div>
    <div class="stat-value saffron">{coverage_pct}%</div>
  </div>
  <div class="stat-card c-slate">
    <div class="stat-label">Notification No. Detected</div>
    <div class="stat-value slate">{number_pct}%</div>
  </div>
  <div class="stat-card c-red">
    <div class="stat-label">Superseded Rules Resolved</div>
    <div class="stat-value red">{superseded_count:,}</div>
  </div>
  <div class="stat-card c-saffron">
    <div class="stat-label">Clauses Partially Amended</div>
    <div class="stat-value saffron">{amended_count:,}</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Chat
#
# A continuing conversation, not one-shot Q&A: st.session_state.chat_turns
# holds what's rendered on screen (question/answer/citations per turn),
# while st.session_state.api_history holds the parallel Anthropic
# `messages`-shaped history (Answer.history_entries from qa/answer.py) that
# gets passed back in as conversation_history= on every follow-up, so Claude
# sees the whole conversation, not just the latest question in isolation.
# Retrieval for a follow-up uses build_conversational_query() to fold in the
# previous question -- a bare "what about clause 5?" has no retrieval signal
# of its own. Every turn still gets fresh, independently-grounded citations
# (SYSTEM_PROMPT rule 8) -- history is for conversational continuity, not a
# substitute for re-citing.
# ---------------------------------------------------------------------------
if "chat_turns" not in st.session_state:
    st.session_state.chat_turns = []  # [{"role", "text", "mode"?, "model"?, "chunks"?}]
if "api_history" not in st.session_state:
    st.session_state.api_history = []

with st.sidebar:
    # Always visible (not conditional on there being a conversation yet) --
    # a chat app's "new chat" control should always be there, same as
    # ChatGPT/Claude.ai. Clicking with an empty conversation is a harmless
    # no-op.
    if st.button("New chat", use_container_width=True, type="primary"):
        st.session_state.chat_turns = []
        st.session_state.api_history = []
        st.rerun()

    st.markdown("### Session")
    st.caption("Answers are generated only from indexed circulars. Every claim is cited; if a circular has been superseded, the current one is cited first and the old one is marked accordingly. Follow-up questions continue this same conversation.")

    st.markdown("### Auto-update")
    auto_update_state = auto_update.read_status()
    _auto_update_icons = {
        "idle": "⏳", "scraping": "🔎", "extracting": "📄", "resolving": "🔗",
        "ocr": "🖼️", "done": "✅", "failed": "⚠️",
    }
    icon = _auto_update_icons.get(auto_update_state.status, "⏳")
    st.caption(f"{icon} {auto_update_state.detail or 'Checking for new circulars...'}")

    st.markdown("### Retrieval mode")
    mode_choice = st.radio(
        "Retrieval mode",
        ["Auto-detect", "Specific (fast)", "Exhaustive (list all / summarize all)"],
        label_visibility="collapsed",
    )

    user_questions = [t["text"] for t in st.session_state.chat_turns if t["role"] == "user"]
    if user_questions:
        st.markdown("**Recent questions**")
        for q in reversed(user_questions[-8:]):
            st.markdown(f"- {q}")


def _render_citations(chunks: list, turn_index: int) -> None:
    # Collapsed by default, inside its own expander -- previously this
    # rendered inline and unconditionally below every answer, which meant
    # a long list of citation cards visually crowded/pushed against the
    # answer text on every single turn. Tucking it behind a click keeps the
    # chat scannable; the count in the label tells you it's there without
    # opening it.
    label = f"📚 Sources ({len(chunks)})" if chunks else "📚 Sources"
    with st.expander(label, expanded=False):
        if not chunks:
            st.caption("No chunks were retrieved for this question.")
            return
        status_labels = {
            "superseded": "Superseded",
            "amended": "Partially amended",
            "ambiguous": "Conflicting signals",
        }
        for i, c in enumerate(chunks, start=1):
            clause = c.clause_ref or "none detected"
            page = page_label(c.page_start, c.page_end)
            letter_no = c.circular_number or "not detected"
            if c.status in status_labels:
                status_badge = f'<div class="status-badge {c.status}">{status_labels[c.status]}: {c.status_note}</div>'
            else:
                status_badge = '<div class="status-badge current">Current</div>'
            st.markdown(
                f"""
<div class="citation-card">
  <div class="citation-title">{i}. {c.title} &mdash; {c.date or 'unknown date'}</div>
  <div class="citation-meta">No. {letter_no} &middot; Clause {clause} &middot; Page {page} &middot; {c.source}</div>
  <div class="citation-section">{c.section_path}</div>
  {status_badge}
</div>
""",
                unsafe_allow_html=True,
            )
            col1, col2 = st.columns([1, 3])
            with col1:
                pdf_path = Path(c.local_path) if c.local_path else None
                if pdf_path and pdf_path.exists():
                    st.download_button(
                        label="Download PDF",
                        data=pdf_path.read_bytes(),
                        file_name=pdf_path.name,
                        mime="application/pdf",
                        key=f"dl_{turn_index}_{c.chunk_id}",
                    )
            with col2:
                st.markdown(f"[Source on railwayboard site]({c.source_url})")
            # st.popover, not a nested st.expander -- Streamlit doesn't
            # allow an expander inside another expander.
            with st.popover("Excerpt text"):
                st.text(c.text)


for turn_index, turn in enumerate(st.session_state.chat_turns):
    with st.chat_message(turn["role"]):
        if turn["role"] == "assistant":
            st.markdown(
                f'<span class="mode-chip">{turn.get("mode", "")}</span>'
                f'<span class="mode-chip">{turn.get("model", "")}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f'<div class="answer-panel">{turn["text"]}</div>', unsafe_allow_html=True)
            _render_citations(turn.get("chunks", []), turn_index)
            web_answer = turn.get("web_answer")
            if web_answer:
                st.markdown(
                    f'<div class="web-answer-panel"><span class="web-answer-label">'
                    f'\U0001f310 General web result &mdash; not from an indexed circular, verify independently</span>'
                    f'{web_answer["text"]} '
                    f'(<a href="{web_answer["source_url"]}">{web_answer["source_name"]}</a>)</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(turn["text"])

prompt = st.chat_input("Ask a question about Traffic Commercial circulars")
if prompt and prompt.strip():
    question = prompt.strip()
    st.session_state.chat_turns.append({"role": "user", "text": question})

    if mode_choice.startswith("Specific"):
        exhaustive = False
    elif mode_choice.startswith("Exhaustive"):
        exhaustive = True
    else:
        exhaustive = is_exhaustive_query(question)  # auto-detect from the new question alone

    previous_question = next(
        (t["text"] for t in reversed(st.session_state.chat_turns[:-1]) if t["role"] == "user"), None
    )
    retrieval_query = build_conversational_query(question, previous_question)

    with st.spinner("Retrieving relevant circulars..."):
        chunks, mode = retrieve(
            retrieval_query, conn, vector_store=vector_store, embedder=embedder, exhaustive=exhaustive
        )

    with st.spinner("Asking Claude..."):
        try:
            answer = answer_question(
                question, chunks, mode=mode, conversation_history=st.session_state.api_history
            )
        except Exception as exc:
            st.session_state.chat_turns.append(
                {"role": "assistant", "text": f"Failed to get an answer: {exc}", "mode": mode, "model": "", "chunks": []}
            )
        else:
            st.session_state.api_history.extend(answer.history_entries)
            web_answer = None
            # Only ever consulted when the circular-grounded path found
            # nothing to cite -- never used to second-guess or replace a
            # real, cited answer. Keeps the "only cite indexed circulars,
            # never guess" guarantee intact for anything this knowledge
            # base actually covers; this is purely a courtesy for basic,
            # out-of-scope questions (e.g. general definitions) so the user
            # isn't left with a bare "not found".
            if not chunks:
                with st.spinner("No matching circular -- checking a basic web answer..."):
                    result = search_basic_answer(question)
                if result:
                    web_answer = {
                        "text": result.text,
                        "source_url": result.source_url,
                        "source_name": result.source_name,
                    }
            st.session_state.chat_turns.append(
                {
                    "role": "assistant",
                    "text": answer.text,
                    "mode": answer.mode,
                    "model": answer.model,
                    "chunks": chunks,
                    "web_answer": web_answer,
                }
            )
    st.rerun()
