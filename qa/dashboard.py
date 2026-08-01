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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from dotenv import load_dotenv

from qa.answer import answer_question, page_label
from qa.retrieval import retrieve
from storage import db as storedb

load_dotenv()

DB_PATH = "data/ir_kb.sqlite3"
VECTOR_DIR = "data/chroma_db"

st.set_page_config(
    page_title="Traffic Commercial Intelligence — Ministry of Railways",
    page_icon="🚆",
    layout="wide",
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
# Search
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.markdown("### Session")
    st.caption("Answers are generated only from indexed circulars. Every claim is cited; if a circular has been superseded, the current one is cited first and the old one is marked accordingly.")
    if st.session_state.history:
        st.markdown("**Recent questions**")
        for q in reversed(st.session_state.history[-8:]):
            st.markdown(f"- {q}")

with st.form("ask_form"):
    question = st.text_input(
        "Ask a question about Traffic Commercial circulars",
        placeholder="e.g. What is the current policy on refund of unused tickets?",
    )
    mode_choice = st.radio(
        "Retrieval mode",
        ["Auto-detect", "Specific (fast)", "Exhaustive (list all / summarize all)"],
        horizontal=True,
    )
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    st.session_state.history.append(question.strip())

    exhaustive = None
    if mode_choice.startswith("Specific"):
        exhaustive = False
    elif mode_choice.startswith("Exhaustive"):
        exhaustive = True

    with st.spinner("Retrieving relevant circulars..."):
        chunks, mode = retrieve(question, conn, vector_store=vector_store, embedder=embedder, exhaustive=exhaustive)

    with st.spinner("Asking Claude..."):
        try:
            answer = answer_question(question, chunks, mode=mode)
        except Exception as exc:
            st.error(f"Failed to get an answer: {exc}")
            st.stop()

    st.markdown(
        f'<span class="mode-chip">{answer.mode}</span><span class="mode-chip">{answer.model}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="answer-panel">{answer.text}</div>', unsafe_allow_html=True)

    st.markdown("#### Sources")
    if not chunks:
        st.caption("No chunks were retrieved for this question.")
    for i, c in enumerate(chunks, start=1):
        clause = c.clause_ref or "none detected"
        page = page_label(c.page_start, c.page_end)
        letter_no = c.circular_number or "not detected"
        status_labels = {
            "superseded": "Superseded",
            "amended": "Partially amended",
            "ambiguous": "Conflicting signals",
        }
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
                    key=f"dl_{c.chunk_id}",
                )
        with col2:
            st.markdown(f"[Source on railwayboard site]({c.source_url})")
        with st.expander("Excerpt text"):
            st.text(c.text)
