# Branded, futuristic local dashboard for the Q&A engine.
# python -m streamlit run qa/dashboard.py
#
# Same backend as qa/webapp.py (qa.retrieval + qa.answer, unmodified) --
# this file is purely presentation: hero header, live coverage stats,
# styled search + answer + citation cards. qa/webapp.py stays as the
# minimal fallback UI.
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
    page_icon="🚄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg-deep: #060911;
    --bg-panel: rgba(24, 26, 46, 0.68);
    --bg-panel-solid: #1a1d33;
    --border-glow: rgba(56, 189, 248, 0.3);
    --accent-cyan: #38bdf8;
    --accent-blue: #3b82f6;
    --accent-violet: #a855f7;
    --accent-magenta: #ec4899;
    --accent-teal: #2dd4bf;
    --accent-gold: #f2b134;
    --text-primary: #f1f4fb;
    --text-muted: #a3aec4;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(ellipse 70% 50% at 8% 0%, rgba(168, 85, 247, 0.35), transparent 60%),
        radial-gradient(ellipse 60% 45% at 95% 8%, rgba(56, 189, 248, 0.32), transparent 60%),
        radial-gradient(ellipse 65% 55% at 30% 100%, rgba(236, 72, 153, 0.22), transparent 60%),
        radial-gradient(ellipse 60% 50% at 90% 95%, rgba(45, 212, 191, 0.22), transparent 60%),
        linear-gradient(160deg, #171a2e 0%, #12142a 45%, #191231 100%);
    background-attachment: fixed;
    color: var(--text-primary);
}

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 2rem; max-width: 1200px; }

/* ---- Hero ---- */
.hero {
    display: flex;
    align-items: center;
    gap: 1.25rem;
    padding: 1.75rem 2rem;
    border-radius: 18px;
    background: linear-gradient(120deg, rgba(168,85,247,0.22), rgba(56,189,248,0.14) 55%, rgba(45,212,191,0.12));
    border: 1px solid var(--border-glow);
    box-shadow: 0 0 60px rgba(168, 85, 247, 0.12);
    margin-bottom: 1.75rem;
}
.hero-mark {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2.1rem;
    width: 64px; height: 64px;
    border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
    color: #05070c !important;
    box-shadow: 0 0 30px rgba(56, 189, 248, 0.45);
    flex-shrink: 0;
}
.hero-text { line-height: 1.25; }
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    color: var(--accent-gold) !important;
    text-transform: uppercase;
    margin-bottom: 0.15rem;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2rem;
    background: linear-gradient(90deg, #ffffff, var(--accent-cyan) 70%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent !important;
    margin: 0.1rem 0;
}
.hero-slogan {
    font-family: 'Space Grotesk', sans-serif;
    font-style: italic;
    font-weight: 500;
    font-size: 0.98rem;
    color: var(--text-muted) !important;
    letter-spacing: 0.01em;
}
.hero-slogan b { color: var(--accent-gold) !important; font-style: normal; }

/* ---- Stat cards ---- */
.stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.9rem; margin-bottom: 1.75rem; }
.stat-card {
    background: var(--bg-panel);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    backdrop-filter: blur(10px);
    transition: border-color 0.2s ease;
}
.stat-card:hover { border-color: var(--border-glow); }
.stat-card.c-cyan   { border-top: 2px solid var(--accent-cyan); }
.stat-card.c-violet { border-top: 2px solid var(--accent-violet); }
.stat-card.c-teal   { border-top: 2px solid var(--accent-teal); }
.stat-card.c-gold   { border-top: 2px solid var(--accent-gold); }
.stat-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted) !important;
    margin-bottom: 0.35rem;
}
.stat-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.65rem;
    font-weight: 700;
    color: var(--text-primary) !important;
}
.stat-value.cyan   { color: var(--accent-cyan) !important; }
.stat-value.violet { color: var(--accent-violet) !important; }
.stat-value.teal   { color: var(--accent-teal) !important; }
.stat-value.gold   { color: var(--accent-gold) !important; }

/* ---- Answer panel ---- */
.answer-panel {
    background: var(--bg-panel);
    border: 1px solid var(--border-glow);
    border-left: 3px solid var(--accent-cyan);
    border-radius: 14px;
    padding: 1.5rem 1.75rem;
    backdrop-filter: blur(10px);
    margin-bottom: 1.25rem;
    line-height: 1.65;
}
.mode-chip {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    background: rgba(56, 189, 248, 0.12);
    color: var(--accent-cyan) !important;
    border: 1px solid rgba(56, 189, 248, 0.35);
    margin-right: 0.5rem;
}

/* ---- Citation cards ---- */
.citation-card {
    background: var(--bg-panel-solid);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
}
.citation-title { font-weight: 600; color: var(--text-primary) !important; font-size: 0.95rem; }
.citation-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--accent-gold) !important;
    margin-top: 0.2rem;
}
.citation-section { color: var(--text-muted) !important; font-size: 0.8rem; margin-top: 0.15rem; }

/* Streamlit widget restyle */
.stTextInput input {
    background: var(--bg-panel-solid) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
}
.stButton button,
[data-testid="stFormSubmitButton"] button,
[data-testid^="stBaseButton"],
button[kind="primary"],
button[kind="secondary"],
button[kind="primaryFormSubmit"],
button[kind="secondaryFormSubmit"] {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan)) !important;
    color: #05070c !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
}
.stRadio label { color: var(--text-primary) !important; }
section[data-testid="stSidebar"] {
    background: var(--bg-panel-solid);
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* Catch-all: Streamlit's default widget/text colors assume a light theme,
   so anything not already explicitly styled above renders dark-on-dark
   against this page's background. Force the readable palette everywhere,
   including inside the sidebar. More specific selectors above (buttons,
   chips, citation cards) already use !important and still win. */
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
    background: var(--bg-panel-solid) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
}
pre {
    background: var(--bg-panel-solid) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}
[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan)) !important;
    color: #05070c !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
}
a { color: var(--accent-cyan) !important; }

/* The catch-all above recolors every nested span/div light, including
   text inside the Ask/Download buttons' own gradient backgrounds -- that
   leaves near-invisible light-on-light text. Force it back to dark inside
   any button, using both class- and attribute-based selectors since
   Streamlit's exact DOM/class names have changed across versions. Placed
   last so it wins same-specificity cascade ties. */
.stButton button, .stButton button *,
[data-testid="stDownloadButton"] button, [data-testid="stDownloadButton"] button *,
[data-testid="stFormSubmitButton"] button, [data-testid="stFormSubmitButton"] button *,
[data-testid^="stBaseButton"], [data-testid^="stBaseButton"] *,
button[kind="primary"], button[kind="primary"] *,
button[kind="secondary"], button[kind="secondary"] *,
button[kind="primaryFormSubmit"], button[kind="primaryFormSubmit"] *,
button[kind="secondaryFormSubmit"], button[kind="secondaryFormSubmit"] * {
    color: #05070c !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <div class="hero-mark">IR</div>
  <div class="hero-text">
    <div class="hero-eyebrow">Ministry of Railways &middot; Railway Board &middot; Traffic Commercial Directorate</div>
    <div class="hero-title">Traffic Commercial Intelligence</div>
    <div class="hero-slogan"><b>Knowledge of policy is power.</b> Every circular, indexed and cited.</div>
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
    coverage_pct = round(100 * (total - flagged) / total, 1) if total else 0.0
    return total, flagged, sections, chunks, coverage_pct


total_docs, flagged_docs, sections_count, chunk_count, coverage_pct = _stats(conn)

st.markdown(
    f"""
<div class="stat-grid">
  <div class="stat-card c-cyan">
    <div class="stat-label">Documents Indexed</div>
    <div class="stat-value cyan">{total_docs:,}</div>
  </div>
  <div class="stat-card c-violet">
    <div class="stat-label">Searchable Chunks</div>
    <div class="stat-value violet">{chunk_count:,}</div>
  </div>
  <div class="stat-card c-teal">
    <div class="stat-label">Sections Covered</div>
    <div class="stat-value teal">{sections_count:,}</div>
  </div>
  <div class="stat-card c-gold">
    <div class="stat-label">Extraction Coverage</div>
    <div class="stat-value gold">{coverage_pct}%</div>
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
    st.caption("Answers are generated only from indexed circulars. Every claim is cited; if nothing relevant is found, the assistant says so explicitly.")
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
        st.markdown(
            f"""
<div class="citation-card">
  <div class="citation-title">{i}. {c.title} &mdash; {c.date or 'unknown date'}</div>
  <div class="citation-meta">No. {letter_no} &middot; Clause {clause} &middot; Page {page} &middot; {c.source}</div>
  <div class="citation-section">{c.section_path}</div>
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
