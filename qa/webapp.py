# Streamlit app: python -m streamlit run qa/webapp.py
# Question box, mode toggle, shows answer AND source chunks/citations used.
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from dotenv import load_dotenv

from qa.answer import answer_question
from qa.retrieval import retrieve
from storage import db as storedb

load_dotenv()

st.set_page_config(page_title="IR Traffic Commercial Circulars — Q&A", layout="wide")
st.title("Indian Railways Traffic Commercial Directorate — Circular Q&A")
st.caption(
    "Answers are generated only from indexed circulars. Every claim is cited with "
    "circular title/date. If nothing relevant is found, the assistant will say so."
)

DB_PATH = "data/ir_kb.sqlite3"
VECTOR_DIR = "data/chroma_db"


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

with st.form("ask_form"):
    question = st.text_input("Ask a question about Traffic Commercial circulars:")
    mode_choice = st.radio(
        "Retrieval mode",
        ["Auto-detect", "Specific (fast)", "Exhaustive (list all / summarize all)"],
        horizontal=True,
    )
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
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

    st.subheader("Answer")
    st.info(f"Retrieval mode: **{answer.mode}** — model: `{answer.model}`")
    st.markdown(answer.text)

    st.subheader("Source chunks used")
    if not chunks:
        st.write("No chunks were retrieved for this question.")
    for i, c in enumerate(chunks, start=1):
        with st.expander(f"{i}. {c.title} — {c.date or 'unknown date'} ({c.source})"):
            st.write(f"**Section:** {c.section_path}")
            st.write(f"**Source URL:** {c.source_url}")
            st.write(f"**Local file:** {c.local_path}")
            st.text(c.text)
