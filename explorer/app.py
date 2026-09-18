"""Medical KG Explorer — Streamlit chat UI.

Run from the project root:
    streamlit run explorer/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from explorer.kg_query import (  # noqa: E402
    chat_reply,
    check_neo4j_connection,
    classify_message,
    get_grounded_schema,
    normalize_question_terms,
    plan_and_answer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_query_expanders(executed: list[dict], *, expanded: bool = False) -> None:
    """Render one collapsible expander per executed query."""
    for i, q in enumerate(executed, 1):
        icon = "❌" if q.get("error") else "✓"
        label = f"{icon} Query {i}: {q.get('rationale', '')[:60]}"
        with st.expander(label, expanded=expanded):
            st.code(q["cypher"], language="cypher")
            if q.get("error"):
                st.error(f"Error: {q['error']}")
            else:
                n = len(q.get("results") or [])
                st.caption(f"{n} row{'s' if n != 1 else ''} returned")


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Medical KG Explorer",
    page_icon="🧠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Medical KG Explorer")
    st.caption("Ask questions about the clinical knowledge graph in plain English.")

    st.divider()

    # Connection status
    ok, status_msg = check_neo4j_connection()
    if ok:
        st.success("Neo4j connected", icon="✅")
    else:
        st.error("Neo4j unavailable", icon="❌")
    st.caption(status_msg)

    st.divider()

    # Schema viewer — loads once, cached
    with st.expander("Graph schema", expanded=False):
        with st.spinner("Loading schema..."):
            schema = get_grounded_schema()
        st.code(schema, language="text")

    st.divider()

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

st.header("Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        queries = msg.get("queries") or []
        if queries:
            _render_query_expanders(queries, expanded=False)

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

question = st.chat_input("Ask a question about the knowledge graph...")

if question:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Build history for multi-turn context (exclude the message just added)
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        # Step 0: classify intent
        with st.spinner("Thinking..."):
            intent = classify_message(question, history)

        # Schema is needed by both paths — load once here
        schema = get_grounded_schema()

        if intent == "chat":
            # Conversational reply — no Cypher, no Neo4j
            with st.spinner("Answering..."):
                try:
                    answer = chat_reply(question, history, schema=schema)
                except Exception as e:
                    answer = f"Error generating reply: {e}"

            st.markdown(answer)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
            })

        else:
            # intent == "query" — multi-round parallel pipeline
            term_hints = normalize_question_terms(question)
            answer = ""
            all_executed: list[dict] = []

            with st.status("Working...", expanded=True) as status:
                def _on_progress(msg: str) -> None:
                    status.update(label=msg)

                try:
                    answer, all_executed = plan_and_answer(
                        question,
                        schema,
                        history,
                        term_hints=term_hints,
                        progress_callback=_on_progress,
                    )
                    status.update(label="Done", state="complete", expanded=False)
                except Exception as exc:
                    status.update(label=f"Error: {exc}", state="error")
                    answer = f"An unexpected error occurred: {exc}"

            st.markdown(answer)
            _render_query_expanders(all_executed, expanded=False)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "queries": all_executed,
            })
