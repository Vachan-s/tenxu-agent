import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from agent.agent import run_agent

st.set_page_config(
    page_title="Ten x You — Product Intelligence",
    page_icon="⚡",
    layout="centered",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ Ten x You")
    st.markdown("---")
    st.markdown(
        "Ask anything about our products. "
        "Compare, assess, and retrieve product information."
    )
    st.markdown("---")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Header ────────────────────────────────────────────────────────────────────

st.title("Ten x You — Product Intelligence")
st.caption("Your internal product knowledge assistant")
st.markdown("---")

# ── Chat history ──────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input ─────────────────────────────────────────────────────────────────────

user_input = st.chat_input("Ask about a product, compare SKUs, check specs…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            response = run_agent(user_input, st.session_state.messages[:-1])
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
