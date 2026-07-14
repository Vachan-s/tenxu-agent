import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from agent.agent import stream_agent
from agent.search import load_embedding_model, get_supabase_client


@st.cache_resource
def get_model():
    """Load the sentence-transformers model once and cache it across reruns."""
    return load_embedding_model()


@st.cache_resource
def get_supabase():
    """Create the Supabase client once and cache it across reruns."""
    return get_supabase_client()

st.set_page_config(
    page_title="Ten x You — Product Intelligence",
    page_icon="⚡",
    layout="centered",
)

with st.spinner("Starting up Ten x You Agent... this takes about 20-30 seconds on first load"):
    get_model()
    get_supabase()

st.toast("Ready! Ask me anything about our products.", icon="⚡")

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
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        all_chunks = []

        gen = stream_agent(
            user_input,
            st.session_state.messages[:-1]
        )

        # Phase 1: tool use — show spinner until first text chunk arrives
        with st.spinner("Searching products..."):
            for chunk in gen:
                all_chunks.append(chunk)
                # first chunk signals streaming has started
                break

        # Phase 2: collect remaining chunks from generator
        # Last item is the sentinel (full assembled response)
        remaining = list(gen)

        if remaining:
            display_chunks = all_chunks + remaining[:-1]
            sentinel = remaining[-1]
        else:
            display_chunks = all_chunks
            sentinel = "".join(all_chunks)

        # Display chunks one by one with typing cursor
        for chunk in display_chunks:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        # Final display without cursor
        message_placeholder.markdown(full_response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": sentinel if sentinel else full_response,
        })
