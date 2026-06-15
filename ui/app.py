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

get_model()
get_supabase()

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
        gen = stream_agent(user_input, st.session_state.messages[:-1])
        sentinel = [None]

        # Advance the generator past the tool-use phase under the spinner.
        # The generator blocks here until the first text chunk is ready,
        # so the spinner is visible for exactly the tool-use phase.
        with st.spinner("Thinking…"):
            first_chunk = next(gen, None)

        # Spinner exits; stream remaining chunks to the UI.
        # Use a shift-by-one wrapper so the sentinel (last item = full response)
        # is held back from display and captured for chat history instead.
        def _display_gen():
            if first_chunk is not None:
                yield first_chunk
            prev = None
            for item in gen:
                if prev is not None:
                    yield prev
                prev = item
            sentinel[0] = prev  # last item is the full assembled response

        st.write_stream(_display_gen())
        response = sentinel[0] or ""

    st.session_state.messages.append({"role": "assistant", "content": response})
