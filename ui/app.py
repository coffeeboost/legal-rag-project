"""
LexRAG — Streamlit chat UI.
Connects to the FastAPI backend. Each conversation is a persistent session
stored server-side; users can resume any past chat from the sidebar.
"""
import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="LexRAG",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Tighten sidebar padding */
    section[data-testid="stSidebar"] { padding-top: 1rem; }
    /* Hide default Streamlit header chrome */
    header[data-testid="stHeader"] { display: none; }
    /* Source pill styling */
    .source-pill {
        display: inline-block;
        background: #1e3a5f;
        color: #cfe2ff;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ── API helpers ───────────────────────────────────────────────────────────────

def api(method: str, path: str, **kwargs):
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=120, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the API. Is the backend running?")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def load_sessions():
    return api("GET", "/sessions") or []


def new_session():
    return api("POST", "/sessions", json={"title": "New Chat"})


def load_session(session_id: str):
    return api("GET", f"/sessions/{session_id}")


def send_message(session_id: str, question: str):
    return api("POST", f"/sessions/{session_id}/chat", json={"question": question})


def remove_session(session_id: str):
    api("DELETE", f"/sessions/{session_id}")


# ── Session state defaults ────────────────────────────────────────────────────

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sessions" not in st.session_state:
    st.session_state.sessions = load_sessions()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚖️ LexRAG")
    st.caption("Legal Document AI")
    st.divider()

    if st.button("＋  New Chat", use_container_width=True, type="primary"):
        session = new_session()
        if session:
            st.session_state.current_session_id = session["id"]
            st.session_state.messages = []
            st.session_state.sessions = load_sessions()
            st.rerun()

    st.markdown("**Recent chats**")

    sessions = st.session_state.sessions
    if not sessions:
        st.caption("No chats yet. Start one above.")
    else:
        for s in sessions:
            is_active = s["id"] == st.session_state.current_session_id
            col1, col2 = st.columns([5, 1])
            with col1:
                label = s["title"][:40] + ("…" if len(s["title"]) > 40 else "")
                if st.button(
                    label,
                    key=f"sess_{s['id']}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    session_data = load_session(s["id"])
                    if session_data:
                        st.session_state.current_session_id = s["id"]
                        st.session_state.messages = session_data.get("messages", [])
                        st.rerun()
            with col2:
                if st.button("🗑", key=f"del_{s['id']}"):
                    remove_session(s["id"])
                    if st.session_state.current_session_id == s["id"]:
                        st.session_state.current_session_id = None
                        st.session_state.messages = []
                    st.session_state.sessions = load_sessions()
                    st.rerun()

    st.divider()
    with st.expander("Upload a document"):
        uploaded = st.file_uploader("PDF only", type=["pdf"])
        doc_type = st.selectbox("Type", ["unknown", "contract", "court_opinion", "statute"])
        jurisdiction = st.text_input("Jurisdiction", placeholder="e.g. ON, NY, Federal")
        if st.button("Ingest", disabled=uploaded is None):
            with st.spinner("Ingesting…"):
                files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
                data = {"doc_type": doc_type, "jurisdiction": jurisdiction or "unknown"}
                try:
                    r = requests.post(f"{API_URL}/ingest", files=files, data=data, timeout=120)
                    r.raise_for_status()
                    result = r.json()
                    st.success(f"Ingested {result['chunks_ingested']} chunks from {result['filename']}")
                except Exception as e:
                    st.error(f"Ingest failed: {e}")


# ── Main chat area ────────────────────────────────────────────────────────────

if st.session_state.current_session_id is None:
    st.markdown("## Welcome to LexRAG ⚖️")
    st.markdown(
        "LexRAG lets you chat with your legal documents — contracts, court opinions, statutes. "
        "All answers are grounded in the documents you've ingested, with source citations."
    )
    st.info("Start a **New Chat** from the sidebar, or select a previous conversation.")

else:
    # Render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            sources = msg.get("sources") or []
            if sources and msg["role"] == "assistant":
                with st.expander(f"Sources ({len(sources)})"):
                    for src in sources:
                        st.markdown(
                            f"**{src['filename']}** · chunk {src['chunk_index']} · score `{src['score']}`"
                        )
                        st.caption(src["preview"])

    # Input
    if prompt := st.chat_input("Ask a question about your legal documents…"):
        # Render user bubble immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        # Call API and render assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                result = send_message(st.session_state.current_session_id, prompt)

            if result:
                st.markdown(result["answer"])
                sources = result.get("sources", [])
                if sources:
                    with st.expander(f"Sources ({len(sources)})"):
                        for src in sources:
                            st.markdown(
                                f"**{src['filename']}** · chunk {src['chunk_index']} · score `{src['score']}`"
                            )
                            st.caption(src["preview"])

                # Update local state
                st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
                st.session_state.messages.append({"role": "assistant", "content": result["answer"], "sources": sources})

                # Refresh sidebar session list (title may have updated)
                st.session_state.sessions = load_sessions()
                st.rerun()
