import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import os
import tempfile
import shutil
import time

st.set_page_config(
    page_title="Research Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS — ChatGPT-style interface ──────────────────────
st.markdown("""
<style>
/* Hide default Streamlit header and footer */
#MainMenu, footer, header {visibility: hidden;}

/* Main background */
.stApp { background-color: #212121; }

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #171717;
    border-right: 1px solid #2f2f2f;
}
[data-testid="stSidebar"] * { color: #ececec !important; }

/* Sidebar buttons — chat sessions */
.stButton > button {
    background: transparent;
    border: none;
    color: #ececec !important;
    text-align: left;
    padding: 8px 12px;
    border-radius: 8px;
    width: 100%;
    font-size: 13px;
    transition: background 0.15s;
}
.stButton > button:hover { background: #2a2a2a !important; }

/* Active session button */
.stButton > button:focus {
    background: #2f2f2f !important;
    box-shadow: none !important;
}

/* New Chat button */
div[data-testid="stSidebarNav"] { display: none; }

/* Chat messages area */
.stChatMessage {
    background: transparent !important;
    border: none !important;
    padding: 12px 0 !important;
}

/* User message bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: #2f2f2f !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    margin: 4px 0 !important;
}

/* Assistant message */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: transparent !important;
    padding: 12px 4px !important;
}

/* Chat input box */
[data-testid="stChatInput"] {
    background: #2f2f2f !important;
    border: 1px solid #3f3f3f !important;
    border-radius: 16px !important;
    color: #ececec !important;
}

/* Expander (sources) */
[data-testid="stExpander"] {
    background: #1a1a1a !important;
    border: 1px solid #2f2f2f !important;
    border-radius: 8px !important;
}

/* Status box */
[data-testid="stStatusWidget"] { background: #2a2a2a !important; }

/* Upload area */
[data-testid="stFileUploader"] {
    background: #1e1e1e !important;
    border: 1px dashed #3f3f3f !important;
    border-radius: 10px !important;
    padding: 8px !important;
}

/* Success / info messages */
[data-testid="stAlert"] { border-radius: 8px !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #171717; }
::-webkit-scrollbar-thumb { background: #3f3f3f; border-radius: 3px; }

/* Text colors */
p, span, label, div { color: #ececec; }
code { background: #2a2a2a !important; color: #a8d8a8 !important; }

/* Sidebar section labels */
.sidebar-label {
    font-size: 11px;
    font-weight: 600;
    color: #8e8ea0 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0 4px;
    margin: 12px 0 6px;
}
</style>
""", unsafe_allow_html=True)

# ── Import pipeline ───────────────────────────────────────────
from src.rag_pipeline import ingest, query, get_indexed_papers

# ── Session state initialisation ──────────────────────────────
if "sessions" not in st.session_state:
    # Each session: {id, title, messages, paper}
    st.session_state.sessions = []

if "active_session" not in st.session_state:
    st.session_state.active_session = None

if "session_counter" not in st.session_state:
    st.session_state.session_counter = 0


def create_new_session():
    st.session_state.session_counter += 1
    session_id = st.session_state.session_counter
    new_session = {
        "id": session_id,
        "title": f"Chat {session_id}",
        "messages": [],
        "paper": None
    }
    st.session_state.sessions.append(new_session)
    st.session_state.active_session = session_id


def get_active_session():
    for s in st.session_state.sessions:
        if s["id"] == st.session_state.active_session:
            return s
    return None


def update_session_title(session_id, title):
    for s in st.session_state.sessions:
        if s["id"] == session_id:
            s["title"] = title[:30]
            break


# Create first session if none exists
if not st.session_state.sessions:
    create_new_session()

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    # App title
    st.markdown("""
    <div style='padding: 8px 4px 16px; display:flex; align-items:center; gap:10px'>
        <span style='font-size:20px'>📄</span>
        <span style='font-size:16px; font-weight:600; color:#ececec'>Research Assistant</span>
    </div>
    """, unsafe_allow_html=True)

    # New Chat button
    if st.button("✏️  New Chat", use_container_width=True, key="new_chat_btn"):
        create_new_session()
        st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Chat history list
    st.markdown("<div class='sidebar-label'>Your chats</div>", unsafe_allow_html=True)

    for session in reversed(st.session_state.sessions):
        is_active = session["id"] == st.session_state.active_session
        paper_icon = "📄 " if session["paper"] else "💬 "
        label = paper_icon + session["title"]

        if is_active:
            st.markdown(f"""
            <div style='background:#2f2f2f; border-radius:8px; padding:8px 12px;
                        font-size:13px; color:#ececec; margin-bottom:4px;
                        border-left: 3px solid #10a37f'>
                {label}
            </div>""", unsafe_allow_html=True)
        else:
            if st.button(label, key=f"sess_{session['id']}", use_container_width=True):
                st.session_state.active_session = session["id"]
                st.rerun()

    st.markdown("---")

    # Active session controls
    active = get_active_session()
    if active:
        st.markdown("<div class='sidebar-label'>This chat</div>",
                    unsafe_allow_html=True)

        # Paper status
        if active["paper"]:
            st.markdown(f"""
            <div style='background:#1a2e1a; border:1px solid #2d4a2d;
                        border-radius:8px; padding:8px 12px; margin-bottom:8px'>
                <div style='font-size:11px; color:#8e8ea0; margin-bottom:2px'>Uploaded paper</div>
                <div style='font-size:12px; color:#a8d8a8'>📄 {active["paper"]}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:#1e1e1e; border:1px solid #2f2f2f;
                        border-radius:8px; padding:8px 12px; margin-bottom:8px'>
                <div style='font-size:12px; color:#8e8ea0'>No paper uploaded yet</div>
            </div>""", unsafe_allow_html=True)

        # Upload PDF
        uploaded_file = st.file_uploader(
            "Upload a PDF",
            type="pdf",
            key=f"upload_{active['id']}"
        )

        if uploaded_file is not None:
            clean_path = f"/tmp/{uploaded_file.name}"
            with open(clean_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.status(f"Indexing {uploaded_file.name}..."):
                n = ingest(clean_path)

            if n > 0:
                active["paper"] = uploaded_file.name
                update_session_title(active["id"], uploaded_file.name.replace(".pdf", ""))
                st.success(f"Indexed {n} chunks!")
                st.rerun()
            else:
                active["paper"] = uploaded_file.name
                st.info("Already indexed this session.")

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        # Search scope
        if active["paper"]:
            scope = st.radio(
                "Search in",
                ["This paper only", "All uploaded papers"],
                key=f"scope_{active['id']}"
            )
        
        # Clear chat button
        if active["messages"]:
            if st.button("🗑️  Clear messages", use_container_width=True,
                         key=f"clear_{active['id']}"):
                active["messages"] = []
                st.rerun()

# ── MAIN AREA ─────────────────────────────────────────────────
active = get_active_session()

if not active:
    st.stop()

# Welcome screen — no messages yet
if not active["messages"] and not active["paper"]:
    st.markdown("""
    <div style='display:flex; flex-direction:column; align-items:center;
                justify-content:center; height:60vh; text-align:center'>
        <div style='font-size:48px; margin-bottom:16px'>📄</div>
        <h2 style='color:#ececec; font-weight:500; margin-bottom:8px'>
            Research Assistant
        </h2>
        <p style='color:#8e8ea0; font-size:15px; max-width:400px; line-height:1.6'>
            Upload a research paper from the sidebar and ask questions.
            Get grounded answers with exact page citations.
        </p>
        <div style='margin-top:24px; display:flex; gap:12px; flex-wrap:wrap;
                    justify-content:center'>
            <div style='background:#2f2f2f; border-radius:10px; padding:10px 16px;
                        font-size:13px; color:#ececec'>
                📝 Summarise the paper
            </div>
            <div style='background:#2f2f2f; border-radius:10px; padding:10px 16px;
                        font-size:13px; color:#ececec'>
                🔍 What dataset was used?
            </div>
            <div style='background:#2f2f2f; border-radius:10px; padding:10px 16px;
                        font-size:13px; color:#ececec'>
                📊 What are the main findings?
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif not active["messages"] and active["paper"]:
    st.markdown(f"""
    <div style='display:flex; flex-direction:column; align-items:center;
                justify-content:center; height:55vh; text-align:center'>
        <div style='font-size:40px; margin-bottom:16px'>✅</div>
        <h3 style='color:#ececec; font-weight:500; margin-bottom:6px'>
            {active["paper"]} is ready
        </h3>
        <p style='color:#8e8ea0; font-size:14px'>
            Ask anything about this paper below
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Render chat messages
    for msg in active["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📚 View sources"):
                    for s in msg["sources"]:
                        st.markdown(f"**{s['file']}** — Page {s['page']}")
                        st.code(s["snippet"], language=None)

# ── Chat input ─────────────────────────────────────────────────
if not active["paper"]:
    placeholder = "Upload a paper first to ask questions..."
    disabled = False
else:
    placeholder = f"Ask about {active['paper']}..."
    disabled = False

if user_input := st.chat_input(placeholder, disabled=disabled):
    if not active["paper"]:
        st.warning("Please upload a PDF first using the sidebar.")
        st.stop()

    # Add user message
    active["messages"].append({"role": "user", "content": user_input})

    # Update session title from first question
    if len(active["messages"]) == 1:
        update_session_title(active["id"], user_input)

    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Determine paper filter
    paper_filter = None
    scope_key = f"scope_{active['id']}"
    if scope_key in st.session_state:
        if st.session_state[scope_key] == "This paper only":
            paper_filter = active["paper"]

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = query(user_input, paper_filter=paper_filter)

        st.markdown(result["answer"])

        if result["sources"]:
            with st.expander("📚 View sources"):
                for s in result["sources"]:
                    st.markdown(f"**{s['file']}** — Page {s['page']}")
                    st.code(s["snippet"], language=None)

    # Save assistant message
    active["messages"].append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })

    st.rerun()
