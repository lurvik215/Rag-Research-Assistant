import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import os
import time

st.set_page_config(
    page_title="Research Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.stApp { background-color: #212121; }
[data-testid="stSidebar"] {
    background-color: #171717;
    border-right: 1px solid #2f2f2f;
}
[data-testid="stSidebar"] * { color: #ececec !important; }
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
.stChatMessage { background: transparent !important; border: none !important; }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: #2f2f2f !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
}
[data-testid="stChatInput"] {
    background: #2f2f2f !important;
    border: 1px solid #3f3f3f !important;
    border-radius: 16px !important;
}
[data-testid="stExpander"] {
    background: #1a1a1a !important;
    border: 1px solid #2f2f2f !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploader"] {
    background: #1e1e1e !important;
    border: 1px dashed #3f3f3f !important;
    border-radius: 10px !important;
}
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #171717; }
::-webkit-scrollbar-thumb { background: #3f3f3f; border-radius: 3px; }
p, span, label, div { color: #ececec; }
code { background: #2a2a2a !important; color: #a8d8a8 !important; }
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

from src.rag_pipeline import ingest, query, get_indexed_papers

def ensure_papers_ingested(papers: list):
    """
    Ensures papers are in ChromaDB.
    Forces re-ingest if chunks are missing from DB.
    """
    from src.rag_pipeline import _ingested_papers, _collection
    for paper in papers:
        path = f"/tmp/{paper}"
        if not os.path.exists(path):
            continue
        # Check if actually in ChromaDB
        try:
            result = _collection.get(
                where={"source_file": paper}, limit=1
            )
            if not result["ids"]:
                # Chunks missing — force re-ingest
                _ingested_papers.discard(paper)
                ingest(path)
        except Exception:
            _ingested_papers.discard(paper)
            ingest(path)

# ── Session state ─────────────────────────────────────────────
if "sessions" not in st.session_state:
    st.session_state.sessions = []
if "active_session" not in st.session_state:
    st.session_state.active_session = None
if "session_counter" not in st.session_state:
    st.session_state.session_counter = 0


def create_new_session():
    st.session_state.session_counter += 1
    sid = st.session_state.session_counter
    st.session_state.sessions.append({
        "id": sid,
        "title": f"Chat {sid}",
        "messages": [],
        "papers": []          # list of paper filenames
    })
    st.session_state.active_session = sid


def get_active():
    for s in st.session_state.sessions:
        if s["id"] == st.session_state.active_session:
            return s
    return None


def update_title(sid, title):
    for s in st.session_state.sessions:
        if s["id"] == sid:
            s["title"] = title[:32]
            break


if not st.session_state.sessions:
    create_new_session()

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:8px 4px 16px;display:flex;align-items:center;gap:10px'>
        <span style='font-size:20px'>📄</span>
        <span style='font-size:16px;font-weight:600;color:#ececec'>Research Assistant</span>
    </div>""", unsafe_allow_html=True)

    if st.button("✏️  New Chat", use_container_width=True, key="new_chat"):
        create_new_session()
        st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-label'>Your chats</div>",
                unsafe_allow_html=True)

    for session in reversed(st.session_state.sessions):
        is_active = session["id"] == st.session_state.active_session
        n_papers = len(session["papers"])
        icon = f"📚 " if n_papers > 1 else "📄 " if n_papers == 1 else "💬 "
        label = icon + session["title"]

        if is_active:
            st.markdown(f"""
            <div style='background:#2f2f2f;border-radius:8px;padding:8px 12px;
                        font-size:13px;color:#ececec;margin-bottom:4px;
                        border-left:3px solid #10a37f'>
                {label}
            </div>""", unsafe_allow_html=True)
        else:
            if st.button(label, key=f"sess_{session['id']}",
                         use_container_width=True):
                st.session_state.active_session = session["id"]
                st.rerun()

    st.markdown("---")

    active = get_active()
    if active:
        st.markdown("<div class='sidebar-label'>This chat</div>",
                    unsafe_allow_html=True)

        # ── Uploaded papers list ──────────────────────────────
        if active["papers"]:
            st.markdown("""
            <div style='background:#1a2e1a;border:1px solid #2d4a2d;
                        border-radius:8px;padding:8px 12px;margin-bottom:8px'>
            <div style='font-size:11px;color:#8e8ea0;margin-bottom:6px'>
                Uploaded papers
            </div>""", unsafe_allow_html=True)

            for paper in list(active["papers"]):
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.markdown(
                        f"<div style='font-size:12px;color:#a8d8a8;"
                        f"padding:3px 0'>📄 {paper}</div>",
                        unsafe_allow_html=True)
                with c2:
                    if st.button("✕", key=f"rm_{paper}_{active['id']}",
                                 help=f"Remove {paper}"):
                        active["papers"].remove(paper)
                        st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:#1e1e1e;border:1px solid #2f2f2f;
                        border-radius:8px;padding:8px 12px;margin-bottom:8px'>
            <div style='font-size:12px;color:#8e8ea0'>No papers uploaded yet</div>
            </div>""", unsafe_allow_html=True)

        # ── Multi-file uploader ───────────────────────────────
        uploaded_files = st.file_uploader(
            "Upload PDFs (select multiple)",
            type="pdf",
            accept_multiple_files=True,
            key=f"upload_{active['id']}"
        )

        if uploaded_files:
            newly_added = False
            for uf in uploaded_files:
                if uf.name not in active["papers"]:
                    clean_path = f"/tmp/{uf.name}"
                    with open(clean_path, "wb") as f:
                        f.write(uf.getbuffer())
                    with st.status(f"Indexing {uf.name}..."):
                        n = ingest(clean_path)
                    if n > 0:
                        active["papers"].append(uf.name)
                        if len(active["papers"]) == 1:
                            update_title(active["id"],
                                         uf.name.replace(".pdf", ""))
                        st.success(f"✓ {uf.name} — {n} chunks")
                        newly_added = True
                    else:
                        if uf.name not in active["papers"]:
                            active["papers"].append(uf.name)
                            newly_added = True
                        st.info(f"{uf.name} already indexed")
            if newly_added:
                st.rerun()

        # ── Search scope ──────────────────────────────────────
        if active["papers"]:
            st.markdown("<div style='height:6px'></div>",
                        unsafe_allow_html=True)
            scope = st.radio(
                "Search in",
                ["All uploaded papers", "Select specific papers"],
                key=f"scope_{active['id']}"
            )

            if scope == "Select specific papers" and len(active["papers"]) > 1:
                st.multiselect(
                    "Choose papers to search",
                    options=active["papers"],
                    default=active["papers"],
                    key=f"sel_{active['id']}"
                )

        # ── Clear chat ────────────────────────────────────────
        if active["messages"]:
            st.markdown("---")
            if st.button("🗑️  Clear messages",
                         use_container_width=True,
                         key=f"clear_{active['id']}"):
                active["messages"] = []
                st.rerun()

        # ── Model selector ────────────────────────────────────
        st.markdown("---")
        st.markdown("<div class='sidebar-label'>Model</div>",
                    unsafe_allow_html=True)
        st.selectbox(
            "LLM",
            options=[
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "llama-4-scout-preview",
                "qwen3-32b",
            ],
            index=0,
            key="groq_model",
            label_visibility="collapsed"
        )

# ── MAIN AREA ─────────────────────────────────────────────────
active = get_active()
if not active:
    st.stop()

# Re-ingest papers if ChromaDB was reset
if active["papers"]:
    ensure_papers_ingested(active["papers"])

# ── Welcome screen ────────────────────────────────────────────
if not active["messages"] and not active["papers"]:
    st.markdown("""
    <div style='display:flex;flex-direction:column;align-items:center;
                justify-content:center;height:60vh;text-align:center'>
        <div style='font-size:48px;margin-bottom:16px'>📄</div>
        <h2 style='color:#ececec;font-weight:500;margin-bottom:8px'>
            Research Assistant
        </h2>
        <p style='color:#8e8ea0;font-size:15px;max-width:420px;line-height:1.6'>
            Upload one or more research papers and ask questions.
            Get grounded answers with exact page citations.
        </p>
        <div style='margin-top:24px;display:flex;gap:10px;flex-wrap:wrap;
                    justify-content:center'>
            <div style='background:#2f2f2f;border-radius:10px;
                        padding:10px 16px;font-size:13px;color:#ececec'>
                📝 Summarise this paper
            </div>
            <div style='background:#2f2f2f;border-radius:10px;
                        padding:10px 16px;font-size:13px;color:#ececec'>
                🔍 Compare methodologies
            </div>
            <div style='background:#2f2f2f;border-radius:10px;
                        padding:10px 16px;font-size:13px;color:#ececec'>
                📊 What datasets were used?
            </div>
        </div>
        <p style='color:#8e8ea0;font-size:13px;margin-top:20px'>
            👈 Upload PDFs from the sidebar to begin
        </p>
    </div>""", unsafe_allow_html=True)

elif not active["messages"] and active["papers"]:
    n = len(active["papers"])
    paper_list = ", ".join(p.replace(".pdf", "") for p in active["papers"])
    st.markdown(f"""
    <div style='display:flex;flex-direction:column;align-items:center;
                justify-content:center;height:55vh;text-align:center'>
        <div style='font-size:40px;margin-bottom:16px'>✅</div>
        <h3 style='color:#ececec;font-weight:500;margin-bottom:6px'>
            {n} paper{"s" if n > 1 else ""} ready
        </h3>
        <p style='color:#8e8ea0;font-size:14px;max-width:400px'>
            {paper_list}
        </p>
        <p style='color:#8e8ea0;font-size:13px;margin-top:8px'>
            Ask anything about {"these papers" if n > 1 else "this paper"} below
        </p>
    </div>""", unsafe_allow_html=True)

else:
    # ── Title ─────────────────────────────────────────────────
    n = len(active["papers"])
    if n > 1:
        st.title(f"📚 {active['title']} ({n} papers)")
    elif n == 1:
        st.title(f"📄 {active['papers'][0].replace('.pdf','')}")
    else:
        st.title("📄 Research Assistant")
    st.markdown("---")

    # ── Chat history ──────────────────────────────────────────
    for msg in active["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📚 View sources"):
                    for s in msg["sources"]:
                        st.markdown(f"**{s['file']}** — Page {s['page']}")
                        st.code(s["snippet"], language=None)

# ── Chat input ────────────────────────────────────────────────
# Always re-fetch active session at bottom to get latest state
active = get_active()
has_papers = active and len(active.get("papers", [])) > 0

placeholder = (
    f"Ask about {len(active['papers'])} paper(s)..."
    if has_papers else "Upload a paper first..."
)

if user_input := st.chat_input(placeholder):
    active = get_active()  # re-fetch again on submit
    if not active or not active["papers"]:
        st.warning("Please upload at least one PDF first.")
        st.stop()

    # Add user message
    active["messages"].append({"role": "user", "content": user_input})
    if len(active["messages"]) == 1:
        update_title(active["id"], user_input)

    with st.chat_message("user"):
        st.markdown(user_input)

    # Determine which papers to search
    scope_key = f"scope_{active['id']}"
    sel_key   = f"sel_{active['id']}"
    scope     = st.session_state.get(scope_key, "All uploaded papers")

    if scope == "Select specific papers":
        paper_filter = st.session_state.get(sel_key, active["papers"])
        if not paper_filter:
            paper_filter = active["papers"]
    else:
        paper_filter = active["papers"]

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner(f"Searching {len(paper_filter)} paper(s)..."):
            result = query(
                user_input,
                paper_filter=paper_filter,
                model=st.session_state.get("groq_model",
                                           "llama-3.3-70b-versatile")
            )

        st.markdown(result["answer"])

        if result["sources"]:
            # Show which papers were cited
            cited = list({s["file"] for s in result["sources"]})
            if len(cited) > 1:
                st.caption(f"Sources pulled from: {', '.join(cited)}")

            with st.expander("📚 View sources"):
                for s in result["sources"]:
                    st.markdown(f"**{s['file']}** — Page {s['page']}")
                    st.code(s["snippet"], language=None)

    active["messages"].append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })

    st.rerun()
