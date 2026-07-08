import warnings
warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv

load_dotenv()

# Works locally (.env) AND on Streamlit Cloud (st.secrets)
def get_secret(key: str) -> str:
    # Try environment variable first (.env locally)
    val = os.getenv(key)
    if val:
        return val
    # Try Streamlit secrets (deployed app)
    try:
        import streamlit as st
        return st.secrets[key]
    except Exception:
        return ""

# Groq LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"

# Embedding model (runs locally, GPU-accelerated on your workstation)
EMBED_MODEL = "all-MiniLM-L6-v2"

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval
TOP_K = 5

