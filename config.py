import warnings
warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv

load_dotenv()

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

