import os
from dotenv import load_dotenv

load_dotenv()

# Disable ChromaDB telemetry logs
os.environ["ANONYMIZED_TELEMETRY"] = "False"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_data")

EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.8-flash"

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not set. Set GEMINI_API_KEY in .env")





