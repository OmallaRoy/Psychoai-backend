import os
import json as _json
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY         = os.getenv("GROQ_API_KEY")

# Fix: strip any accidental whitespace or equals signs from QDRANT_URL
# Railway variable was being stored with leading " =" prefix
# causing [Errno -2] Name or service not known DNS failure
_raw_qdrant_url = os.getenv("QDRANT_URL", "")
QDRANT_URL      = _raw_qdrant_url.strip().lstrip("=").strip()

QDRANT_API_KEY       = os.getenv("QDRANT_API_KEY")
COACHING_THRESHOLD   = float(os.getenv("COACHING_THRESHOLD", "0.55"))
COLLECTION_NAME      = os.getenv("COLLECTION_NAME", "Psychoai")
MAX_HISTORY_SESSIONS = int(os.getenv("MAX_HISTORY_SESSIONS", "5"))
EXPORT_DIR           = "./production_export/"
WINDOW_SIZE          = 50
EMBED_DIM            = 64
NOTIFICATION_HOUR    = "08:00"   # daily notification time

# Firebase credentials — supports two modes:
# Local development: reads from serviceAccountKey.json file
# Railway production: reads from GOOGLE_CREDENTIALS environment variable
# This avoids pushing the private key file to GitHub
_firebase_creds_raw = os.getenv("GOOGLE_CREDENTIALS")
if _firebase_creds_raw:
    # Railway — parse JSON string stored as environment variable
    FIREBASE_CREDS_DICT = _json.loads(_firebase_creds_raw)
    FIREBASE_CREDS_PATH = None
else:
    # Local — use file path from .env or default
    FIREBASE_CREDS_DICT = None
    FIREBASE_CREDS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH", "./serviceAccountKey.json")