
import os
from dotenv import load_dotenv
 
load_dotenv()
 
GROQ_API_KEY            = os.getenv("GROQ_API_KEY")
QDRANT_URL              = os.getenv("QDRANT_URL")
QDRANT_API_KEY          = os.getenv("QDRANT_API_KEY")
FIREBASE_CREDS_PATH     = os.getenv("FIREBASE_CREDENTIALS_PATH", "./serviceAccountKey.json")
COACHING_THRESHOLD      = float(os.getenv("COACHING_THRESHOLD", "0.55"))
COLLECTION_NAME         = os.getenv("COLLECTION_NAME", "Psychoai")
MAX_HISTORY_SESSIONS    = int(os.getenv("MAX_HISTORY_SESSIONS", "5"))
EXPORT_DIR              = "./production_export/"
WINDOW_SIZE             = 50
EMBED_DIM               = 64
NOTIFICATION_HOUR       = "08:00"   # daily notification time
