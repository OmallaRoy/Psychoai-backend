# ================================================================
# FILE: memory.py
# Firestore-based trader memory — gives LLM persistent context
# ================================================================

import logging
from datetime import datetime
import firebase_admin
from firebase_admin import firestore, credentials
from config import MAX_HISTORY_SESSIONS, FIREBASE_CREDS_PATH, FIREBASE_CREDS_DICT

log = logging.getLogger("memory")

# Only initialize Firebase if main.py has not already done it
# Fix: prevents "default Firebase app already exists" error
# Supports both local file and Railway environment variable
if not firebase_admin._apps:
    if FIREBASE_CREDS_DICT:
        # Railway — credentials from GOOGLE_CREDENTIALS env variable
        cred = credentials.Certificate(FIREBASE_CREDS_DICT)
    else:
        # Local — credentials from serviceAccountKey.json file
        cred = credentials.Certificate(FIREBASE_CREDS_PATH)
    firebase_admin.initialize_app(cred)

db = firestore.client()


def load_trader_memory(trader_id: str) -> str:
    """
    Load trader behavioral history from Firestore.
    Returns a formatted string injected into the Groq prompt.
    This is what gives Plutus persistent memory across sessions.
    """
    try:
        doc = db.collection("trader_profiles").document(trader_id).get()
        if not doc.exists:
            return "No previous sessions recorded for this trader."

        profile  = doc.to_dict()
        patterns = profile.get("pattern_history", [])
        if not patterns:
            return "First coaching session for this trader."

        recent = patterns[-MAX_HISTORY_SESSIONS:]
        lines  = [
            "  - {}: {} ({:.0f}% confidence)".format(
                p.get("date", "unknown"),
                p.get("pattern", "unknown"),
                p.get("confidence", 0) * 100)
            for p in recent
        ]

        all_patterns = [p.get("pattern", "") for p in patterns]
        if all_patterns:
            most_common = max(
                set(all_patterns),
                key=lambda x: all_patterns.count(x))
            recurring = "Most recurring pattern overall: {}".format(most_common)
        else:
            recurring = ""

        return (
            "Last {} sessions:\\n{}\\n{}"
        ).format(len(recent), "\\n".join(lines), recurring)

    except Exception as e:
        log.error("load_trader_memory error for %s: %s", trader_id, e)
        return "History unavailable."


def save_trader_session(
    trader_id: str,
    pattern: str,
    confidence: float,
    coaching: str,
):
    """
    Save a coaching session to Firestore.
    Called after every successful coaching generation.
    Keeps last 50 sessions per trader.
    """
    try:
        doc_ref = db.collection("trader_profiles").document(trader_id)
        doc     = doc_ref.get()

        entry = {
            "date":             datetime.utcnow().isoformat()[:10],
            "pattern":          pattern,
            "confidence":       round(float(confidence), 4),
            "coaching_snippet": coaching[:200] if coaching else "",
            "timestamp":        datetime.utcnow().isoformat(),
        }

        if doc.exists:
            existing = doc.to_dict().get("pattern_history", [])
            existing.append(entry)
            doc_ref.update({"pattern_history": existing[-50:]})
        else:
            doc_ref.set({
                "trader_id":       trader_id,
                "created_at":      datetime.utcnow().isoformat(),
                "pattern_history": [entry],
            })

        log.info("Saved session for trader %s: %s", trader_id, pattern)

    except Exception as e:
        log.error("save_trader_session error for %s: %s", trader_id, e)


def save_fcm_token(trader_id: str, token: str):
    """Save or update FCM device token for push notifications."""
    try:
        db.collection("fcm_tokens").document(trader_id).set({
            "token":      token,
            "updated_at": datetime.utcnow().isoformat(),
        })
    except Exception as e:
        log.error("save_fcm_token error: %s", e)


def get_fcm_token(trader_id: str):
    """Retrieve FCM token for a trader."""
    try:
        doc = db.collection("fcm_tokens").document(trader_id).get()
        if doc.exists:
            return doc.to_dict().get("token")
        return None
    except Exception as e:
        log.error("get_fcm_token error: %s", e)
        return None


def get_weekly_summary(trader_id: str) -> dict:
    """
    Get last 7 days pattern summary for daily notification.
    Returns dict with patterns, most_common, count, total.
    """
    from datetime import timedelta

    try:
        doc = db.collection("trader_profiles").document(trader_id).get()
        if not doc.exists:
            return {}

        profile   = doc.to_dict()
        history   = profile.get("pattern_history", [])
        cutoff    = (datetime.utcnow() - timedelta(days=7)).isoformat()[:10]
        recent    = [h for h in history if h.get("date", "") >= cutoff]

        if not recent:
            return {}

        patterns    = [r["pattern"] for r in recent]
        most_common = max(set(patterns), key=patterns.count)

        return {
            "patterns":    recent,
            "most_common": most_common,
            "count":       patterns.count(most_common),
            "total":       len(recent),
        }

    except Exception as e:
        log.error("get_weekly_summary error: %s", e)
        return {}