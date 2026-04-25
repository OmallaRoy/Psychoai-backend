# ================================================================
# FILE: main.py
# Run locally:  uvicorn main:app --reload --port 8000
# Deploy:       Railway (Procfile: web: uvicorn main:app --host 0.0.0.0 --port $PORT)
#
# v1.1: ChatRequest accepts messages list (stateless fix)
# v1.2: Tavily dual RAG integrated
# v1.3: Mandatory 3-step coaching prompt
# v1.4: History query fix (no composite Firestore index needed)
# v1.5: Session-based chat history + auto-generated titles
#        + username passed to Plutus so it calls trader by name
#        New endpoints:
#          GET  /trader/{id}/sessions  — list of session cards for drawer
#          GET  /session/{session_id}  — full session for restore/continue
# ================================================================

import json
import logging
import datetime
from typing import List, Optional, Dict

import numpy as np
import torch
import firebase_admin
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import credentials, firestore
from pydantic import BaseModel

from config import (EXPORT_DIR, COLLECTION_NAME,
                    FIREBASE_CREDS_PATH, FIREBASE_CREDS_DICT,
                    COACHING_THRESHOLD, WINDOW_SIZE)
from models import TCN
from feature_engineering import (
    build_window, get_feature_signals,
    numerical_features, categorical_features
)
from coaching import (
    generate_coaching,
    generate_chat_response,
    generate_session_title,
)
from memory import (load_trader_memory, save_trader_session, save_fcm_token)
from notifications import send_coaching_notification, start_scheduler

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(name)s  %(levelname)s  %(message)s"
)
log = logging.getLogger("main")

# ── Firebase ───────────────────────────────────────────────────
if not firebase_admin._apps:
    if FIREBASE_CREDS_DICT:
        cred = credentials.Certificate(FIREBASE_CREDS_DICT)
    else:
        cred = credentials.Certificate(FIREBASE_CREDS_PATH)
    firebase_admin.initialize_app(cred)
db = firestore.client()

# ── Qdrant ─────────────────────────────────────────────────────
from qdrant_client import QdrantClient
from config import QDRANT_URL, QDRANT_API_KEY
qdrant = QdrantClient(
    url         = QDRANT_URL,
    api_key     = QDRANT_API_KEY,
    prefer_grpc = False,
    timeout     = 30
)

# ── TCN model ──────────────────────────────────────────────────
with open(EXPORT_DIR + "tcn_config.json") as f:
    tcn_cfg = json.load(f)
with open(EXPORT_DIR + "target_mapping.json") as f:
    target_mapping = {int(k): v for k, v in json.load(f).items()}

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tcn_model = TCN(**tcn_cfg).to(device)
tcn_model.load_state_dict(
    torch.load(EXPORT_DIR + "tcn_weights.pt", map_location=device))
tcn_model.eval()
log.info("TCN loaded on %s", device)

# ── FastAPI ─────────────────────────────────────────────────────
app = FastAPI(title="Psychoai API", version="1.5.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"]
)


# ── Pydantic schemas ───────────────────────────────────────────
class TradeData(BaseModel):
    session:          str
    pair:             str
    direction:        str
    lot_size:         float
    entry_price:      float
    risk_percentage:  float
    risk_to_reward:   str
    market_condition: str
    emotion_before:   str
    stop_loss_used:   bool
    pre_trade_plan:   str
    hour:             int
    day_of_week:      int
    is_night:         int = 0

class AnalyzeRequest(BaseModel):
    trader_id: str
    fcm_token: Optional[str] = None
    trade:     TradeData
    history:   Optional[List[TradeData]] = None

class AnalyzeResponse(BaseModel):
    trader_id:          str
    predicted_mistake:  str
    confidence:         float
    coaching_pending:   bool
    feature_signals:    List[str]
    similar_traders:    List[dict]
    message:            str

class ChatRequest(BaseModel):
    trader_id:    str
    message:      str
    messages:     List[Dict[str, str]] = []
    last_pattern: str = ""
    # v1.5: session_id groups all messages in one conversation into
    # a single Firestore document so the drawer shows one card per
    # session instead of one card per message.
    session_id:   str = ""
    # v1.5: display name from Firebase so Plutus calls trader by name
    username:     str = ""

class SaveTokenRequest(BaseModel):
    trader_id: str
    token:     str


# ── Embedding extraction ───────────────────────────────────────
def extract_embedding(num_w: np.ndarray, cat_w: np.ndarray) -> np.ndarray:
    activation = {}
    def hook(m, i, o):
        activation["e"] = o.detach().cpu()
    h  = tcn_model.classifier[1].register_forward_hook(hook)
    nt = torch.tensor(num_w).unsqueeze(0).to(device)
    ct = torch.tensor(cat_w).unsqueeze(0).to(device)
    with torch.no_grad():
        _ = tcn_model(nt, ct)
    h.remove()
    emb  = activation["e"].numpy()[0]
    return emb / (np.linalg.norm(emb) + 1e-9)


# ── Qdrant retrieval ───────────────────────────────────────────
def retrieve_similar(embedding: np.ndarray, top_k: int = 3) -> list:
    hits = qdrant.search(
        collection_name = COLLECTION_NAME,
        query_vector    = embedding.tolist(),
        limit           = top_k
    )
    return [
        {
            "rank":       i + 1,
            "trader_id":  h.payload["trader_id"],
            "last_date":  h.payload["last_date"],
            "true_label": h.payload["true_label"],
            "similarity": round(h.score, 3),
        }
        for i, h in enumerate(hits)
    ]


# ── Session Firestore helpers ──────────────────────────────────
def _session_ref(session_id: str):
    """Return Firestore DocumentReference for a session."""
    return db.collection("chat_sessions_v2").document(session_id)


def _upsert_session(
    session_id:   str,
    trader_id:    str,
    user_message: str,
    ai_response:  str,
    title:        str = "",
):
    """
    Create or update a session document.

    Structure:
      chat_sessions_v2/{session_id}:
        trader_id:  str
        title:      str  (auto-generated or "New Chat")
        date:       str  (YYYY-MM-DD of session start)
        created_at: str  (ISO timestamp, set once)
        updated_at: str  (ISO timestamp, updated every message)
        messages:   [ {role, content, timestamp}, ... ]

    We append the new user + assistant messages every call.
    The title is only updated when a non-empty title is provided
    AND the existing title is still the default placeholder.
    """
    now       = datetime.datetime.utcnow().isoformat()
    today_str = now[:10]

    ref = _session_ref(session_id)
    doc = ref.get()

    new_msgs = [
        {"role": "user",      "content": user_message, "timestamp": now},
        {"role": "assistant", "content": ai_response,  "timestamp": now},
    ]

    if doc.exists:
        data     = doc.to_dict()
        existing = data.get("messages", [])
        existing.extend(new_msgs)

        update_payload = {
            "messages":   existing,
            "updated_at": now,
        }
        # Only overwrite title if we got a real one and current is placeholder
        if title and data.get("title", "") in ("", "New Chat", "Trading Psychology Session"):
            update_payload["title"] = title

        ref.update(update_payload)
    else:
        ref.set({
            "session_id": session_id,
            "trader_id":  trader_id,
            "title":      title if title else "New Chat",
            "date":       today_str,
            "created_at": now,
            "updated_at": now,
            "messages":   new_msgs,
        })


def _session_to_dict(doc_data: dict) -> dict:
    """Normalize a Firestore session document for API response."""
    return {
        "session_id": doc_data.get("session_id", ""),
        "trader_id":  doc_data.get("trader_id",  ""),
        "title":      doc_data.get("title",      "Trading Psychology Session"),
        "date":       doc_data.get("date",        ""),
        "created_at": doc_data.get("created_at",  ""),
        "updated_at": doc_data.get("updated_at",  ""),
        "messages":   doc_data.get("messages",    []),
    }


# ── Background coaching task ───────────────────────────────────
async def coaching_background_task(
    trader_id:       str,
    pred_label:      str,
    confidence:      float,
    feature_signals: list,
    retrieved_cases: list,
    fcm_token:       Optional[str],
):
    try:
        trader_memory = load_trader_memory(trader_id)
        coaching      = generate_coaching(
            trader_id, pred_label, confidence,
            feature_signals, retrieved_cases, trader_memory
        )
        save_trader_session(trader_id, pred_label, confidence, coaching)
        db.collection("coaching_results").add({
            "trader_id":  trader_id,
            "timestamp":  datetime.datetime.utcnow().isoformat(),
            "pattern":    pred_label,
            "confidence": confidence,
            "coaching":   coaching,
            "signals":    feature_signals,
            "retrieved":  retrieved_cases,
        })
        if fcm_token:
            send_coaching_notification(
                fcm_token, pred_label, coaching, confidence)
        log.info("Coaching complete for %s: %s", trader_id, pred_label)
    except Exception as e:
        log.error("coaching_background_task error for %s: %s", trader_id, e)


# ════════════════════════════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════════════════════════════

# ── 1. Analyze trade ───────────────────────────────────────────
@app.post("/analyze_trade", response_model=AnalyzeResponse)
async def analyze_trade(req: AnalyzeRequest, background_tasks: BackgroundTasks):
    try:
        trade_dict   = req.trade.model_dump()
        history_list = [h.model_dump() for h in req.history] if req.history else None

        num_w, cat_w, num_vec = build_window(trade_dict, history_list)

        nt = torch.tensor(num_w).unsqueeze(0).to(device)
        ct = torch.tensor(cat_w).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = tcn_model(nt, ct)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_class = int(probs.argmax())
        confidence = float(probs[pred_class])
        pred_label = target_mapping[pred_class]
        signals    = get_feature_signals(num_vec)
        emb        = extract_embedding(num_w, cat_w)
        similar    = retrieve_similar(emb, top_k=3)

        coaching_pending = False
        if confidence >= COACHING_THRESHOLD:
            coaching_pending = True
            background_tasks.add_task(
                coaching_background_task,
                req.trader_id, pred_label, confidence,
                signals, similar, req.fcm_token
            )

        return AnalyzeResponse(
            trader_id         = req.trader_id,
            predicted_mistake = pred_label,
            confidence        = round(confidence, 4),
            coaching_pending  = coaching_pending,
            feature_signals   = signals,
            similar_traders   = similar,
            message           = (
                "Coaching is being generated and will arrive via notification."
                if coaching_pending else
                "Confidence below threshold — no coaching triggered."
            ),
        )
    except Exception as e:
        log.error("analyze_trade error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── 2. Get latest coaching result ─────────────────────────────
@app.get("/coaching/{trader_id}/latest")
async def get_latest_coaching(trader_id: str):
    try:
        docs = (
            db.collection("coaching_results")
              .where("trader_id", "==", trader_id)
              .order_by("timestamp", direction=firestore.Query.DESCENDING)
              .limit(1)
              .stream()
        )
        for doc in docs:
            d = doc.to_dict()
            return {
                "pattern":    d["pattern"],
                "confidence": d["confidence"],
                "coaching":   d["coaching"],
                "signals":    d.get("signals", []),
                "timestamp":  d["timestamp"],
            }
        return {"message": "No coaching available yet"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 3. Trader profile ──────────────────────────────────────────
@app.get("/trader/{trader_id}/profile")
async def get_trader_profile(trader_id: str):
    try:
        doc = db.collection("trader_profiles").document(trader_id).get()
        if not doc.exists:
            return {"trader_id": trader_id, "sessions": 0, "history": []}
        profile = doc.to_dict()
        history = profile.get("pattern_history", [])
        return {"trader_id": trader_id, "sessions": len(history), "history": history[-10:]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 4. Unified history (coaching sessions) ─────────────────────
@app.get("/trader/{trader_id}/history")
async def get_full_history(trader_id: str):
    """
    Returns coaching sessions from trader_profiles for the history view.
    Chat sessions are now in /trader/{id}/sessions (v1.5).
    """
    try:
        items = []
        doc = db.collection("trader_profiles").document(trader_id).get()
        if doc.exists:
            for entry in doc.to_dict().get("pattern_history", []):
                items.append({
                    "type":       "coaching",
                    "date":       entry.get("date", ""),
                    "timestamp":  entry.get("timestamp", ""),
                    "title":      entry.get("pattern", ""),
                    "preview":    entry.get("coaching_snippet", ""),
                    "pattern":    entry.get("pattern", ""),
                    "confidence": entry.get("confidence", 0.0),
                })
        items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return {"trader_id": trader_id, "items": items[:30]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 5. Chat sessions list (v1.5 — for history drawer) ─────────
@app.get("/trader/{trader_id}/sessions")
async def get_chat_sessions(trader_id: str):
    """
    Returns all chat sessions for this trader, sorted newest first.
    Each session has one card in the Android history drawer.
    Only metadata + last assistant message snippet is returned here
    to keep the payload small. Full messages come from /session/{id}.
    """
    try:
        # Fetch without order_by to avoid composite index requirement
        docs = (
            db.collection("chat_sessions_v2")
              .where("trader_id", "==", trader_id)
              .limit(50)
              .stream()
        )

        sessions = []
        for doc in docs:
            d = doc.to_dict()
            # Return metadata only — no full messages list here
            # to keep response size small
            messages = d.get("messages", [])
            sessions.append({
                "session_id": d.get("session_id", doc.id),
                "trader_id":  d.get("trader_id",  ""),
                "title":      d.get("title",       "Trading Psychology Session"),
                "date":       d.get("date",        ""),
                "created_at": d.get("created_at",  ""),
                "updated_at": d.get("updated_at",  ""),
                # Include messages so Android can show preview and restore
                # short sessions without a second network call
                "messages":   messages,
            })

        # Sort newest first in Python (no Firestore index needed)
        sessions.sort(
            key     = lambda x: x.get("updated_at", ""),
            reverse = True
        )

        return {"trader_id": trader_id, "sessions": sessions}

    except Exception as e:
        log.error("get_chat_sessions error for %s: %s", trader_id, e)
        raise HTTPException(status_code=500, detail=str(e))


# ── 6. Single session (v1.5 — for restore/continue) ───────────
@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """
    Returns the full session document including all messages.
    Called when a user taps a session card in the history drawer
    to restore and continue the conversation.
    """
    try:
        doc = _session_ref(session_id).get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Session not found")
        return _session_to_dict(doc.to_dict())
    except HTTPException:
        raise
    except Exception as e:
        log.error("get_session error for %s: %s", session_id, e)
        raise HTTPException(status_code=500, detail=str(e))


# ── 7. Chat with Plutus ────────────────────────────────────────
@app.post("/chat")
async def chat_with_plutus(req: ChatRequest):
    """
    v1.1: Full conversation history (stateless fix)
    v1.2: Tavily smart web search
    v1.3: Mandatory 3-step coaching prompt
    v1.5: session_id groups messages into one Firestore doc per session.
          username passed to coaching so Plutus uses trader's real name.
          Auto-generated title returned on first exchange of each session.
    """
    try:
        trader_memory = load_trader_memory(req.trader_id)

        # v1.5: pass username so Plutus calls trader by name
        response_text = generate_chat_response(
            trader_id     = req.trader_id,
            username      = req.username,
            user_message  = req.message,
            trader_memory = trader_memory,
            messages      = req.messages,
            last_pattern  = req.last_pattern,
        )

        # ── Session persistence (v1.5) ─────────────────────────
        # Generate a title only on the first message of a new session.
        # We detect "first message" by checking if the session doc
        # already exists in Firestore.
        title = ""
        if req.session_id:
            is_new_session = not _session_ref(req.session_id).get().exists

            if is_new_session:
                # Generate title from the first user message
                title = generate_session_title(req.message)
                log.info("Generated session title: '%s' for session %s",
                         title, req.session_id[:8])

            _upsert_session(
                session_id   = req.session_id,
                trader_id    = req.trader_id,
                user_message = req.message,
                ai_response  = response_text,
                title        = title,
            )

        return {
            "response":   response_text,
            "trader_id":  req.trader_id,
            # Return title only on first message so Android can update
            # the drawer card immediately without a separate fetch
            "title":      title,
        }

    except Exception as e:
        log.error("/chat error for %s: %s", req.trader_id, e)
        raise HTTPException(status_code=500, detail=str(e))


# ── 8. Save FCM token ──────────────────────────────────────────
@app.post("/fcm_token")
async def update_fcm_token(req: SaveTokenRequest):
    try:
        save_fcm_token(req.trader_id, req.token)
        log.info("FCM token saved for %s", req.trader_id)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 9. Health check ────────────────────────────────────────────
@app.get("/health")
async def health():
    from config import TAVILY_API_KEY
    return {
        "status":  "ok",
        "device":  str(device),
        "version": "1.5.0",
        "features": {
            "conversation_history": True,
            "web_rag_tavily":       bool(TAVILY_API_KEY),
            "qdrant_rag":           True,
            "fcm_notifications":    True,
            "topic_restriction":    True,
            "session_history":      True,
            "username_addressing":  True,
        }
    }


# ── Startup ────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    start_scheduler()
    from config import TAVILY_API_KEY
    log.info("Psychoai backend v1.5.0 started on %s", device)
    log.info("Tavily: %s", "ENABLED" if TAVILY_API_KEY else "DISABLED")
    log.info("Session history: ENABLED (chat_sessions_v2 collection)")
    log.info("Username addressing: ENABLED")