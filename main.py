# ================================================================
# FILE: main.py
# Run locally:  uvicorn main:app --reload --port 8000
# Deploy:       Railway reads Procfile automatically
#
# FIX: ChatRequest now accepts messages list.
#      /chat endpoint passes full history to generate_chat_response.
# ================================================================

import json
import logging
import pickle
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
from coaching import generate_coaching, generate_chat_response
from memory import (load_trader_memory, save_trader_session,
                    save_fcm_token)
from notifications import send_coaching_notification, start_scheduler

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(name)s  %(levelname)s  %(message)s"
)
log = logging.getLogger("main")

# ── Firebase init ──────────────────────────────────────────────
if not firebase_admin._apps:
    if FIREBASE_CREDS_DICT:
        cred = credentials.Certificate(FIREBASE_CREDS_DICT)
    else:
        cred = credentials.Certificate(FIREBASE_CREDS_PATH)
    firebase_admin.initialize_app(cred)
db = firestore.client()

# ── Qdrant init ────────────────────────────────────────────────
from qdrant_client import QdrantClient
from config import QDRANT_URL, QDRANT_API_KEY
qdrant = QdrantClient(
    url         = QDRANT_URL,
    api_key     = QDRANT_API_KEY,
    prefer_grpc = False,
    timeout     = 30
)

# ── Load TCN ───────────────────────────────────────────────────
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

# ── FastAPI app ────────────────────────────────────────────────
app = FastAPI(title="Psychoai API", version="1.1.0")
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

# ── FIX: ChatRequest now includes messages list ────────────────
# Before: only trader_id and message were sent.
# Groq received one message with no prior context.
# Plutus re-introduced itself on every follow-up reply.
#
# After: messages carries the full conversation history.
# Groq receives the entire back-and-forth and responds correctly
# to follow-ups like "yes", "go ahead", "tell me more".
class ChatRequest(BaseModel):
    trader_id: str
    message:   str
    # Full conversation history from client
    # List of {"role": "user"/"assistant", "content": "..."}
    # Defaults to [] for backwards compatibility with older app versions
    messages:  List[Dict[str, str]] = []

class SaveTokenRequest(BaseModel):
    trader_id: str
    token:     str


# ── Embedding extraction ───────────────────────────────────────
def extract_embedding(num_w: np.ndarray, cat_w: np.ndarray) -> np.ndarray:
    activation = {}

    def hook(m, i, o):
        activation["e"] = o.detach().cpu()

    h   = tcn_model.classifier[1].register_forward_hook(hook)
    nt  = torch.tensor(num_w).unsqueeze(0).to(device)
    ct  = torch.tensor(cat_w).unsqueeze(0).to(device)
    with torch.no_grad():
        _ = tcn_model(nt, ct)
    h.remove()
    emb  = activation["e"].numpy()[0]
    norm = emb / (np.linalg.norm(emb) + 1e-9)
    return norm


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
            "timestamp":  __import__("datetime").datetime.utcnow().isoformat(),
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


# ── ENDPOINT 1: Analyze trade ──────────────────────────────────
@app.post("/analyze_trade", response_model=AnalyzeResponse)
async def analyze_trade(
    req: AnalyzeRequest,
    background_tasks: BackgroundTasks
):
    try:
        trade_dict   = req.trade.model_dump()
        history_list = [h.model_dump() for h in req.history] \
                       if req.history else None

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

        msg = (
            "Coaching is being generated and will arrive via notification."
            if coaching_pending else
            "Confidence below threshold — no coaching triggered."
        )
        return AnalyzeResponse(
            trader_id         = req.trader_id,
            predicted_mistake = pred_label,
            confidence        = round(confidence, 4),
            coaching_pending  = coaching_pending,
            feature_signals   = signals,
            similar_traders   = similar,
            message           = msg,
        )
    except Exception as e:
        log.error("analyze_trade error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── ENDPOINT 2: Get latest coaching ───────────────────────────
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
            data = doc.to_dict()
            return {
                "pattern":    data["pattern"],
                "confidence": data["confidence"],
                "coaching":   data["coaching"],
                "signals":    data.get("signals", []),
                "timestamp":  data["timestamp"],
            }
        return {"message": "No coaching available yet"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── ENDPOINT 3: Trader profile ─────────────────────────────────
@app.get("/trader/{trader_id}/profile")
async def get_trader_profile(trader_id: str):
    try:
        doc = db.collection("trader_profiles").document(trader_id).get()
        if not doc.exists:
            return {"trader_id": trader_id, "sessions": 0, "history": []}
        profile = doc.to_dict()
        history = profile.get("pattern_history", [])
        return {
            "trader_id": trader_id,
            "sessions":  len(history),
            "history":   history[-10:],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── ENDPOINT 4: Unified history ────────────────────────────────
@app.get("/trader/{trader_id}/history")
async def get_full_history(trader_id: str):
    """
    Returns coaching sessions + chat sessions combined for the
    history drawer in the Android app.
    """
    try:
        import datetime
        items = []

        # Coaching sessions from trader_profiles
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

        # Chat sessions
        try:
            chats = (
                db.collection("chat_sessions")
                  .where("trader_id", "==", trader_id)
                  .order_by("timestamp", direction=firestore.Query.DESCENDING)
                  .limit(20)
                  .stream()
            )
            for chat in chats:
                d = chat.to_dict()
                items.append({
                    "type":       "chat",
                    "date":       d.get("session_date", ""),
                    "timestamp":  d.get("timestamp", ""),
                    "title":      "Chat session",
                    "preview":    d.get("user_message", "")[:120],
                    "pattern":    "",
                    "confidence": 0.0,
                })
        except Exception as e:
            log.warning("Could not load chat sessions: %s", e)

        items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return {"trader_id": trader_id, "items": items[:30]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── ENDPOINT 5: Chat with Plutus ───────────────────────────────
# FIX: Now passes req.messages (full conversation history) to
# generate_chat_response so Groq sees the full context.
@app.post("/chat")
async def chat_with_plutus(req: ChatRequest):
    """
    Conversational endpoint. Trader chats with Plutus directly.

    FIX: req.messages now carries the full back-and-forth history
    from the Android client. This is passed to generate_chat_response
    which builds the correct Groq messages array so the model
    sees all prior context and responds accordingly.
    """
    try:
        trader_memory = load_trader_memory(req.trader_id)

        # ── FIX: Pass messages history to coaching.py ──────────
        response = generate_chat_response(
            trader_id     = req.trader_id,
            user_message  = req.message,
            trader_memory = trader_memory,
            messages      = req.messages  # ← THE KEY FIX
        )

        # Save exchange to Firestore for history drawer
        # Only save when the message has enough content to be meaningful
        if len(req.message.strip()) > 5:
            import datetime
            db.collection("chat_sessions").add({
                "trader_id":    req.trader_id,
                "timestamp":    datetime.datetime.utcnow().isoformat(),
                "user_message": req.message,
                "ai_response":  response,
                "session_date": datetime.datetime.utcnow().isoformat()[:10],
            })

        return {"response": response, "trader_id": req.trader_id}

    except Exception as e:
        log.error("/chat error for %s: %s", req.trader_id, e)
        raise HTTPException(status_code=500, detail=str(e))


# ── ENDPOINT 6: Save FCM token ─────────────────────────────────
@app.post("/fcm_token")
async def update_fcm_token(req: SaveTokenRequest):
    try:
        save_fcm_token(req.trader_id, req.token)
        log.info("FCM token saved for trader %s", req.trader_id)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── ENDPOINT 7: Health check ───────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status":             "ok",
        "device":             str(device),
        "coaching_threshold": COACHING_THRESHOLD,
        "version":            "1.1.0",
        "fix":                "conversation history enabled"
    }


# ── Startup ────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    start_scheduler()
    log.info("Psychoai backend v1.1.0 started on %s", device)