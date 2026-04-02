# ================================================================
# FILE: main.py — Complete FastAPI application
# ================================================================

# main.py
# Run locally: uvicorn main:app --reload --port 8000
# Deploy:      Railway reads Procfile automatically

import json
import logging
import pickle
from typing import List, Optional

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
    level  =logging.INFO,
    format ="%(asctime)s  %(name)s  %(levelname)s  %(message)s"
)
log = logging.getLogger("main")

# ── Firebase init ──────────────────────────────────────────────
# Fix: use _apps check so memory.py import does not cause
# "default Firebase app already exists" error
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

# ── Qdrant init ────────────────────────────────────────────────
# Fix: force REST over gRPC and add timeout
# gRPC caused [Errno -2] Name or service not known on Railway
# REST is more reliable across cloud providers
from qdrant_client import QdrantClient
from config import QDRANT_URL, QDRANT_API_KEY
qdrant = QdrantClient(
    url         =QDRANT_URL,
    api_key     =QDRANT_API_KEY,
    prefer_grpc =False,
    timeout     =30
)

# ── Load TCN ───────────────────────────────────────────────────
with open(EXPORT_DIR + "tcn_config.json")    as f: tcn_cfg = json.load(f)
with open(EXPORT_DIR + "target_mapping.json") as f:
    target_mapping = {int(k): v for k, v in json.load(f).items()}

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tcn_model = TCN(**tcn_cfg).to(device)
tcn_model.load_state_dict(
    torch.load(EXPORT_DIR + "tcn_weights.pt", map_location=device))
tcn_model.eval()
log.info("TCN loaded on %s", device)

# ── App ────────────────────────────────────────────────────────
app = FastAPI(title="Psychoai API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Schemas ───────────────────────────────────────────────────
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
    trader_id: str
    message:   str

class SaveTokenRequest(BaseModel):
    trader_id: str
    token:     str


# ── Embedding extraction ───────────────────────────────────────
def extract_embedding(num_w: np.ndarray, cat_w: np.ndarray) -> np.ndarray:
    activation = {}
    def hook(m, i, o):
        activation["e"] = o.detach().cpu()
    h = tcn_model.classifier[1].register_forward_hook(hook)
    nt = torch.tensor(num_w).unsqueeze(0).to(device)
    ct = torch.tensor(cat_w).unsqueeze(0).to(device)
    with torch.no_grad():
        _ = tcn_model(nt, ct)
    h.remove()
    emb  = activation["e"].numpy()[0]
    norm = emb / (np.linalg.norm(emb) + 1e-9)
    return norm


# ── Qdrant retrieval ───────────────────────────────────────────
def retrieve_similar(embedding: np.ndarray, top_k: int = 3) -> list:
    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector   =embedding.tolist(),
        limit          =top_k
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


# ── Background coaching task (async — does NOT block response) ─
async def coaching_background_task(
    trader_id:       str,
    pred_label:      str,
    confidence:      float,
    feature_signals: list,
    retrieved_cases: list,
    fcm_token:       Optional[str],
):
    """
    This implements the ASYNC DECOUPLED ARCHITECTURE:
    TCN inference (Thread 1) returns immediately.
    Coaching generation (Thread 2) runs here in background.
    Trader receives push notification when coaching is ready.
    """
    try:
        trader_memory = load_trader_memory(trader_id)

        coaching = generate_coaching(
            trader_id, pred_label, confidence,
            feature_signals, retrieved_cases, trader_memory
        )

        # Save to Firestore (memory + result storage)
        save_trader_session(trader_id, pred_label, confidence, coaching)

        db.collection("coaching_results").add({
            "trader_id":      trader_id,
            "timestamp":      __import__("datetime").datetime.utcnow().isoformat(),
            "pattern":        pred_label,
            "confidence":     confidence,
            "coaching":       coaching,
            "signals":        feature_signals,
            "retrieved":      retrieved_cases,
        })

        # Send push notification
        if fcm_token:
            send_coaching_notification(
                fcm_token, pred_label, coaching, confidence)

        log.info("Coaching complete for trader %s: %s", trader_id, pred_label)

    except Exception as e:
        log.error("coaching_background_task error for %s: %s", trader_id, e)


# ── ENDPOINT 1: Analyze trade (main endpoint) ─────────────────
@app.post("/analyze_trade", response_model=AnalyzeResponse)
async def analyze_trade(
    req: AnalyzeRequest,
    background_tasks: BackgroundTasks
):
    """
    THREAD 1 — Real-time TCN inference (<50ms).
    Returns immediately with prediction + similar traders.
    Coaching is generated asynchronously in background.
    """
    try:
        trade_dict   = req.trade.model_dump()
        history_list = [h.model_dump() for h in req.history] \
                       if req.history else None

        num_w, cat_w, num_vec = build_window(trade_dict, history_list)

        # TCN inference — Thread 1, real-time
        nt = torch.tensor(num_w).unsqueeze(0).to(device)
        ct = torch.tensor(cat_w).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = tcn_model(nt, ct)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_class  = int(probs.argmax())
        confidence  = float(probs[pred_class])
        pred_label  = target_mapping[pred_class]
        signals     = get_feature_signals(num_vec)

        # Qdrant retrieval
        emb      = extract_embedding(num_w, cat_w)
        similar  = retrieve_similar(emb, top_k=3)

        # Schedule coaching on background thread (Thread 2)
        coaching_pending = False
        if confidence >= COACHING_THRESHOLD:
            coaching_pending = True
            background_tasks.add_task(
                coaching_background_task,
                req.trader_id, pred_label, confidence,
                signals, similar, req.fcm_token
            )

        msg = ("Coaching is being generated and will arrive via notification."
               if coaching_pending else
               "Confidence below threshold — no coaching triggered.")

        return AnalyzeResponse(
            trader_id         =req.trader_id,
            predicted_mistake =pred_label,
            confidence        =round(confidence, 4),
            coaching_pending  =coaching_pending,
            feature_signals   =signals,
            similar_traders   =similar,
            message           =msg,
        )

    except Exception as e:
        log.error("analyze_trade error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── ENDPOINT 2: Get latest coaching result ────────────────────
@app.get("/coaching/{trader_id}/latest")
async def get_latest_coaching(trader_id: str):
    """
    App calls this after receiving FCM push notification.
    Returns the full coaching text for display in app.
    """
    try:
        docs = (db.collection("coaching_results")
                  .where("trader_id", "==", trader_id)
                  .order_by("timestamp",
                            direction=firestore.Query.DESCENDING)
                  .limit(1)
                  .stream())

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


# ── ENDPOINT 3: Trader profile / memory ───────────────────────
@app.get("/trader/{trader_id}/profile")
async def get_trader_profile(trader_id: str):
    """Returns trader pattern history for in-app display."""
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


# ── ENDPOINT 4: Chat with Plutus ──────────────────────────────
@app.post("/chat")
async def chat_with_plutus(req: ChatRequest):
    """
    Conversational endpoint. Trader can discuss their psychology
    with Plutus directly. LLM remembers trader history.
    """
    try:
        trader_memory = load_trader_memory(req.trader_id)
        response      = generate_chat_response(
            req.trader_id, req.message, trader_memory)
        return {"response": response, "trader_id": req.trader_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── ENDPOINT 5: Save FCM token ────────────────────────────────
@app.post("/fcm_token")
async def update_fcm_token(req: SaveTokenRequest):
    """Called by Kotlin app when FCM token refreshes."""
    try:
        save_fcm_token(req.trader_id, req.token)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── ENDPOINT 6: Health check ──────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status":             "ok",
        "device":             str(device),
        "coaching_threshold": COACHING_THRESHOLD,
        "version":            "1.0.0",
        # Debug — shows what Qdrant URL Railway is actually reading
        # Remove this after fixing the connection issue
        "qdrant_url":         QDRANT_URL[:50] if QDRANT_URL else "NOT SET"
    }


# ── Startup ───────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    start_scheduler()
    log.info("Psychoai backend started on %s", device)