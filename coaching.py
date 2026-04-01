# ================================================================
# FILE: coaching.py
# Groq-powered coaching with Plutus system prompt
# ================================================================

import logging
from groq import Groq
from config import GROQ_API_KEY
 
log = logging.getLogger("coaching")
groq_client = Groq(api_key=GROQ_API_KEY)
 
PLUTUS_SYSTEM = """You are Plutus, an AI trading psychology coach with deep expertise in behavioral finance and trader psychology.
 
PRECISE BEHAVIORAL DEFINITIONS — apply these exactly and never confuse them:
- Revenge Trading: Entering a NEW trade immediately after a loss, usually with LARGER position size, driven by the urge to emotionally recover losses from the market. Key: it is a NEW position opened after losing.
- Held Loser Too Long: STAYING IN an existing LOSING position for too long, refusing to accept the loss, hoping price will reverse. Key: staying in an existing position, not opening a new one.
- Cut Winner Early: EXITING a WINNING trade too soon, driven by fear of losing the unrealized gain. Key: closed a profitable position before the target was reached.
- FOMO (Fear of Missing Out): Entering a trade AFTER the price has already moved significantly, driven by fear of missing the opportunity. Key: late entry after the move already happened.
- Impulsive Entry: Entering a trade WITHOUT any plan or technical setup, acting purely on emotion in the moment. Key: no pre-trade analysis, no setup.
- No Stop Loss: Trading WITHOUT setting a predefined stop loss level, leaving capital fully exposed to unlimited loss.
- Oversized Position: Risking MORE capital than the trader's own rules allow on a single trade.
- No Mistake: The trade followed the plan completely and correctly — this is disciplined trading.
 
CRITICAL RULES YOU MUST FOLLOW:
1. ONLY reference traders listed in the RETRIEVED HISTORICAL CASES section. Never invent trader IDs, names, or dates.
2. State specifically WHICH SIGNALS from the Feature Signals section triggered this detection.
3. If the trader has a history of this pattern, reference it to show continuity of coaching.
4. Never give price predictions, entry/exit points, or financial recommendations.
5. Never confuse Revenge with Held Loser — they are opposite behaviors.
6. Keep your response to 5-7 sentences. Be warm, direct, and specific."""
 
 
def build_coaching_prompt(
    trader_id: str,
    pred_label: str,
    confidence: float,
    feature_signals: list,
    retrieved_cases: list,
    trader_memory: str,
) -> str:
    sorted_idx_label = pred_label
 
    anomaly_ctx = "Detected pattern: {} (confidence: {:.0f}%)".format(
        pred_label, confidence * 100)
 
    rag_ctx = "\\n".join(
        "  - Rank {}: Trader {} on {} showed pattern \'{}\'".format(
            r["rank"], r["trader_id"], r["last_date"], r["true_label"])
        for r in retrieved_cases
    )
 
    signals_text = "\\n".join(
        "  - {}".format(s) for s in feature_signals)
 
    return (
        "TRADER ID: {}\\n\\n"
        "TRADER BEHAVIORAL HISTORY:\\n{}\\n\\n"
        "CURRENT SESSION — DETECTED PATTERN:\\n{}\\n\\n"
        "FEATURE SIGNALS THAT TRIGGERED THIS DETECTION:\\n{}\\n\\n"
        "RETRIEVED HISTORICAL CASES FROM PEER TRADERS:\\n{}\\n\\n"
        "Write a coaching response (5-7 sentences) that:\\n"
        "1. Names the detected pattern and explains exactly what it means behaviorally.\\n"
        "2. States which specific signals triggered this detection.\\n"
        "3. If the trader history shows this pattern before, reference that to show it is recurring.\\n"
        "4. References at least one specific trader from the Retrieved Historical Cases by their exact ID and date.\\n"
        "5. Gives ONE concrete, specific action the trader can take before their next session.\\n"
        "6. Ends with a reflective question that helps the trader examine their own emotional trigger.\\n"
        "Do not mention confidence scores, model probabilities, or technical ML terms."
    ).format(
        trader_id, trader_memory, anomaly_ctx,
        signals_text, rag_ctx
    )
 
 
def generate_coaching(
    trader_id: str,
    pred_label: str,
    confidence: float,
    feature_signals: list,
    retrieved_cases: list,
    trader_memory: str,
) -> str:
    """
    Call Groq to generate coaching. Returns coaching text string.
    Groq Llama 3.1 8B: ~0.5-1 second latency, free tier.
    """
    user_msg = build_coaching_prompt(
        trader_id, pred_label, confidence,
        feature_signals, retrieved_cases, trader_memory
    )
 
    response = groq_client.chat.completions.create(
        model    = "llama-3.1-8b-instant",
        messages = [
            {"role": "system", "content": PLUTUS_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        temperature = 0.30,
        max_tokens  = 500,
    )
    return response.choices[0].message.content.strip()
 
 
def generate_chat_response(
    trader_id: str,
    user_message: str,
    trader_memory: str,
) -> str:
    """
    Conversational endpoint — trader chats directly with Plutus.
    Plutus knows the trader history and can discuss psychology.
    """
    system = (
        PLUTUS_SYSTEM +
        "\\n\\nYou are in a CONVERSATIONAL SESSION with trader {}.\\n"
        "Their behavioral history:\\n{}\\n\\n"
        "Use this history to make the conversation personal and specific. "
        "Reference their actual patterns when relevant. "
        "Be conversational, warm, and supportive."
    ).format(trader_id, trader_memory)
 
    response = groq_client.chat.completions.create(
        model    = "llama-3.1-8b-instant",
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_message},
        ],
        temperature = 0.35,
        max_tokens  = 400,
    )
    return response.choices[0].message.content.strip()
 
 
def generate_daily_insight(
    trader_id: str,
    recent_patterns: list,
    most_common: str,
    count: int,
    total: int,
) -> str:
    """
    Short motivational message for daily push notification.
    Sent every morning at 8am.
    """
    prompt = (
        "Trader {} has shown the pattern \'{}\' in {}/{} trading sessions this week. "
        "Write a warm, 2-sentence motivational morning message for this trader. "
        "Acknowledge their recurring pattern, give one specific mindset tip for today. "
        "Be encouraging and human — not clinical or technical."
    ).format(trader_id, most_common, count, total)
 
    response = groq_client.chat.completions.create(
        model    = "llama-3.1-8b-instant",
        messages = [
            {"role": "system", "content": "You are Plutus, a warm and supportive trading psychology coach."},
            {"role": "user",   "content": prompt},
        ],
        temperature = 0.45,
        max_tokens  = 150,
    )
    return response.choices[0].message.content.strip()

