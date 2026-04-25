# ================================================================
# FILE: coaching.py  v1.5
# v1.5: Session title generation + username-aware system prompt
# ================================================================

import logging
from groq import Groq
from config import GROQ_API_KEY
from web_search import (
    should_search_web, search_web,
    build_coaching_search_query, build_chat_search_query,
)

log = logging.getLogger("coaching")
groq_client = Groq(api_key=GROQ_API_KEY)

PLUTUS_SYSTEM = """You are Plutus, an expert AI trading psychology coach. \
Your ONLY area of expertise is trading psychology, behavioral finance, \
emotional discipline in trading, and performance coaching for retail traders.

CRITICAL TOPIC BOUNDARY:
You ONLY respond to questions about trading psychology, emotional discipline, \
risk management mindset, behavioral trading patterns, financial markets, and \
the trader's own journal data. If anyone asks about ANYTHING outside this scope \
(food, health, medicine, relationships, sex, sports, politics, general science, \
coding, geography, or any non-trading general knowledge) you MUST respond with:
"I'm Plutus, your trading psychology coach. I can only help with trading \
psychology and emotional discipline in trading. Is there something about \
your trading mindset I can help you with today?"
Never partially answer then redirect. Redirect immediately and completely.

PRECISE BEHAVIORAL DEFINITIONS:
- Revenge Trading: Entering a NEW trade immediately after a loss with LARGER \
size to recover losses emotionally. A new position opened after losing.
- Held Loser Too Long: STAYING IN an existing losing position too long, \
refusing the loss, hoping for reversal. Not opening a new one.
- Cut Winner Early: Exiting a winning trade too soon out of fear of losing \
the unrealized gain before target was reached.
- FOMO: Entering AFTER price already moved significantly, driven by fear of \
missing the move. Late entry after the move happened.
- Impulsive Entry: Entering WITHOUT any plan or setup, purely on emotion. \
No pre-trade analysis whatsoever.
- No Stop Loss: Trading without a predefined stop loss, capital fully exposed.
- Oversized Position: Risking more capital than personal rules allow.
- No Mistake: Trade followed the plan completely. Disciplined trading.

CORE RULES:
1. MAINTAIN CONVERSATION CONTINUITY. Never re-introduce yourself mid-conversation.
2. ONLY reference traders listed in RETRIEVED HISTORICAL CASES. Never invent IDs.
3. Name specific feature signals — never speak in vague generalities.
4. When web context is provided, reference specific facts from it.
5. Never give price predictions, entry/exit points, or financial advice.
6. Never confuse Revenge Trading with Held Loser Too Long.
7. Be warm, direct, specific. Coach, not diagnostician."""


def generate_session_title(first_message: str) -> str:
    """
    Generate a short 4-6 word session title from the user's first message.
    Called once per session after the first exchange — exactly like Claude.
    Returns titles like "Revenge Trading Recovery Plan" or
    "Managing FOMO in Volatile Markets".
    """
    try:
        response = groq_client.chat.completions.create(
            model    = "llama-3.1-8b-instant",
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You generate short session titles for a trading psychology "
                        "coaching app. Given the user's first message, respond with "
                        "ONLY a 4-6 word title capturing the topic. "
                        "No punctuation at the end. No quotes. No explanation. "
                        "Examples: 'Managing Revenge Trading Patterns' | "
                        "'FOMO and Impulsive Entry Help' | "
                        "'Risk Management Mindset Session' | "
                        "'Emotional Discipline After Losses' | "
                        "'Understanding My Trading Mistakes' | "
                        "'Stop Loss Psychology and Discipline'"
                    )
                },
                {
                    "role": "user",
                    "content": f"User's first message: {first_message[:200]}"
                }
            ],
            temperature = 0.4,
            max_tokens  = 20,
        )
        title = response.choices[0].message.content.strip().strip('"\'').rstrip('.!?')
        words = title.split()
        if len(words) > 7:
            title = " ".join(words[:6])
        return title if title else "Trading Psychology Session"
    except Exception as e:
        log.warning("Session title generation failed: %s", e)
        return "Trading Psychology Session"


def build_coaching_prompt(
    trader_id:       str,
    pred_label:      str,
    confidence:      float,
    feature_signals: list,
    retrieved_cases: list,
    trader_memory:   str,
    web_context:     str = None,
) -> str:
    anomaly_ctx  = "Detected pattern: {} (confidence: {:.0f}%)".format(
        pred_label, confidence * 100)

    rag_ctx = "\n".join(
        "  - Rank {rank}: Trader {tid} on {date} showed pattern '{label}'".format(
            rank=r["rank"], tid=r["trader_id"],
            date=r["last_date"], label=r["true_label"])
        for r in retrieved_cases
    ) if retrieved_cases else "  - No peer cases retrieved for this session."

    signals_text = "\n".join(
        "  - {}".format(s) for s in feature_signals
    ) if feature_signals else "  - No specific signals available."

    web_section = (
        "\n\nWEB RESEARCH CONTEXT (live search — published sources):\n{}\n"
        "You MUST reference at least TWO specific facts from above in STEP 3."
        .format(web_context)
    ) if web_context else (
        "\n\nWEB RESEARCH CONTEXT: Not available. "
        "Draw on established behavioral finance research for Step 3."
    )

    history_note = (
        "\n\nIMPORTANT: Trader history shows recurring patterns. "
        "Reference specific dates and patterns to demonstrate ongoing theme."
        if "No previous" not in trader_memory and "First coaching" not in trader_memory
        else ""
    )

    return (
        "TRADER ID: {trader_id}\n\n"
        "TRADER BEHAVIORAL HISTORY:\n{memory}{history_note}\n\n"
        "CURRENT SESSION — TCN DETECTION:\n{anomaly}\n\n"
        "FEATURE SIGNALS THAT TRIGGERED THIS DETECTION:\n{signals}\n\n"
        "RETRIEVED HISTORICAL CASES FROM PEER TRADER DATABASE (Qdrant):\n"
        "{rag_ctx}{web_section}\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "COACHING RESPONSE — FOLLOW THIS EXACT STRUCTURE:\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "STEP 1 — IDENTIFICATION (1 sentence, mandatory):\n"
        "  Name the pattern precisely and explain it behaviorally using the "
        "Feature Signals to specify WHICH behaviors triggered detection.\n\n"
        "STEP 2 — EVIDENCE (1-2 sentences, mandatory):\n"
        "  Name at least one peer trader by exact Trader ID and exact date "
        "from Retrieved Cases. Describe what they experienced. "
        "Never invent or paraphrase trader IDs.\n\n"
        "STEP 3 — SOLUTION (2-3 sentences, mandatory):\n"
        "  Reference TWO specific facts/techniques from Web Research Context. "
        "Give ONE concrete actionable step before next session. "
        "End with a reflective question about the emotional state that preceded "
        "this trade.\n\n"
        "ABSOLUTE RULES:\n"
        "- Total: 5-7 sentences. Never skip Step 2 or 3.\n"
        "- No confidence scores. No ML/AI terminology.\n"
        "- Tone: warm, direct, expert — coach not diagnostician."
    ).format(
        trader_id=trader_id, memory=trader_memory,
        history_note=history_note, anomaly=anomaly_ctx,
        signals=signals_text, rag_ctx=rag_ctx, web_section=web_section,
    )


def build_chat_system_injection(
    trader_id:     str,
    username:      str,
    trader_memory: str,
    web_context:   str = None,
) -> str:
    """v1.5: username parameter — Plutus uses display name not raw UID."""
    web_note = (
        "\n\nLIVE WEB RESEARCH CONTEXT:\n{}\n"
        "Reference specific facts from above when relevant."
        .format(web_context)
    ) if web_context else ""

    # Decide how Plutus addresses the trader
    if username and username.strip() and len(username) < 30:
        address_instruction = (
            "The trader's name is {}. Address them by name occasionally "
            "— naturally, as a coach would, not on every sentence."
            .format(username)
        )
    else:
        address_instruction = (
            "The trader has not set a display name. "
            "Address them as 'you' or 'hey'. "
            "NEVER use the raw ID string as a name."
        )

    return (
        PLUTUS_SYSTEM +
        "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "ACTIVE SESSION CONTEXT\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "{address_instruction}\n\n"
        "Trader behavioral history (reference only if relevant or asked):\n"
        "{memory}"
        "{web_note}\n\n"
        "CONVERSATION RULES:\n"
        "- Maintain continuity at all times. Never reset mid-session.\n"
        "- If the trader answered your question, CONTINUE from there.\n"
        "- Do not volunteer trading history unless they ask.\n"
        "- Be conversational, warm, 3-5 sentences in chat mode.\n"
        "- REDIRECT any off-topic question immediately per topic boundary."
    ).format(
        address_instruction=address_instruction,
        memory=trader_memory,
        web_note=web_note,
    )


def generate_coaching(
    trader_id: str, pred_label: str, confidence: float,
    feature_signals: list, retrieved_cases: list, trader_memory: str,
) -> str:
    web_context = None
    try:
        q = build_coaching_search_query(pred_label, feature_signals)
        web_context = search_web(q, max_results=3)
        if web_context:
            log.info("Web context fetched for pattern: %s", pred_label)
    except Exception as e:
        log.warning("Web search skipped for coaching: %s", e)

    user_msg = build_coaching_prompt(
        trader_id=trader_id, pred_label=pred_label, confidence=confidence,
        feature_signals=feature_signals, retrieved_cases=retrieved_cases,
        trader_memory=trader_memory, web_context=web_context,
    )
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": PLUTUS_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.30, max_tokens=600,
    )
    return response.choices[0].message.content.strip()


def generate_chat_response(
    trader_id:     str,
    username:      str,
    user_message:  str,
    trader_memory: str,
    messages:      list = None,
    last_pattern:  str  = "",
) -> str:
    """v1.5: username parameter for personalised addressing."""
    web_context = None
    if should_search_web(user_message):
        try:
            q = build_chat_search_query(user_message, last_pattern)
            web_context = search_web(q, max_results=3)
            if web_context:
                log.info("Web context fetched for chat: %s...", user_message[:50])
        except Exception as e:
            log.warning("Web search skipped for chat: %s", e)

    system = build_chat_system_injection(
        trader_id=trader_id, username=username,
        trader_memory=trader_memory, web_context=web_context,
    )
    groq_messages = [{"role": "system", "content": system}]

    if messages and len(messages) > 0:
        for msg in messages:
            role    = msg.get("role", "")
            content = msg.get("content", "").strip()
            if role in ("user", "assistant") and content:
                groq_messages.append({"role": role, "content": content})
    else:
        groq_messages.append({"role": "user", "content": user_message})

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=groq_messages,
        temperature=0.35, max_tokens=450,
    )
    return response.choices[0].message.content.strip()


def generate_daily_insight(
    trader_id: str, recent_patterns: list,
    most_common: str, count: int, total: int,
) -> str:
    web_context = None
    if most_common and most_common != "No Mistake":
        try:
            web_context = search_web(
                f"{most_common} trading psychology morning mindset tip",
                max_results=2)
        except Exception:
            pass

    web_note = (
        "\n\nResearch tip (use ONE specific insight):\n" + web_context[:300]
    ) if web_context else ""

    prompt = (
        f"Trader showed '{most_common}' in {count}/{total} sessions this week.\n"
        f"Write a warm 2-sentence morning message.\n"
        f"Sentence 1: Acknowledge with empathy — do not shame.\n"
        f"Sentence 2: One specific actionable mindset tip for today.{web_note}"
    )
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content":
             "You are Plutus, a warm trading psychology coach. Exactly 2 sentences."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.45, max_tokens=150,
    )
    return response.choices[0].message.content.strip()