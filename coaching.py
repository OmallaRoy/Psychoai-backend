# ================================================================
# FILE: coaching.py
# Groq-powered coaching with Plutus system prompt
#
# v1.0: Initial coaching with Qdrant RAG
# v1.1: Accepts full conversation history (stateless fix)
# v1.2: Dual RAG — Qdrant peer cases + Tavily web context
# v1.3: Mandatory 3-step prompt structure
# v1.4: Topic boundary enforcement — Plutus redirects all
#        off-topic questions (food, health, general knowledge)
#        back to trading psychology. Fixes screenshot issue where
#        Plutus answered "what is sex" and nutrition questions.
# ================================================================

import logging
from groq import Groq
from config import GROQ_API_KEY
from web_search import (
    should_search_web,
    search_web,
    build_coaching_search_query,
    build_chat_search_query,
)

log = logging.getLogger("coaching")
groq_client = Groq(api_key=GROQ_API_KEY)

PLUTUS_SYSTEM = """You are Plutus, an expert AI trading psychology coach. \
Your ONLY area of expertise is trading psychology, behavioral finance, \
emotional discipline in trading, and performance coaching for retail traders. \
You have no other expertise.

CRITICAL TOPIC BOUNDARY — READ THIS FIRST:
You ONLY respond to questions about:
- Trading psychology and emotional discipline
- Specific trading mistakes (revenge trading, FOMO, oversized positions, etc.)
- Risk management mindset and behavioral patterns
- Performance coaching for traders
- Financial markets, trading strategies, and market psychology
- The trader's own journal data and coaching sessions

If ANYONE asks you about ANYTHING outside this scope — including but not limited to:
food, nutrition, health, medicine, biology, relationships, sex, sports, politics,
general science, entertainment, coding, or any other non-trading topic —
you MUST respond with exactly this type of redirection and nothing more:
"I'm Plutus, your trading psychology coach. I can only help with trading \
psychology, behavioral patterns, and emotional discipline in trading. \
Is there something about your trading mindset I can help you with?"

Never answer off-topic questions even briefly. Never say "however, here's the answer." \
Redirect immediately and offer to help with trading psychology instead.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRECISE BEHAVIORAL DEFINITIONS — for on-topic responses:

- Revenge Trading: Entering a NEW trade immediately after a loss, usually with a \
LARGER position size, driven by the emotional urge to recover losses. \
Key: it is a NEW position opened after losing. Not the same as holding.

- Held Loser Too Long: STAYING IN an existing LOSING position for too long, refusing \
to accept the loss, hoping price will reverse. \
Key: staying in an existing position. Not opening a new one.

- Cut Winner Early: EXITING a WINNING trade too soon out of fear of losing the \
unrealized gain. Key: closing a profitable trade before the target was reached.

- FOMO (Fear of Missing Out): Entering a trade AFTER the price has already moved \
significantly, driven by fear of missing the opportunity. \
Key: late entry after the move already happened.

- Impulsive Entry: Entering a trade WITHOUT any plan or technical setup, acting \
purely on emotion in the moment. Key: no pre-trade analysis, no setup whatsoever.

- No Stop Loss: Trading WITHOUT setting a predefined stop loss level, leaving \
capital fully exposed to unlimited downside.

- Oversized Position: Risking MORE capital than the trader's own rules allow on \
a single trade.

- No Mistake: The trade followed the plan completely and correctly. \
This is disciplined, professional trading.

CORE RULES for on-topic responses:

1. MAINTAIN CONVERSATION CONTINUITY. If the conversation history shows that you \
asked a question and the trader is responding to it ("yes", "go ahead", "tell me \
more"), CONTINUE from that exact point. Never re-introduce yourself mid-conversation. \
Never say "Hello, I'm Plutus" unless this is genuinely the first message ever.

2. ONLY reference traders listed in the RETRIEVED HISTORICAL CASES section. \
Never invent trader IDs, names, or dates under any circumstances.

3. When feature signals are provided, name them specifically.

4. When web research context is provided, you MUST reference specific facts, \
studies, or techniques from it.

5. Never give price predictions, entry/exit points, buy/sell signals, \
or financial recommendations of any kind.

6. Never confuse Revenge Trading with Held Loser Too Long.

7. Be warm, direct, and specific. You are a coach, not a diagnostician."""


def build_coaching_prompt(
    trader_id:       str,
    pred_label:      str,
    confidence:      float,
    feature_signals: list,
    retrieved_cases: list,
    trader_memory:   str,
    web_context:     str = None,
) -> str:
    anomaly_ctx = "Detected pattern: {} (confidence: {:.0f}%)".format(
        pred_label, confidence * 100
    )

    rag_ctx = "\n".join(
        "  - Rank {rank}: Trader {tid} on {date} showed pattern '{label}'".format(
            rank  = r["rank"],
            tid   = r["trader_id"],
            date  = r["last_date"],
            label = r["true_label"],
        )
        for r in retrieved_cases
    ) if retrieved_cases else "  - No peer cases retrieved for this session."

    signals_text = "\n".join(
        "  - {}".format(s) for s in feature_signals
    ) if feature_signals else "  - No specific signals available."

    web_section = ""
    if web_context:
        web_section = (
            "\n\nWEB RESEARCH CONTEXT (from live search — published sources):\n"
            "{}\n"
            "You MUST reference at least TWO specific facts, techniques, or study "
            "findings from the above web context in your SOLUTION step. "
            "Do not substitute your own training knowledge for what is written above."
        ).format(web_context)
    else:
        web_section = (
            "\n\nWEB RESEARCH CONTEXT: Not available for this session. "
            "Draw on established behavioral finance research for the Solution step."
        )

    history_note = (
        "\n\nIMPORTANT: The trader history above shows recurring patterns. "
        "Reference the specific dates and patterns when relevant to show "
        "this is an ongoing behavioral theme, not an isolated incident."
        if "No previous" not in trader_memory and "First coaching" not in trader_memory
        else ""
    )

    return (
        "TRADER ID: {trader_id}\n\n"
        "TRADER BEHAVIORAL HISTORY (from Firestore memory):\n"
        "{memory}{history_note}\n\n"
        "CURRENT SESSION — TCN DETECTION:\n"
        "{anomaly}\n\n"
        "FEATURE SIGNALS THAT TRIGGERED THIS DETECTION:\n"
        "{signals}\n\n"
        "RETRIEVED HISTORICAL CASES FROM PEER TRADER DATABASE (Qdrant):\n"
        "{rag_ctx}"
        "{web_section}\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "COACHING RESPONSE — YOU MUST FOLLOW THIS EXACT STRUCTURE:\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "STEP 1 — IDENTIFICATION (1 sentence, mandatory):\n"
        "  Name the detected pattern precisely and explain what it means "
        "behaviorally using the Feature Signals to specify WHICH behaviors "
        "triggered this detection.\n\n"
        "STEP 2 — EVIDENCE (1-2 sentences, mandatory):\n"
        "  Reference the Qdrant peer database. You MUST name at least one trader "
        "by their exact Trader ID and the exact date shown. Describe what that "
        "trader experienced and how it mirrors the current situation. "
        "Never invent or paraphrase trader IDs.\n\n"
        "STEP 3 — SOLUTION (2-3 sentences, mandatory):\n"
        "  You MUST reference at least TWO specific facts, techniques, or study "
        "findings from the Web Research Context. Give ONE concrete, immediately "
        "actionable step the trader can take before their very next session. "
        "End with a single reflective question about the emotional state that "
        "preceded this trade.\n\n"
        "ABSOLUTE RULES:\n"
        "- Total length: 5-7 sentences. No more, no less.\n"
        "- Never skip Step 2 or Step 3.\n"
        "- Never invent trader IDs, dates, or study names not present above.\n"
        "- Never mention confidence scores, probabilities, or ML/AI terminology.\n"
        "- Tone: warm, direct, expert — a coach, not a diagnostician."
    ).format(
        trader_id    = trader_id,
        memory       = trader_memory,
        history_note = history_note,
        anomaly      = anomaly_ctx,
        signals      = signals_text,
        rag_ctx      = rag_ctx,
        web_section  = web_section,
    )


def build_chat_system_injection(
    trader_id:     str,
    trader_memory: str,
    web_context:   str = None,
) -> str:
    web_note = ""
    if web_context:
        web_note = (
            "\n\nLIVE WEB RESEARCH CONTEXT (fetched for this query):\n"
            "{}\n"
            "When relevant, reference specific facts or techniques from the above "
            "web context in your response."
        ).format(web_context)

    return (
        PLUTUS_SYSTEM +
        "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "ACTIVE SESSION CONTEXT\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Trader identifier: {tid}\n"
        "IMPORTANT: If the trader ID resembles a real name, address them warmly "
        "by that name. If it is a random string (Firebase UID), do not use it — "
        "just say 'hey' or address them without a name.\n\n"
        "Trader behavioral history:\n"
        "{memory}"
        "{web_note}\n\n"
        "CONVERSATION RULES:\n"
        "- Maintain continuity at all times.\n"
        "- If the trader is answering a question you asked, CONTINUE — do not reset.\n"
        "- Do not volunteer trading history unless they ask.\n"
        "- Be conversational, warm, and supportive.\n"
        "- Keep responses to 3-5 sentences in chat mode.\n"
        "- REDIRECT any off-topic question immediately per the topic boundary above."
    ).format(
        tid      = trader_id,
        memory   = trader_memory,
        web_note = web_note,
    )


def generate_coaching(
    trader_id:       str,
    pred_label:      str,
    confidence:      float,
    feature_signals: list,
    retrieved_cases: list,
    trader_memory:   str,
) -> str:
    web_context = None
    try:
        search_query = build_coaching_search_query(pred_label, feature_signals)
        web_context  = search_web(search_query, max_results=3)
        if web_context:
            log.info("Web context fetched for pattern: %s", pred_label)
    except Exception as e:
        log.warning("Web search skipped for coaching (non-critical): %s", e)

    user_msg = build_coaching_prompt(
        trader_id       = trader_id,
        pred_label      = pred_label,
        confidence      = confidence,
        feature_signals = feature_signals,
        retrieved_cases = retrieved_cases,
        trader_memory   = trader_memory,
        web_context     = web_context,
    )

    response = groq_client.chat.completions.create(
        model    = "llama-3.1-8b-instant",
        messages = [
            {"role": "system", "content": PLUTUS_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        temperature = 0.30,
        max_tokens  = 600,
    )
    return response.choices[0].message.content.strip()


def generate_chat_response(
    trader_id:     str,
    user_message:  str,
    trader_memory: str,
    messages:      list = None,
    last_pattern:  str  = "",
) -> str:
    web_context = None
    if should_search_web(user_message):
        try:
            search_query = build_chat_search_query(user_message, last_pattern)
            web_context  = search_web(search_query, max_results=3)
            if web_context:
                log.info("Web context fetched for chat: %s...", user_message[:50])
        except Exception as e:
            log.warning("Web search skipped for chat (non-critical): %s", e)

    system = build_chat_system_injection(
        trader_id     = trader_id,
        trader_memory = trader_memory,
        web_context   = web_context,
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
        model    = "llama-3.1-8b-instant",
        messages = groq_messages,
        temperature = 0.35,
        max_tokens  = 450,
    )
    return response.choices[0].message.content.strip()


def generate_daily_insight(
    trader_id:       str,
    recent_patterns: list,
    most_common:     str,
    count:           int,
    total:           int,
) -> str:
    web_context = None
    if most_common and most_common != "No Mistake":
        try:
            query = (
                "{} trading psychology morning routine mindset improvement tip"
                .format(most_common)
            )
            web_context = search_web(query, max_results=2)
        except Exception:
            pass

    web_note = ""
    if web_context:
        web_note = (
            "\n\nCurrent research tip (incorporate ONE specific insight "
            "from this into your message):\n" + web_context[:300]
        )

    prompt = (
        "Trader {tid} has shown the pattern '{pattern}' in {count}/{total} "
        "trading sessions this week.\n\n"
        "Write a warm, 2-sentence motivational morning message.\n"
        "Sentence 1: Acknowledge their recurring pattern with empathy.\n"
        "Sentence 2: Give one specific, actionable mindset tip for today's session.\n"
        "Tone: encouraging, human, expert — not clinical or robotic.{web_note}"
    ).format(
        tid      = trader_id,
        pattern  = most_common,
        count    = count,
        total    = total,
        web_note = web_note,
    )

    response = groq_client.chat.completions.create(
        model    = "llama-3.1-8b-instant",
        messages = [
            {
                "role":    "system",
                "content": "You are Plutus, a warm and supportive trading "
                           "psychology coach. Keep responses to exactly 2 sentences."
            },
            {"role": "user", "content": prompt},
        ],
        temperature = 0.45,
        max_tokens  = 150,
    )
    return response.choices[0].message.content.strip()