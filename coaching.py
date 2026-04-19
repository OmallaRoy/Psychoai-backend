# ================================================================
# FILE: coaching.py
# Groq-powered coaching with Plutus system prompt
#
# v1.0: Initial coaching with Qdrant RAG
# v1.1: Accepts full conversation history (stateless fix)
# v1.2: Dual RAG — Qdrant peer cases + Tavily web context
# v1.3: Mandatory 3-step prompt structure — Identification,
#        Evidence, Solution — forces model to always cite both
#        Qdrant peer cases and Tavily web research explicitly
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

# ── System prompt — Plutus identity and behavioral definitions ─
# This is loaded once and never changes per request.
# The mandatory 3-step structure is in build_coaching_prompt()
# not here — instructions are more effective when they appear
# directly alongside the data they reference.
PLUTUS_SYSTEM = """You are Plutus, an expert AI trading psychology coach with deep \
expertise in behavioral finance, cognitive psychology, and performance coaching for \
retail traders. You combine data from your proprietary trader database, current \
psychological research, and your deep understanding of trading behavior to deliver \
coaching that feels personal, evidence-based, and genuinely useful.

PRECISE BEHAVIORAL DEFINITIONS — memorise these and never confuse them:

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

CORE RULES — these apply to every single response:

1. MAINTAIN CONVERSATION CONTINUITY. If the conversation history shows that you \
asked a question and the trader is responding to it ("yes", "go ahead", "tell me \
more"), CONTINUE from that exact point. Never re-introduce yourself mid-conversation. \
Never say "Hello, I'm Plutus" unless this is genuinely the first message ever.

2. ONLY reference traders listed in the RETRIEVED HISTORICAL CASES section. \
Never invent trader IDs, names, or dates under any circumstances.

3. When feature signals are provided, name them specifically — do not speak \
in vague generalities about "your trading behavior."

4. When web research context is provided, you MUST reference specific facts, \
studies, or techniques from it. Do not summarise your own training knowledge \
and pretend it came from the web search.

5. Never give price predictions, entry/exit points, buy/sell signals, \
or financial recommendations of any kind.

6. Never confuse Revenge Trading with Held Loser Too Long — \
they are opposite behaviors and confusing them destroys credibility.

7. Be warm, direct, and specific. You are a coach, not a diagnostician. \
Traders need to feel understood and capable, not diagnosed and judged."""


def build_coaching_prompt(
    trader_id:       str,
    pred_label:      str,
    confidence:      float,
    feature_signals: list,
    retrieved_cases: list,
    trader_memory:   str,
    web_context:     str = None,
) -> str:
    """
    Build the one-shot coaching prompt sent to Groq after TCN detection.

    v1.3: Mandatory 3-step structure — Identification, Evidence, Solution.
    Instructions are explicit and non-optional so the model always cites
    both Qdrant peer cases (Evidence) and Tavily web research (Solution).

    The structure is defined HERE in the user prompt, not in the system
    prompt, because instructions are more reliably followed when they
    appear directly adjacent to the data they reference.
    """
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

    # Web context section — only included when Tavily returned results
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

    # Recurring pattern note for memory injection
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
        "behaviorally — not in abstract terms but in terms of what the trader "
        "actually did in this trade. Use the Feature Signals to specify WHICH "
        "behaviors triggered this detection.\n\n"

        "STEP 2 — EVIDENCE (1-2 sentences, mandatory):\n"
        "  Reference the Qdrant peer database. You MUST name at least one trader "
        "by their exact Trader ID and the exact date shown in the Retrieved "
        "Historical Cases. Describe what that trader experienced and how it "
        "mirrors the current situation. This is your 'scientific' evidence layer. "
        "Never invent or paraphrase trader IDs — use them exactly as written.\n\n"

        "STEP 3 — SOLUTION (2-3 sentences, mandatory):\n"
        "  You MUST reference at least TWO specific facts, techniques, or study "
        "findings from the Web Research Context. Do not skip this even if the "
        "facts seem general — cite them with the same specificity they appear in "
        "the web context. Then give ONE concrete, immediately actionable step "
        "the trader can take before their very next session. "
        "End with a single reflective question that invites the trader to examine "
        "what emotional state preceded this trade.\n\n"

        "ABSOLUTE RULES:\n"
        "- Total length: 5-7 sentences. No more, no less.\n"
        "- Never skip Step 2 or Step 3.\n"
        "- Never invent trader IDs, dates, or study names not present in the data above.\n"
        "- Never mention confidence scores, probabilities, or ML/AI terminology.\n"
        "- Never confuse Revenge Trading with Held Loser Too Long.\n"
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
    trader_id:    str,
    trader_memory: str,
    web_context:  str = None,
) -> str:
    """
    Build the system prompt for conversational /chat sessions.

    Injects trader memory and web context into the system message
    so Groq has full context before seeing the conversation history.
    """
    web_note = ""
    if web_context:
        web_note = (
            "\n\nLIVE WEB RESEARCH CONTEXT (fetched for this query):\n"
            "{}\n"
            "When relevant, reference specific facts or techniques from the above "
            "web context in your response. Make it clear the insight comes from "
            "current research, not generic advice."
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
        "Trader behavioral history (reference only if the trader brings it up, "
        "or if directly relevant to what they are asking):\n"
        "{memory}"
        "{web_note}\n\n"
        "CONVERSATION RULES:\n"
        "- This is a live chat session. Maintain continuity at all times.\n"
        "- If the conversation history shows you asked a question and the trader "
        "is answering it, CONTINUE — do not reset or re-introduce yourself.\n"
        "- Do not volunteer the trader's history unless they ask about it.\n"
        "- Be conversational, warm, and supportive.\n"
        "- Keep responses to 3-5 sentences in chat mode — shorter than coaching mode."
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
    """
    One-shot coaching after TCN detects a psychological mistake.
    Called asynchronously in the background after /analyze_trade.

    Flow:
    1. Build Tavily search query from detected pattern + feature signals
    2. Fetch web context (always triggered — pattern is always finance-relevant)
    3. Build mandatory 3-step prompt with Qdrant cases + Tavily context
    4. Call Groq — Llama 3.1 8B generates Plutus coaching response
    5. Return coaching string for Firestore save + FCM notification
    """
    # Step 1 — Tavily web search for detected pattern
    web_context = None
    try:
        search_query = build_coaching_search_query(pred_label, feature_signals)
        web_context  = search_web(search_query, max_results=3)
        if web_context:
            log.info("Web context fetched for pattern: %s", pred_label)
        else:
            log.info("Tavily returned no results for: %s — continuing without web", pred_label)
    except Exception as e:
        log.warning("Web search skipped for coaching (non-critical): %s", e)

    # Step 2 — Build mandatory 3-step coaching prompt
    user_msg = build_coaching_prompt(
        trader_id       = trader_id,
        pred_label      = pred_label,
        confidence      = confidence,
        feature_signals = feature_signals,
        retrieved_cases = retrieved_cases,
        trader_memory   = trader_memory,
        web_context     = web_context,
    )

    # Step 3 — Call Groq with Plutus system prompt
    response = groq_client.chat.completions.create(
        model    = "llama-3.1-8b-instant",
        messages = [
            {"role": "system", "content": PLUTUS_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        temperature = 0.30,   # low temperature = more consistent structure
        max_tokens  = 600,    # increased from 500 to accommodate 3-step structure
    )
    return response.choices[0].message.content.strip()


def generate_chat_response(
    trader_id:     str,
    user_message:  str,
    trader_memory: str,
    messages:      list = None,
    last_pattern:  str  = "",
) -> str:
    """
    Conversational endpoint — trader chats directly with Plutus.

    v1.1: Full conversation history passed to fix stateless problem.
           "yes" and "go ahead" now correctly continue the conversation.

    v1.2: Smart web search — Tavily called only when the message
           contains finance/psychology keywords. Filler replies
           ("yes", "ok", "go ahead") never trigger a search.

    v1.3: Web context injected into system message via
           build_chat_system_injection() for cleaner prompt structure.
    """
    # Smart web search decision — only search on substantive queries
    web_context = None
    if should_search_web(user_message):
        try:
            search_query = build_chat_search_query(user_message, last_pattern)
            web_context  = search_web(search_query, max_results=3)
            if web_context:
                log.info("Web context fetched for chat: %s...", user_message[:50])
        except Exception as e:
            log.warning("Web search skipped for chat (non-critical): %s", e)

    # Build system prompt with trader memory and web context injected
    system = build_chat_system_injection(
        trader_id     = trader_id,
        trader_memory = trader_memory,
        web_context   = web_context,
    )

    # Build Groq messages array with full conversation history
    groq_messages = [{"role": "system", "content": system}]

    if messages and len(messages) > 0:
        # Inject full conversation history from Android client.
        # Filters invalid roles and empty content to prevent API errors.
        for msg in messages:
            role    = msg.get("role", "")
            content = msg.get("content", "").strip()
            if role in ("user", "assistant") and content:
                groq_messages.append({"role": role, "content": content})
    else:
        # Fallback: no history provided — single user turn.
        # This handles first messages and older app versions.
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
    """
    Short 2-sentence motivational morning message.
    Called by the daily scheduler at 08:00 UTC.

    Optionally enriched with Tavily search for the most common pattern.
    Uses minimal search depth to keep daily job credit usage low.
    """
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
        "Sentence 1: Acknowledge their recurring pattern with empathy — "
        "do not shame them, make them feel understood.\n"
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