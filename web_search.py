# ================================================================
# FILE: web_search.py
# Tavily-powered web search for RAG enrichment in Plutus coaching
#
# Design principles:
#   1. Smart routing — only search when the query is genuinely
#      finance/psychology relevant. Never search for "yes" or
#      "tell me more" — that wastes credits and adds noise.
#   2. Credit-efficient — uses search_depth="basic" on free tier
#      (1,000 credits/month). Each search = 1 credit.
#   3. Graceful degradation — if Tavily fails or quota is exceeded,
#      coaching continues with Qdrant only. Never blocks the response.
#   4. Result filtering — only injects results with score > 0.4
#      to prevent low-quality pages polluting the prompt.
# ================================================================

import logging
from typing import Optional
from config import TAVILY_API_KEY

log = logging.getLogger("web_search")

# ── Lazy client init — only instantiate if key is present ─────
_client = None

def _get_client():
    global _client
    if _client is None:
        if not TAVILY_API_KEY:
            return None
        try:
            from tavily import TavilyClient
            _client = TavilyClient(api_key=TAVILY_API_KEY)
        except ImportError:
            log.warning("tavily-python not installed. Run: pip install tavily-python")
            return None
        except Exception as e:
            log.error("Tavily client init failed: %s", e)
            return None
    return _client


# ── Financial / psychology keyword set ─────────────────────────
# Only queries matching these terms trigger a Tavily search.
# Designed to be comprehensive for trading psychology while
# filtering out generic conversational noise.
FINANCE_KEYWORDS = {
    # Core psychological mistakes (model's 8 classes)
    "revenge", "fomo", "fear of missing", "oversized", "position size",
    "stop loss", "impulsive", "held loser", "cut winner",
    # Psychology and emotions
    "psychology", "emotion", "mindset", "discipline", "habit",
    "anxiety", "fear", "greed", "bias", "cognitive", "behavioral",
    "stress", "tilt", "overconfidence", "loss aversion", "panic",
    "self-control", "patience", "consistency", "routine",
    # Risk and money management
    "risk management", "drawdown", "capital", "leverage",
    "position sizing", "risk reward", "r:r", "expectancy",
    "account management", "money management",
    # Market concepts
    "market", "forex", "crypto", "stock", "futures", "options",
    "volatility", "trend", "breakout", "liquidity", "session",
    "london", "new york", "asian session", "overlap",
    # Trading practice
    "trading plan", "journal", "backtesting", "strategy",
    "technical analysis", "fundamental", "setup", "entry", "exit",
    "trade management", "scaling", "partial",
    # Improvement / coaching
    "how to improve", "how to stop", "how to control",
    "tips for", "advice on", "help with", "struggling with",
    "overcome", "fix", "change", "develop",
    # Research / explanations
    "what is", "explain", "why do", "research", "study",
    "data shows", "statistics", "proven", "evidence",
}

# Phrases that definitively indicate conversational filler —
# never search for these even if another keyword is present.
SKIP_PHRASES = {
    "yes", "no", "ok", "okay", "sure", "go ahead", "continue",
    "tell me more", "sounds good", "i see", "got it", "thanks",
    "thank you", "great", "perfect", "noted", "understood",
    "next", "please", "yes please", "alright",
}


def should_search_web(query: str) -> bool:
    """
    Determine whether this query warrants a Tavily web search.

    Returns True only when:
    1. The query is NOT a conversational filler phrase
    2. The query contains at least one finance/psychology keyword
    3. The query is long enough to be a meaningful question (>12 chars)

    This keeps credit usage low while ensuring searches happen
    precisely when they add value.
    """
    if not TAVILY_API_KEY:
        return False

    query_lower = query.lower().strip()

    # Reject one-word / short filler responses
    if len(query_lower) <= 12:
        return False

    # Reject known filler phrases
    for phrase in SKIP_PHRASES:
        if query_lower == phrase or query_lower.startswith(phrase + " "):
            return False

    # Accept if any finance keyword is present
    for kw in FINANCE_KEYWORDS:
        if kw in query_lower:
            return True

    return False


def search_web(query: str, max_results: int = 3) -> Optional[str]:
    """
    Execute a Tavily search and return formatted context string
    ready to inject into the Groq prompt.

    Args:
        query:       The search query (usually derived from
                     the detected pattern or user message)
        max_results: Number of results to fetch (default 3
                     to stay credit-efficient on free tier)

    Returns:
        Formatted string with source titles + snippets,
        or None if search fails or returns no useful results.
    """
    client = _get_client()
    if client is None:
        return None

    try:
        response = client.search(
            query        = query,
            search_depth = "basic",     # basic = 1 credit; advanced = 2
            max_results  = max_results,
            include_answer = False,     # saves tokens — we want raw results
            topic        = "general",
        )

        results = response.get("results", [])
        if not results:
            return None

        # Filter low-quality results
        good_results = [r for r in results if r.get("score", 0) >= 0.4]
        if not good_results:
            log.info("Tavily returned results but all below score threshold")
            return None

        # Format for prompt injection
        lines = ["RELEVANT WEB CONTEXT (use to enrich your response):"]
        for i, r in enumerate(good_results[:max_results], 1):
            title   = r.get("title", "Source")
            content = r.get("content", "").strip()
            url     = r.get("url", "")

            # Truncate content to keep prompt size manageable
            if len(content) > 400:
                content = content[:400] + "..."

            lines.append(f"\n[Source {i}] {title}")
            lines.append(f"{content}")

        log.info("Tavily search: %d results for query: %s...",
                 len(good_results), query[:60])
        return "\n".join(lines)

    except Exception as e:
        log.warning("Tavily search failed (coaching continues without web): %s", e)
        return None


def build_coaching_search_query(pattern: str, feature_signals: list) -> str:
    """
    Build an optimised Tavily search query for the one-shot
    coaching case (journal analysis → detected pattern).

    Instead of searching the raw pattern name, we build a
    descriptive query that returns practical psychology content.

    Examples:
        "Revenge Trading" → "revenge trading psychology how to stop recovery"
        "FOMO"           → "FOMO trading psychology strategies discipline"
        "No Stop Loss"   → "trading without stop loss risk management dangers"
    """
    pattern_queries = {
        "Revenge Trading":     "revenge trading psychology how to stop emotional recovery",
        "FOMO":                "FOMO trading fear of missing out psychology strategies",
        "Held Loser Too Long": "holding losing trades too long loss aversion psychology",
        "Cut Winner Early":    "cutting winning trades early fear profit psychology",
        "Oversized":           "position sizing psychology risk discipline trading",
        "Impulsive Entry":     "impulsive trading emotional entry discipline mindset",
        "No Stop Loss":        "trading without stop loss risk management psychology",
        "No Mistake":          "disciplined trading psychology consistent performance",
    }

    base_query = pattern_queries.get(
        pattern,
        f"{pattern} trading psychology strategies improvement"
    )

    # Enrich query with signal context if relevant
    enrichments = []
    for signal in feature_signals[:2]:
        if "stop loss" in signal.lower():
            enrichments.append("risk management")
        if "plan" in signal.lower():
            enrichments.append("trading plan discipline")
        if "position" in signal.lower() or "lot" in signal.lower():
            enrichments.append("position sizing")

    if enrichments:
        base_query += " " + " ".join(set(enrichments))

    return base_query


def build_chat_search_query(user_message: str, trader_pattern: str = "") -> str:
    """
    Build a search query from a conversational user message.
    Keeps the query focused on finance/psychology.
    """
    query = user_message.strip()

    # Remove filler words that would confuse the search
    for filler in ["can you", "could you", "please", "i want to", "help me"]:
        query = query.lower().replace(filler, "").strip()

    # Append pattern context if available and relevant
    if trader_pattern and trader_pattern != "No Mistake":
        if trader_pattern.lower() not in query.lower():
            query = f"{query} {trader_pattern} trading"

    # Ensure it's trading-focused
    if "trading" not in query.lower() and "psychology" not in query.lower():
        query = f"trading psychology {query}"

    return query[:200]    # Tavily max query length