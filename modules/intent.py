"""Intent classifier for the ITCS chat agent.

Classifies the latest user message into one of the supported :class:`Intent`
values and groups intents into broad categories so the chat layer can route
them appropriately (chitchat vs. knowledge retrieval vs. meta/control, etc.).
"""

from __future__ import annotations

import re
from enum import Enum


class Intent(str, Enum):
    """All intents recognized by the agent."""

    # ---- Social / chitchat ------------------------------------------------
    GREETING = "greeting"
    GOODBYE = "goodbye"
    APPRECIATION = "appreciation"
    APOLOGY = "apology"
    COMPLIMENT = "compliment"
    COMPLAINT = "complaint"
    FEEDBACK = "feedback"
    BOT_IDENTITY = "bot_identity"
    BOT_CAPABILITY = "bot_capability"

    # ---- Meta / conversational control -----------------------------------
    CLARIFY_USER_REQUEST = "clarify_user_request"
    ASK_FOR_EXAMPLES = "ask_for_examples"
    CONFIRM_UNDERSTANDING = "confirm_understanding"
    REPEAT = "repeat"
    SUMMARIZE = "summarize"
    DISAMBIGUATE_ENTITY = "disambiguate_entity"
    CONTRADICTION_RESOLUTION = "contradiction_resolution"
    SORT = "sort"
    CONFIRM_ENTITIES = "confirm_entities"
    SESSION_RESET = "session_reset"

    # ---- Knowledge / information-seeking ---------------------------------
    ASK_FACTUAL = "ask_factual_question"
    ASK_DEFINITION = "ask_definition"
    ASK_PROCEDURAL = "ask_procedural_question"
    ASK_COMPARISON = "ask_comparison"
    ASK_RECOMMENDATION = "ask_recommendation"
    ASK_TROUBLESHOOTING = "ask_troubleshooting"

    # ---- Routing / escalation --------------------------------------------
    PROVIDE_CONTACT_INFO = "provide_contact_info"
    HUMAN_HANDOFF = "human_handoff"
    OUT_OF_SCOPE = "out_of_scope"
    FALLBACK = "fallback"


# Broad categories used by the chat layer to pick a handler.
CHITCHAT_INTENTS: frozenset[Intent] = frozenset(
    {
        Intent.GREETING,
        Intent.GOODBYE,
        Intent.APPRECIATION,
        Intent.APOLOGY,
        Intent.COMPLIMENT,
        Intent.BOT_IDENTITY,
        Intent.BOT_CAPABILITY,
    }
)

META_INTENTS: frozenset[Intent] = frozenset(
    {
        Intent.CLARIFY_USER_REQUEST,
        Intent.ASK_FOR_EXAMPLES,
        Intent.CONFIRM_UNDERSTANDING,
        Intent.REPEAT,
        Intent.SUMMARIZE,
        Intent.DISAMBIGUATE_ENTITY,
        Intent.CONTRADICTION_RESOLUTION,
        Intent.SORT,
        Intent.CONFIRM_ENTITIES,
        Intent.SESSION_RESET,
    }
)

KNOWLEDGE_INTENTS: frozenset[Intent] = frozenset(
    {
        Intent.ASK_FACTUAL,
        Intent.ASK_DEFINITION,
        Intent.ASK_PROCEDURAL,
        Intent.ASK_COMPARISON,
        Intent.ASK_RECOMMENDATION,
        Intent.ASK_TROUBLESHOOTING,
    }
)

ESCALATION_INTENTS: frozenset[Intent] = frozenset(
    {
        Intent.PROVIDE_CONTACT_INFO,
        Intent.HUMAN_HANDOFF,
        Intent.COMPLAINT,
        Intent.FEEDBACK,
    }
)


def _p(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


# Order matters: patterns are evaluated top-to-bottom and the first match wins.
# More specific intents should appear before more generic ones.
_INTENT_PATTERNS: list[tuple[Intent, re.Pattern[str]]] = [
    # ---- Session reset (check early; resets override other cues) ---------
    # Must refer to the chat/conversation/context itself, not user data like
    # passwords or settings.
    (Intent.SESSION_RESET, _p(
        r"\b(reset|restart|clear|wipe)\s*(the\s*|this\s*|our\s*)?(chat|conversation|session|history|context|memory)\b"
        r"|\bnew\s*(chat|session|conversation)\b"
        r"|\bstart\s*over\b"
        r"|\bforget\s*(everything|all|this|our\s*conversation|what\s*(i|we)\s*said)\b"
    )),

    # ---- Human handoff ---------------------------------------------------
    (Intent.HUMAN_HANDOFF, _p(r"\b(human|agent|representative|real\s*person|live\s*(agent|support|chat)|customer\s*(service|support)|talk\s*to\s*(someone|a\s*person)|speak\s*to\s*(someone|a\s*person|an?\s*agent))\b")),

    # ---- Provide contact info -------------------------------------------
    (Intent.PROVIDE_CONTACT_INFO, _p(r"\b(contact(\s*info|\s*details|\s*us)?|phone(\s*number)?|email(\s*address)?|address|office\s*location|reach\s*(you|itcs)|get\s*in\s*touch|support\s*(email|number))\b")),

    # ---- Bot identity ----------------------------------------------------
    (Intent.BOT_IDENTITY, _p(r"^(who|what)\s*(are|r)\s*(you|u)\b|\bwhat\s*is\s*your\s*name\b|\bare\s*you\s*(a\s*)?(bot|human|ai|robot|chatbot)\b|\bintroduce\s*yourself\b")),

    # ---- Bot capability --------------------------------------------------
    (Intent.BOT_CAPABILITY, _p(r"\bwhat\s*can\s*you\s*do\b|\bwhat\s*(do|are)\s*you\s*(able\s*to\s*do|capable\s*of)\b|\bhow\s*(can|could)\s*you\s*help\b|\bwhat\s*are\s*your\s*(features|capabilities|skills|functions)\b|\bhelp\s*me\s*with\s*what\b")),

    # ---- Repeat ----------------------------------------------------------
    (Intent.REPEAT, _p(r"\b(say\s*(that\s*)?again|repeat(\s*that|\s*please|\s*it)?|come\s*again|pardon(\s*me)?|what\s*did\s*you\s*say)\b")),

    # ---- Summarize -------------------------------------------------------
    (Intent.SUMMARIZE, _p(r"\b(summar(ize|ise|y|ise\s*it)|tl;dr|in\s*short|short\s*version|recap|brief(\s*summary|\s*version)?|give\s*me\s*(a|the)\s*summary)\b")),

    # ---- Ask for examples ------------------------------------------------
    (Intent.ASK_FOR_EXAMPLES, _p(r"\b(give|show|provide)\s*(me\s*)?(an?\s*)?(example|examples|sample|samples|use\s*cases?)\b|\bfor\s*example\??$|\blike\s*what\??$|\bsuch\s*as\s*what\??$")),

    # ---- Confirm understanding -------------------------------------------
    (Intent.CONFIRM_UNDERSTANDING, _p(r"\b(do\s*you\s*understand|did\s*you\s*(get|understand)\s*(that|me)|does\s*that\s*make\s*sense|are\s*we\s*(clear|on\s*the\s*same\s*page)|got\s*it\??)\b")),

    # ---- Confirm entities ------------------------------------------------
    (Intent.CONFIRM_ENTITIES, _p(r"\b(is\s*that\s*(correct|right)|did\s*i\s*(say|get)\s*that\s*right|is\s*this\s*what\s*you\s*mean(t)?|are\s*these\s*(details|entities)\s*(correct|right)|please\s*confirm)\b")),

    # ---- Disambiguate entity --------------------------------------------
    (Intent.DISAMBIGUATE_ENTITY, _p(r"\bwhich\s*(one|of\s*(them|these|those))\b|\bdo\s*you\s*mean\b|\b(do\s*you\s*mean|are\s*you\s*referring\s*to)\b|\bwhich\s*(product|service|option|plan)\b")),

    # ---- Contradiction resolution ---------------------------------------
    (Intent.CONTRADICTION_RESOLUTION, _p(r"\b(that\s*contradicts|but\s*(earlier|before|previously)\s*you\s*said|you\s*(just\s*)?said\s*.*\s*(different|opposite|otherwise)|that\'?s?\s*(inconsistent|contradictory)|this\s*doesn\'?t\s*match\s*what\s*you\s*said)\b")),

    # ---- Clarify user request -------------------------------------------
    (Intent.CLARIFY_USER_REQUEST, _p(r"\b(what\s*do\s*you\s*mean|can\s*you\s*clarify|please\s*clarify|i\s*don\'?t\s*(get|understand)|not\s*sure\s*what\s*you\s*mean|explain\s*(that|this)\s*(more|better|again)|be\s*more\s*specific)\b")),

    # ---- Sort ------------------------------------------------------------
    (Intent.SORT, _p(r"\b(sort|order|rank|arrange|organize|organise)\s*(by|them|these|those|it|results|list)?\b|\b(in\s*(ascending|descending|alphabetical|chronological)\s*order)\b|\b(cheapest|most\s*expensive|highest|lowest|newest|oldest)\s*first\b")),

    # ---- Ask comparison --------------------------------------------------
    (Intent.ASK_COMPARISON, _p(r"\b(compare|comparison|vs\.?|versus|difference\s*between|differences\s*between|which\s*is\s*(better|best)|pros\s*and\s*cons|better\s*than)\b")),

    # ---- Ask recommendation ---------------------------------------------
    (Intent.ASK_RECOMMENDATION, _p(r"\b(recommend|suggest|suggestion|what\s*(do|would)\s*you\s*(recommend|suggest)|which\s*(should|would)\s*i\s*(use|choose|pick)|best\s*(option|choice|fit|one)\s*for)\b")),

    # ---- Ask troubleshooting --------------------------------------------
    (Intent.ASK_TROUBLESHOOTING, _p(r"\b(not\s*working|doesn\'?t\s*work|isn\'?t\s*working|broken|error|failed|failing|issue|problem|bug|crash(ed|ing)?|stuck|can\'?t\s*(connect|login|log\s*in|access|open|start)|fix|troubleshoot|debug|help\s*me\s*(fix|solve))\b")),

    # ---- Ask definition --------------------------------------------------
    (Intent.ASK_DEFINITION, _p(r"^(what\s*(is|are|does)\s*(a|an|the)?\s*)|\bdefine\b|\bdefinition\s*of\b|\bmeaning\s*of\b|\bwhat\s*does\s*.*\s*mean\b")),

    # ---- Ask procedural question ----------------------------------------
    (Intent.ASK_PROCEDURAL, _p(r"^how\s*(do|can|to|should|would)\s*(i|you|we|one)?\b|\bsteps?\s*to\b|\bguide\s*(me\s*)?(on|to|through)\b|\bwalk\s*me\s*through\b|\btutorial\b|\binstructions?\s*(for|on)\b")),

    # ---- Ask factual question -------------------------------------------
    (Intent.ASK_FACTUAL, _p(r"^(who|what|when|where|why|which|how\s*(many|much|long|often|old))\b|\bis\s*it\s*true\b|\btell\s*me\s*(about|the)\b")),

    # ---- Compliment ------------------------------------------------------
    (Intent.COMPLIMENT, _p(
        r"\byou\s*(\'?re|are)\s*(so\s*|really\s*|very\s*)?(great|awesome|amazing|helpful|smart|the\s*best|good|wonderful|brilliant|fantastic)\b"
        r"|\b(great|good|nice|excellent|awesome|amazing)\s*(job|work|bot|assistant|answer|response)\b"
        r"|\bwell\s*done\b|\bimpressive\b|\blove\s*(you|this\s*bot)\b"
    )),

    # ---- Complaint -------------------------------------------------------
    (Intent.COMPLAINT, _p(r"\b(useless|terrible|awful|horrible|stupid|dumb|bad\s*(bot|answer|response|service)|(this|you)\s*(sucks?|suck)|waste\s*of\s*time|unhelpful|frustrat(ed|ing)|annoy(ed|ing)|disappointed|hate\s*(this|you))\b")),

    # ---- Feedback --------------------------------------------------------
    (Intent.FEEDBACK, _p(r"\b(feedback|suggestion\s*for\s*you|(i\s*have|here\'?s)\s*(some\s*)?feedback|you\s*should\s*(improve|add|fix)|it\s*would\s*be\s*(better|nice|great)\s*if|please\s*(improve|consider))\b")),

    # ---- Apology ---------------------------------------------------------
    (Intent.APOLOGY, _p(r"^(sorry|my\s*(bad|apologies)|i\s*apologize|apologies|i\'?m\s*sorry|excuse\s*me)\b")),

    # ---- Appreciation / thanks ------------------------------------------
    (Intent.APPRECIATION, _p(r"^(thanks?|thank\s*you|thankyou|thx|ty|shukriya|shukria|much\s*appreciated|i\s*appreciate|appreciate\s*(it|that|you))\b")),

    # ---- Goodbye ---------------------------------------------------------
    (Intent.GOODBYE, _p(r"^(bye|goodbye|see\s*(you|ya)|take\s*care|good\s*night|cya|farewell|catch\s*you\s*later|talk\s*to\s*you\s*later|ttyl|khuda\s*hafiz|allah\s*hafiz)\b")),

    # ---- Greeting --------------------------------------------------------
    (Intent.GREETING, _p(r"^(hi|hello|hey|howdy|hiya|yo|greetings|salaam|salam|assalam[ou]?\s*alaikum)\b|^good\s*(morning|afternoon|evening|day)\b|^how\s*(are|r)\s*(you|u|ya)\b|^(what\'?s?\s*up|sup|how\s*is\s*it\s*going|how\'?s?\s*it\s*going)\b")),
]


# Topics the bot is allowed to answer about. If the user asks about something
# obviously unrelated (weather, sports, celebrity gossip, etc.) we mark it as
# out-of-scope so the chat layer can respond appropriately.
_OUT_OF_SCOPE_PATTERNS: list[re.Pattern[str]] = [
    _p(r"\b(weather|forecast|temperature\s*(today|tomorrow|outside))\b"),
    _p(r"\b(stock\s*price|crypto\s*price|bitcoin\s*price)\b"),
    _p(r"\b(score|match\s*result|who\s*won\s*the\s*(game|match))\b"),
    _p(r"\b(joke|tell\s*me\s*a\s*joke|sing\s*(a\s*)?song|write\s*a\s*poem)\b"),
    _p(r"\b(horoscope|zodiac|lottery\s*numbers)\b"),
]


def classify_intent(text: str) -> Intent:
    """Classify *text* (English, lowercased internally) into an :class:`Intent`.

    Returns :attr:`Intent.FALLBACK` for empty input and :attr:`Intent.OUT_OF_SCOPE`
    for messages that clearly fall outside the ITCS support domain.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return Intent.FALLBACK

    # Topic-based out-of-scope wins over syntactic patterns: "what is the
    # weather" parses as a definition question but is not in-domain.
    for pattern in _OUT_OF_SCOPE_PATTERNS:
        if pattern.search(cleaned):
            return Intent.OUT_OF_SCOPE

    for intent, pattern in _INTENT_PATTERNS:
        if pattern.search(cleaned):
            return intent

    # Nothing matched but the user asked something – treat as a knowledge
    # query so it hits the search index; the LLM will fall back gracefully
    # if the index has no relevant data.
    if cleaned.endswith("?") or re.search(r"\b(tell|explain|describe|list)\b", cleaned, re.IGNORECASE):
        return Intent.ASK_FACTUAL

    return Intent.FALLBACK


def is_chitchat(text: str) -> bool:
    """Backwards-compatible helper: True for social small-talk intents."""
    return classify_intent(text) in CHITCHAT_INTENTS
