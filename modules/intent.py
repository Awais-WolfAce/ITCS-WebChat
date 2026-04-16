"""Lightweight intent classifier to route chitchat vs. knowledge queries."""

from __future__ import annotations

import re

# Patterns that indicate chitchat / small-talk (matched against lowercased,
# stripped English text coming out of the translation layer).
_CHITCHAT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        # greetings
        r"^(hi|hello|hey|howdy|hiya|yo|greetings|salaam|salam|assalam[ou]?\s*alaikum)\b",
        r"^good\s*(morning|afternoon|evening|night|day)\b",
        # farewells
        r"^(bye|goodbye|see\s*you|take\s*care|good\s*night|khuda\s*hafiz|allah\s*hafiz)\b",
        # gratitude
        r"^(thanks?|thank\s*you|thankyou|shukriya|shukria|much\s*appreciated)\b",
        # how are you
        r"^how\s*(are|r)\s*(you|u|ya)\b",
        r"^(how\s*do\s*you\s*do|what\'?s?\s*up|sup|how\s*is\s*it\s*going)\b",
        r"^(i\'?m?\s*(fine|good|great|okay|ok)|doing\s*(well|good|fine))\b",
        # identity
        r"^(who|what)\s*(are|r)\s*(you|u)\b",
        r"^what\s*is\s*your\s*name\b",
        # pleasantries
        r"^(nice|good)\s*to\s*(meet|talk|chat)\b",
        r"^(welcome|you\'?re?\s*welcome|no\s*problem|no\s*worries)\b",
        # filler
        r"^(ok|okay|alright|sure|yes|no|yep|nope|hmm+|haha|lol|cool)\s*[.!?]*$",
    ]
]


def is_chitchat(text: str) -> bool:
    """Return True if *text* (English) looks like small-talk, not a knowledge query."""
    cleaned = text.strip()
    if not cleaned:
        return True
    return any(p.search(cleaned) for p in _CHITCHAT_PATTERNS)
