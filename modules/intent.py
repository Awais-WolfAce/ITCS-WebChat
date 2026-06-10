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
    AFFIRMATION = "affirmation"
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
    ASK_PRICE = "ask_price"     

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
        Intent.ASK_PRICE, 
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


# ---------------------------------------------------------------------------
# Reusable phrase groups (English, broken English, SMS-shorthand, Roman Urdu).
# Keep these as plain alternation strings so we can compose larger patterns.
# ---------------------------------------------------------------------------

# Greetings: English + SMS shortenings + Roman Urdu salutations.
_GREETING_WORDS = (
    r"hi+|hello+|hey+|heyy+|hay|hii+|helo|hlo|hallo|howdy|hiya|yo+|greetings"
    r"|salaam|salam|aslam|asalam|asalamualaikum|assalam[ou]?\s*alaikum|aoa|slm"
)

# Roman Urdu "how are you" style openers.
_GREETING_HOW_ARE_YOU = (
    r"(aap|tum|tu)\s*(kese|kaise|kaisi|kesa|kaisa)\s*(ho|hain|hai)"
    r"|k(i|y)a\s*(hal|haal)(\s*(hai|hain|chal\s*raha))?"
    r"|kese\s*ho|kesa\s*hai|theek\s*ho|all\s*good"
)

# Goodbyes / chat-end phrases.
# Note: order matters inside the alternation - put multi-word forms like
# "good bye" / "bye bye" BEFORE the bare "bye+" token so the longer match
# wins and we don't accidentally classify "good bye" as a partial match
# that falls through to FALLBACK.
_GOODBYE_WORDS = (
    r"good[\s\-]*bye|bye[\s\-]*bye|bye+|see\s*(you|ya)|take\s*care"
    r"|good\s*night|gn|cya"
    r"|farewell|catch\s*you\s*later|talk\s*to\s*you\s*later|ttyl"
    r"|khuda\s*hafiz|allah\s*hafiz|alvida"
)

# Thanks: English + abbreviations + Roman Urdu / Arabic-borrowed.
_THANKS_WORDS = (
    r"thanks?|thank\s*you|thankyou|thnx|thnks|thanx|thx|ty|tysm"
    r"|shukriya|shukria|shukrya|shukariya"
    r"|much\s*appreciated|i\s*appreciate|appreciate\s*(it|that|you)"
    r"|meherbani|meharbani|bohat\s*shukriya|bht\s*shukriya"
    r"|jazak\s*allah(\s*khair(an)?)?|jazakallah(\s*khair(an)?)?|jzk"
)

# Apologies (English + Roman Urdu).
_APOLOGY_WORDS = (
    r"sorry|my\s*(bad|apologies)|(i\s*)?apologize|apologies|i\'?m\s*sorry"
    r"|excuse\s*me|pardon\s*me"
    r"|maaf\s*(karna|kijiye|kar\s*do|kar\s*dein|do)"
    r"|maazrat|maafi|mafi|meafi(\s*maang(na|ta|ti)?)?"
    r"|galti\s*ho\s*(gai|gayi|gaya)"
)

# Negative sentiment / complaint markers (English + Roman Urdu).
_COMPLAINT_WORDS = (
    r"useless|terrible|awful|horrible|stupid|dumb"
    r"|bad\s*(bot|answer|response|service|experience)"
    r"|(this|you)\s*(sucks?|suck)|waste\s*of\s*time|unhelpful"
    r"|frustrat(ed|ing)|annoy(ed|ing)|disappointed|hate\s*(this|you)"
    r"|bakwas|ghatiya|ghalat|bura|worst|fed\s*up|gussa|pareshan"
    r"|rude|unprofessional|misbehavi(o)?ur|misbehave"
    r"|ghalat\s*(jawab|service|info)"
)

# Trouble / brokenness / glitch words (English + Roman Urdu).
_TROUBLE_WORDS = (
    r"not\s*working|doesn\'?t\s*work|isn\'?t\s*working|broken|defective"
    r"|error(\s*aa\s*raha|\s*code|\s*message)?|failed|failing|fail"
    r"|issue|problem|bug|glitch|fault|kharab|kharaab"
    # Generic "not / doesn't / won't / can't + action verb" so phrases like
    # 'page not opening', 'form not submitting', 'site won't load',
    # 'video doesn't play', 'login not happening' route to troubleshooting.
    r"|(not|doesn\'?t|don\'?t|isn\'?t|won\'?t|wont|cannot|can\'?t|cant|unable\s*to)"
    r"\s*(work(ing)?|open(ing)?|load(ing)?|launch(ing)?|respond(ing)?"
    r"|start(ing)?|connect(ing)?|submit(ting)?|save|saving|send(ing)?"
    r"|appear(ing)?|show(ing)?|happen(ing)?|run(ning)?|play(ing)?"
    r"|display(ing)?|updat(e|ing)|refresh(ing)?|click(ing)?|press(ing)?"
    r"|login|log\s*in|sign\s*in|access(ing)?|reach(ing)?)"
    # Hangs / crashes / freezes (cover plural / -ing / Roman Urdu forms).
    r"|crash(es|ed|ing|\s*ho\s*(jata|jati|jaa))?"
    r"|hangs?|hanging|hang\s*ho\s*(jata|jati)"
    r"|stuck|frozen|freeze(s|d|n)?|freezing"
    # Page / form / submit / button stays in a loading state forever.
    r"|loading|loads?|loaded\s*nahi"
    r"|(keeps?|still|stuck|endless(ly)?|forever|infinite(ly)?|long\s*time)\s*loading"
    r"|loading\s*(forever|hi|hee|aa\s*raha|aati\s*hai|aata\s*hai|hota\s*hai|hoti\s*hai|rehta\s*hai|rehti\s*hai|nahi\s*ruk(ti|ta)?)"
    r"|won\'?t\s*load|wont\s*load|doesn\'?t\s*load|did\s*not\s*load"
    r"|cannot\s*load|can\'?t\s*load|fails?\s*to\s*load|fail\s*to\s*load"
    # "Nothing happens after I click" / Roman Urdu equivalent.
    r"|nothing\s*happens|nothing\s*is\s*happening"
    r"|kuch\s*(nahi|ni|na)\s*(hota|hoti|ho\s*raha|ho\s*rahi)"
    # Page / site won't open.
    r"|page\s*(nahi\s*khul(ta|ti|raha|rahi)?|nahi\s*chal(ta|ti|raha|rahi)?)"
    r"|nahi\s*khul(ta|ti|raha|rahi)?|khulta\s*nahi|khulti\s*nahi"
    r"|open\s*nahi\s*ho(ta|ti|\s*raha|\s*rahi)?"
    # Slowness.
    r"|slow|lag(g(ed|ing))?|laggy|heavy|delay(ed|ing)?"
    # Generic "can't login / can't connect / can't open".
    r"|can\'?t\s*(connect|login|log\s*in|access|open|start|reach|sign\s*in|submit|press|click)"
    r"|cant\s*(connect|login|log\s*in|access|open|start|reach|sign\s*in|submit|press|click)"
    r"|unable\s*to\s*(connect|login|log\s*in|access|open|start|reach|sign\s*in|submit|click)"
    # Submit / form / button specific Roman-Urdu trouble phrasings.
    r"|submit\s*(button)?\s*(nahi\s*ho|nahi\s*hota|nahi\s*hoti|nahi\s*ho\s*raha|nahi\s*ho\s*rahi)"
    r"|submit\s*(button)?\s*(fail(ed|s)?|stuck|hang|loading)"
    r"|(button|form)\s*(nahi\s*chal|nahi\s*kaam|nahi\s*khul|nahi\s*ho\s*raha)"
    r"|click\s*karne\s*(par|pe|ke\s*baad)"
    # Connectivity / login / permission Roman Urdu.
    r"|connect\s*nahi\s*ho\s*raha|login\s*nahi\s*ho\s*raha|signin\s*nahi\s*ho\s*raha"
    r"|kaam\s*nahi\s*kar\s*raha|kaam\s*ni\s*kr\s*raha|kam\s*nahi\s*kr\s*raha"
    r"|permission\s*nahi\s*mil(ti|\s*rahi)?"
    r"|masla|msla|dikkat|pareshani"
    # Help-me-fix verbs.
    r"|fix|troubleshoot|debug|help\s*me\s*(fix|solve)|theek\s*kar(o|na|do)|hal\s*karo"
)
# Price / quotation / commercial queries (English + Roman Urdu + SMS shorthands).
_PRICE_WORDS = (
    # ── direct price / cost asks ────────────────────────────────────────
    r"price|prices|pricing|cost|costs|charges?|fee|fees|rate|rates"
    r"|how\s*much|kitna\s*(hai|hoga|lagta|parta)|kitni\s*(hai|hogi|lagti|parti)"
    r"|kya\s*qeemat|kya\s*keemat|qeemat\s*kya|keemat\s*kya"
    r"|kitna\s*paisa|kitne\s*paise|kitna\s*rupees?|amount"
    # ── quotation / estimate ─────────────────────────────────────────────
    r"|quot(e|ation)|quotations?|estimate|estimates?|proposal"
    r"|qoute|qoutation"                          # common misspellings
    r"|send\s*(me\s*)?(a\s*)?(quote|quotation|proposal|estimate)"
    r"|quote\s*(bhejo|karo|do|dein|send\s*karo)"
    r"|quotation\s*(chahiye|chahye|do|dein|bhejo|send\s*karo)"
    # ── packages / plans / bundles ───────────────────────────────────────
    r"|package|packages|plan|plans|bundle|bundles"
    r"|monthly\s*(plan|package|cost|price|charges?)"
    r"|yearly\s*(plan|package|cost|price|charges?)"
    r"|annual\s*(plan|package|cost|price|charges?)"
    r"|subscription\s*(cost|price|charges?|fee)"
    # ── discounts / deals / offers ───────────────────────────────────────
    r"|discount|discounts?|promo|promotion|offer|offers?|deal|deals?"
    r"|special\s*(price|offer|rate|deal)|best\s*price|last\s*price|final\s*price"
    r"|koi\s*(offer|discount|deal)\s*(hai|hy|h)?"
    r"|sale|clearance|reduced|reduction"
    # ── affordability / budget ───────────────────────────────────────────
    r"|budget|affordable|reasonable|cheap|sasta|sasti|mehnga|mehngi|expensive"
    r"|value\s*for\s*money|cost[\s\-]?effective|economical"
    # ── billing / invoice / tax ──────────────────────────────────────────
    r"|invoice|bill|receipt|billing|tax\s*invoice|gst|vat"
    r"|payment\s*(plan|schedule|option|method)"
    r"|installment|installments?|qist|qisti|emi"
    r"|advance|down\s*payment|partial\s*payment"
    # ── trial / demo ─────────────────────────────────────────────────────
    r"|free\s*trial|trial\s*period|trial\s*available|demo\s*(available|chahiye|milega)"
    r"|try\s*for\s*free|kya\s*(free|trial|demo)\s*(hai|hy|milta|milti)?"
)
# Compositional heuristic: a "loading / hang / stuck / not working / nothing
# happens" verb co-occurring with a UI action ("submit / click / press / tap /
# button / form / page / website") in either order is a clear trouble report.
# We compile this once at module load and check it as an additional pass.
_TROUBLE_COMPOSITIONAL = re.compile(
    r"\b(loading|loads?|hang(s|ing|ed)?|stuck|frozen|crash(es|ed|ing)?"
    r"|nothing\s*happen(s|ing)?|kuch\s*(nahi|ni)|fails?|won\'?t|wont"
    r"|doesn\'?t|cannot|can\'?t|cant|unable)\b"
    r".*?"
    r"\b(submit(\s*button)?|click(s|ed|ing)?|press(es|ed|ing)?"
    r"|tap(s|ped|ping)?|button|form|page|website|site|portal|link)\b"
    r"|"
    r"\b(submit(\s*button)?|click(s|ed|ing)?|press(es|ed|ing)?"
    r"|tap(s|ped|ping)?|button|form|page|website|site|portal|link)\b"
    r".*?"
    r"\b(loading|loads?|hang(s|ing|ed)?|stuck|frozen|crash(es|ed|ing)?"
    r"|nothing\s*happen(s|ing)?|kuch\s*(nahi|ni)|fails?"
    r"|not\s*working|doesn\'?t\s*work|won\'?t|wont|nahi)\b",
    re.IGNORECASE,
)

# Phrases that escalate to a human / live agent / customer-care channel.
_HUMAN_HANDOFF_WORDS = (
    r"human|agent|representative|real\s*person|operator|customer\s*care"
    r"|live\s*(agent|support|chat|person|operator)"
    r"|customer\s*(service|support)"
    r"|talk\s*to\s*(someone|a\s*person|human|operator)"
    r"|speak\s*to\s*(someone|a\s*person|an?\s*agent|human|operator)"
    r"|connect\s*me\s*(to|with)?\s*(a\s*)?(human|agent|representative|operator|person)?"
    r"|callback|call\s*me(\s*back)?|(can|could)\s*you\s*call\s*me"
    r"|(escalate|escalation)|manager|supervisor|senior(\s*support)?|higher\s*support"
)

# Phrases that ask the bot to provide ITCS contact details or office
# addresses. Keep "where are you / where is your office / located / based /
# situated / locations / branches / city / cities" entries here so
# location-style questions ("Where are you located?", "Do you have an
# office in Lahore?", "What is your address?") all route to the
# PROVIDE_CONTACT_INFO intent and return the full street addresses
# instead of falling through to a generic ASK_FACTUAL/RAG answer that
# would only list the cities.
_CONTACT_WORDS = (
    r"contact(\s*info|\s*details|\s*us|\s*number)?"
    r"|phone(\s*number)?|whatsapp(\s*number)?|mobile(\s*number)?"
    r"|email(\s*address)?|mail\s*(address|id)|e[\-\s]?mail"
    r"|address(es)?|postal\s*address|street\s*address|office\s*address"
    r"|office\s*location|office\s*locations|head\s*office|branch(es)?"
    r"|location(s)?|locate(d)?|where\s*are\s*you(\s*(located|based|situated))?"
    r"|where\s*is\s*(your|the|itcs)\s*(office|head\s*office|hq|headquarters)"
    r"|where\s*do\s*you\s*operate|where\s*is\s*itcs(\s*(located|based))?"
    r"|do\s*you\s*have\s*(an?\s*)?(office|branch|presence)"
    r"|offices?\s*in\s*\w+|presence\s*in\s*\w+"
    r"|directions?\s*to(\s*your|\s*the)?\s*office|map\s*location"
    r"|kahan\s*(ho|hain|hai|ho\s*aap|aap\s*ho|located\s*ho)"
    r"|(office|daftar)\s*kahan(\s*hai|\s*hain)?"
    r"|reach\s*(you|itcs|out\s*to\s*you)|get\s*in\s*touch"
    r"|support\s*(email|number|line)|helpline|uan"
    r"|raabta|rabta"
)

# Session-reset / end-chat phrases.
_RESET_WORDS = (
    r"(reset|restart|clear|wipe)\s*(the\s*|this\s*|our\s*)?(chat|conversation|session|history|context|memory)"
    r"|(reset|restart|clear)\s*(kar\s*do|kr\s*do|kar\s*dein)"
    r"|(chat|conversation|session)\s*(reset|restart|clear)\s*(kar\s*do|kr\s*do)?"
    r"|new\s*(chat|session|conversation)|naya\s*(chat|session|conversation)"
    r"|start\s*over|reset\s*conversation|restart\s*conversation"
    r"|forget\s*(everything|all|this|our\s*conversation|what\s*(i|we)\s*said)"
    r"|end\s*chat|close\s*chat|exit\s*chat|exit\s*conversation"
)

# Compliments / positive feedback.
_COMPLIMENT_WORDS = (
    r"you\s*(\'?re|are)\s*(so\s*|really\s*|very\s*)?"
    r"(great|awesome|amazing|helpful|smart|the\s*best|good|wonderful|brilliant|fantastic|perfect)"
    r"|(great|good|nice|excellent|awesome|amazing|perfect|shandar|zabardast)\s*"
    r"(job|work|bot|assistant|answer|response)"
    r"|well\s*done|impressive|love\s*(you|this\s*bot)"
)

# Identity probes ("who are you?").
_IDENTITY_WORDS = (
    r"^(who|what)\s*(are|r)\s*(you|u)\b"
    r"|\bwhat\s*is\s*your\s*name\b|\btumhara\s*naam\s*kya\s*hai\b"
    r"|\bare\s*you\s*(a\s*)?(bot|human|ai|robot|chatbot|machine)\b"
    r"|\bintroduce\s*yourself\b"
    r"|\b(tum|aap)\s*(kon|kaun)\s*(ho|hain)\b"
)

# Capability probes ("what can you do?").
_CAPABILITY_WORDS = (
    r"\bwhat\s*can\s*you\s*do\b"
    r"|\bwhat\s*(do|are)\s*you\s*(able\s*to\s*do|capable\s*of)\b"
    r"|\bhow\s*(can|could)\s*you\s*help\b"
    r"|\bwhat\s*are\s*your\s*(features|capabilities|skills|functions)\b"
    r"|\bhelp\s*me\s*with\s*what\b"
    r"|\bshow\s*(me\s*)?(the\s*)?(menu|options|categories|list)\b"
    r"|\b(menu|options|categories)\s*(dikhao|dikha\s*do|dikha\s*dein|btao|bata\s*do|do)\b"
)


# ---------------------------------------------------------------------------
# Order matters: patterns are evaluated top-to-bottom and the first match wins.
# More specific intents should appear before more generic ones.
# ---------------------------------------------------------------------------
_INTENT_PATTERNS: list[tuple[Intent, re.Pattern[str]]] = [
    # ---- Session reset (check early; resets override other cues) ---------
    # Must refer to the chat/conversation/context itself, not user data like
    # passwords or settings.
    (Intent.SESSION_RESET, _p(rf"\b({_RESET_WORDS})\b")),

    # ---- Human handoff ---------------------------------------------------
    (Intent.HUMAN_HANDOFF, _p(rf"\b({_HUMAN_HANDOFF_WORDS})\b")),

    # ---- Provide contact info -------------------------------------------
    (Intent.PROVIDE_CONTACT_INFO, _p(rf"\b({_CONTACT_WORDS})\b")),

    # ---- Bot identity ----------------------------------------------------
    (Intent.BOT_IDENTITY, _p(_IDENTITY_WORDS)),

    # ---- Bot capability --------------------------------------------------
    (Intent.BOT_CAPABILITY, _p(_CAPABILITY_WORDS)),

    # ---- Repeat ----------------------------------------------------------
    (Intent.REPEAT, _p(
        r"\b(say\s*(that\s*)?again|repeat(\s*that|\s*please|\s*it)?"
        r"|come\s*again|pardon(\s*me)?|what\s*did\s*you\s*say"
        r"|phir\s*(se\s*)?(batao|bolo|kaho)|dobara\s*(batao|bolo|kaho))\b"
    )),

    # ---- Summarize -------------------------------------------------------
    (Intent.SUMMARIZE, _p(
        r"\b(summar(ize|ise|y|ise\s*it)|tl;dr|in\s*short|short\s*version"
        r"|recap|brief(\s*summary|\s*version)?"
        r"|give\s*me\s*(a|the)\s*summary"
        r"|short\s*mein\s*(batao|bolo|bata\s*do))\b"
    )),

    # ---- Ask for examples ------------------------------------------------
    (Intent.ASK_FOR_EXAMPLES, _p(
        r"\b(give|show|provide)\s*(me\s*)?(an?\s*)?(example|examples|sample|samples|use\s*cases?|misal|misalain)\b"
        r"|\bfor\s*example\??$|\blike\s*what\??$|\bsuch\s*as\s*what\??$"
    )),

    # ---- Confirm understanding -------------------------------------------
    (Intent.CONFIRM_UNDERSTANDING, _p(r"\b(do\s*you\s*understand|did\s*you\s*(get|understand)\s*(that|me)|does\s*that\s*make\s*sense|are\s*we\s*(clear|on\s*the\s*same\s*page)|got\s*it\??)\b")),

    # ---- Confirm entities ------------------------------------------------
    (Intent.CONFIRM_ENTITIES, _p(r"\b(is\s*that\s*(correct|right)|did\s*i\s*(say|get)\s*that\s*right|is\s*this\s*what\s*you\s*mean(t)?|are\s*these\s*(details|entities)\s*(correct|right)|please\s*confirm|tasdeeq)\b")),

    # ---- Disambiguate entity --------------------------------------------
    (Intent.DISAMBIGUATE_ENTITY, _p(r"\bwhich\s*(one|of\s*(them|these|those))\b|\bdo\s*you\s*mean\b|\b(do\s*you\s*mean|are\s*you\s*referring\s*to)\b|\bwhich\s*(product|service|option|plan)\b|\bkonsa\s*wala\b")),

    # ---- Contradiction resolution ---------------------------------------
    (Intent.CONTRADICTION_RESOLUTION, _p(r"\b(that\s*contradicts|but\s*(earlier|before|previously)\s*you\s*said|you\s*(just\s*)?said\s*.*\s*(different|opposite|otherwise)|that\'?s?\s*(inconsistent|contradictory)|this\s*doesn\'?t\s*match\s*what\s*you\s*said)\b")),

    # ---- Clarify user request -------------------------------------------
    (Intent.CLARIFY_USER_REQUEST, _p(
        r"\b(what\s*do\s*you\s*mean|can\s*you\s*clarify|please\s*clarify"
        r"|i\s*don\'?t\s*(get|understand)|not\s*sure\s*what\s*you\s*mean"
        r"|explain\s*(that|this)?\s*(more|better|again|karo|kar\s*do)?"
        r"|be\s*more\s*specific|detail\s*(do|den|dein)"
        r"|samjha\s*(do|dein)|samjhao|samajh\s*nahi\s*aya|samajh\s*ni\s*aya"
        r"|matlab\s*kya\s*(hai|hua)?|m(a)?tlb\s*kya"
        r"|iska\s*matlab\s*kya|meaning\s*kya)\b"
    )),

    # ---- Sort ------------------------------------------------------------
    (Intent.SORT, _p(r"\b(sort|order|rank|arrange|organize|organise)\s*(by|them|these|those|it|results|list)?\b|\b(in\s*(ascending|descending|alphabetical|chronological)\s*order)\b|\b(cheapest|most\s*expensive|highest|lowest|newest|oldest)\s*first\b")),

    # ---- Ask comparison --------------------------------------------------
    (Intent.ASK_COMPARISON, _p(r"\b(compare|comparison|vs\.?|versus|difference\s*between|differences\s*between|which\s*is\s*(better|best)|pros\s*and\s*cons|better\s*than|farq\s*(kya|kia))\b")),

    # ---- Ask recommendation ---------------------------------------------
    (Intent.ASK_RECOMMENDATION, _p(r"\b(recommend|suggest|suggestion|what\s*(do|would)\s*you\s*(recommend|suggest)|which\s*(should|would)\s*i\s*(use|choose|pick)|best\s*(option|choice|fit|one)\s*for|mashwara)\b")),

    # ---- Ask troubleshooting --------------------------------------------
    (Intent.ASK_TROUBLESHOOTING, _p(rf"\b({_TROUBLE_WORDS})\b")),

    # ---- Ask definition --------------------------------------------------
    (Intent.ASK_DEFINITION, _p(r"^(what\s*(is|are|does)\s*(a|an|the)?\s*)|\bdefine\b|\bdefinition\s*of\b|\bmeaning\s*of\b|\bwhat\s*does\s*.*\s*mean\b")),

    # ---- Ask procedural question ----------------------------------------
    (Intent.ASK_PROCEDURAL, _p(r"^how\s*(do|can|to|should|would)\s*(i|you|we|one)?\b|\bsteps?\s*to\b|\bguide\s*(me\s*)?(on|to|through)\b|\bwalk\s*me\s*through\b|\btutorial\b|\binstructions?\s*(for|on)\b|\bstep\s*by\s*step\b|\brehnumai\b")),

    # ---- Ask factual question -------------------------------------------
    (Intent.ASK_FACTUAL, _p(r"^(who|what|when|where|why|which|how\s*(many|much|long|often|old))\b|\bis\s*it\s*true\b|\btell\s*me\s*(about|the)\b")),

    # ---- Compliment ------------------------------------------------------
    (Intent.COMPLIMENT, _p(_COMPLIMENT_WORDS)),

    # ---- Complaint -------------------------------------------------------
    (Intent.COMPLAINT, _p(rf"\b({_COMPLAINT_WORDS})\b")),

    # ---- Feedback --------------------------------------------------------
    (Intent.FEEDBACK, _p(r"\b(feedback|suggestion\s*for\s*you|(i\s*have|here\'?s)\s*(some\s*)?feedback|you\s*should\s*(improve|add|fix)|it\s*would\s*be\s*(better|nice|great)\s*if|please\s*(improve|consider))\b")),

    # ---- Apology ---------------------------------------------------------
    (Intent.APOLOGY, _p(rf"^({_APOLOGY_WORDS})\b")),

    # ---- Appreciation / thanks ------------------------------------------
    (Intent.APPRECIATION, _p(rf"^({_THANKS_WORDS})\b")),

    # ---- Short affirmation ("yes", "haan", "sure", "please do", "go ahead")
    # These almost always answer a yes/no offer the bot just made
    # (e.g. "Would you like help connecting with sales?" -> "yes").
    # Route to the regular LLM-with-history path so the model reads the
    # offer from prior turns and fulfils it - never to the canned
    # "You're welcome" reply. Must be matched BEFORE the bare-ack
    # pattern below.
    (Intent.AFFIRMATION, _p(
        r"^("
        r"yes|yess+|yup|yep|yeah|y"
        r"|haan|haa+n?|han|hanji|hanjee|haanji|jee|ji|g"
        r"|sure|surely|of\s*course|definitely|absolutely"
        r"|please|pls|plz|pliz"
        r"|please\s*do|please\s*share|please\s*help|please\s*continue"
        r"|yes\s*please|yes\s*pls|yes\s*plz"
        r"|haan\s*ji|han\s*ji|ji\s*haan|ji\s*han"
        r"|do\s*it|go\s*ahead|carry\s*on|keep\s*going|continue"
        r"|kar\s*do|kar\s*dein|kr\s*do"
        r"|share\s*kar(o|do|dein|ein)?"
        r")\s*[.!?]*$"
    )),

    # ---- Short bare acknowledgements ("ok", "theek", "sahi", "noted", ...)
    # Pure acknowledgements that follow a successful answer. Route to
    # APPRECIATION so the bot acknowledges naturally instead of firing
    # the "I didn't understand" fallback. Yes/sure/please tokens have
    # been moved to the AFFIRMATION pattern above because they almost
    # always answer an open offer rather than expressing thanks.
    (Intent.APPRECIATION, _p(
        r"^("
        r"ok|okay|okey|k|kk|kay"
        r"|alright|all\s*right|fine|done|cool"
        r"|got\s*it|noted|understood|clear|makes\s*sense"
        r"|theek|thik|teek|tik|sahi|bilkul|achi?|accha|acha"
        r"|ok\s*thanks?|ok\s*thnx|ok\s*thx|ok\s*ji"
        r")\s*[.!]*$"
    )),

    # ---- Goodbye ---------------------------------------------------------
    (Intent.GOODBYE, _p(rf"^({_GOODBYE_WORDS})\b")),

    # ---- Greeting --------------------------------------------------------
    (Intent.GREETING, _p(
        rf"^({_GREETING_WORDS})\b"
        r"|^good\s*(morning|afternoon|evening|day)\b"
        r"|^(g\s*morning|gm|ga|ge)\b"
        r"|^how\s*(are|r)\s*(you|u|ya)\b"
        r"|^(what\'?s?\s*up|sup|how\s*is\s*it\s*going|how\'?s?\s*it\s*going)\b"
        rf"|^({_GREETING_HOW_ARE_YOU})\b"
    )),

    # In the _INTENT_PATTERNS list, REPLACE this existing entry:
    (Intent.ASK_COMPARISON, _p(r"\b(compare|comparison|vs\.?|versus| ... )\b")),

# WITH these two entries (price first, then comparison):
    # ---- Ask price / quotation ------------------------------------------
    (Intent.ASK_PRICE, _p(rf"\b({_PRICE_WORDS})\b")),

    # ---- Ask comparison --------------------------------------------------
    (Intent.ASK_COMPARISON, _p(r"\b(compare|comparison|vs\.?|versus|difference\s*between|differences\s*between|which\s*is\s*(better|best)|pros\s*and\s*cons|better\s*than|farq\s*(kya|kia))\b")),
]


# ---------------------------------------------------------------------------
# Knowledge-signal fallback vocabulary.
#
# When no explicit intent regex above matches, we still want to route obvious
# information / support / shopping / careers / IT questions to the RAG layer
# instead of dead-ending at FALLBACK. This is deliberately broad and covers
# English, broken English, SMS shorthand, common Roman Urdu, and ITCS-specific
# topical vocabulary (cloud, M365, networking, careers, payments, etc.).
# ---------------------------------------------------------------------------
_KNOWLEDGE_SIGNALS = re.compile(
    r"\b("
    # ── general info-request verbs / nouns ──────────────────────────────
    r"tell|explain|describe|list|show|share|need|want|looking|find|search|explore"
    r"|info|information|details|detail|tafseel|tafsil|maloomat|maloom"
    r"|btao|btayen|bhejo|forward|attach"
    # ── ITCS company / pages ────────────────────────────────────────────
    r"|home|homepage|main\s*page"
    r"|about(\s*us)?|company|hamare\s*bare\s*mein|vision|mission"
    r"|services?|solutions?|products?|offerings?|khidmaat|khidmat"
    r"|faq|sawalat|sawaalat|help\s*center|support\s*center"
    r"|blog|news|updates|article|articles"
    r"|terms|conditions|privacy(\s*policy)?|refund\s*policy|policy"
    # ── pricing / commercial ────────────────────────────────────────────
    r"|price|pricing|cost|charge|charges|fee|fees|rate|rates|budget"
    r"|estimate|quotation|quote|qoute|qeemat|keemat"
    r"|package|packages|plan|plans|bundle|bundles|offer|offers|deal|deals"
    r"|discount|discounts|promo|promotion|sale"
    r"|monthly|yearly|annually|per\s*month|per\s*year"
    r"|installment|installments|qist|qisti|advance|down\s*payment"
    r"|expensive|mehnga|mehngi|cheap|sasta|sasti|affordable|reasonable"
    r"|negotiate|negotiation|final\s*price|best\s*price|last\s*price"
    r"|tax|gst|vat|invoice|bill|receipt|tax\s*invoice"
    r"|availability|in\s*stock|out\s*of\s*stock|stock|inventory|mojud|maujood"
    r"|demo|trial|free\s*trial|testing|sample"
    r"|features|specs|specifications|functions|capability|capabilities"
    r"|comparison|compare|difference|farq"
    # ── orders / delivery ───────────────────────────────────────────────
    r"|order|orders|booking|bookings|reserve|request"
    r"|track|tracking|status|order\s*status"
    r"|delivery|shipping|dispatch|courier|rider|parcel|shipment"
    r"|expected\s*date|eta|kab\s*tak|kitni\s*dair"
    r"|delay|late|jaldi|urgent|asap|fatafat|emergency|critical"
    r"|address\s*change|delivery\s*change|location\s*update"
    r"|cancel\s*order|return|exchange|replace|refund|chargeback"
    r"|damaged|defective|broken|kharab|fault"
    r"|complaint|complain|shikayat|report|ticket|case"
    r"|response\s*kab|reply\s*kab|no\s*response|follow\s*up|followup"
    # ── account / auth ──────────────────────────────────────────────────
    r"|account|profile|user(\s*id)?|customer(\s*id)?|client"
    r"|login|log\s*in|sign\s*in|signin|signup|sign\s*up|register|registration"
    r"|create\s*account|password|pass|pwd|pin|security\s*code"
    r"|forgot\s*password|reset\s*password|change\s*password"
    r"|otp|verification(\s*code)?|sms\s*code|email\s*code"
    r"|locked|block(ed)?|banned|suspended"
    r"|username|user\s*name|logout|sign\s*out"
    r"|privacy|security|2fa|two\s*factor|authentication|sso"
    # ── payments ────────────────────────────────────────────────────────
    r"|payment|pay|paid|pending|due"
    r"|card|debit|credit|visa|mastercard"
    r"|bank\s*transfer|iban|account\s*number"
    r"|cash\s*on\s*delivery|cod"
    r"|subscription|renew|renewal|cancel\s*subscription"
    r"|failed\s*payment|declined|transaction(\s*error|\s*failed)?"
    r"|recurring"
    # ── general IT / networking ─────────────────────────────────────────
    r"|server|servers|network|networking|internet|wifi|wi[\-\s]?fi|lan|wan|vpn"
    r"|firewall|antivirus|malware|ransomware|phishing|threat|cyber|cybersecurity"
    r"|backup|backups|restore|recovery|disaster\s*recovery|dr"
    r"|downtime|outage|service\s*down|unavailable"
    r"|performance|latency|bandwidth|throughput|uptime|sla"
    r"|patch|patches|update|upgrade|version|hotfix"
    # ── cloud / Microsoft / Azure / AI ──────────────────────────────────
    r"|azure|aws|gcp|cloud|hybrid\s*cloud|multi[\-\s]?cloud"
    r"|tenant|resource\s*group|rg|subscription|region"
    r"|vm|virtual\s*machine|storage|blob|sql|database|cosmos|table|queue"
    r"|entra|azure\s*ad|aad|active\s*directory|identity"
    r"|m365|o365|office\s*365|microsoft\s*365|teams|sharepoint|onedrive|exchange|outlook"
    r"|copilot|ai|artificial\s*intelligence|chatbot|chat\s*bot|llm|gpt"
    r"|rag|ai\s*search|index|embedding|vector"
    r"|api|apis|integration|connector|webhook|sdk"
    r"|deployment|pipeline|ci\s*cd|devops|dev|qa|prod|environment|env"
    r"|permissions|access|role|roles|rbac|policy"
    r"|sync|refresh|cache(\s*clear)?|cookies"
    r"|configuration|config|settings|setup|install|uninstall|reinstall"
    r"|error\s*code|401|403|404|500|502|503"
    # ── careers / hiring ────────────────────────────────────────────────
    r"|career|careers|job|jobs|vacancy|vacancies|opening|openings|position|role"
    r"|hire|hiring|recruitment|interview|screening"
    r"|apply|application|cv|resume|cover\s*letter"
    r"|internship|intern|trainee|fresh\s*graduate|fresher|entry\s*level"
    r"|experience|exp|years|skills|qualification|qualifications"
    r"|salary|stipend|pay|package"
    r"|remote|onsite|on[\-\s]site|hybrid|wfh|work\s*from\s*home"
    r"|notice\s*period|joining|start\s*date"
    r"|career\s*path|growth|promotion|hr|human\s*resources|team\s*lead|manager"
    # ── time / scheduling ───────────────────────────────────────────────
    r"|today|aaj|tomorrow|kal|yesterday|parson"
    r"|abhi|right\s*now|now"
    r"|timing|hours|working\s*hours|office\s*hours|business\s*hours"
    r"|appointment|meeting|schedule|slot|session"
    r"|morning|afternoon|evening|night|shaam|raat|dopahar|subah"
    r"|monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    r"|somwar|mangal|budh|jumeraat|juma|haftay|hafta|itwar"
    # ── help / guidance ─────────────────────────────────────────────────
    r"|help|madad|support|guide|guidance|rehnumai|assistance|assist"
    # ── form / contact-form fields (often signal a request) ─────────────
    r"|name|naam|subject|topic|reason|message|description|captcha"
    # ── Roman Urdu information / question / request words ──────────────
    r"|kya|kia|kiya|kaisa|kaise|kese|kesay|kaisay|kahan|kab|kyun|kyon|q|qn"
    r"|kitna|kitni|kitne|kon|kaun|konsa|konsi|kis|kyon\s*ki"
    r"|batao|bata|batayein|bata\s*do|bata\s*dein|btao|btayen"
    r"|chahiye|chahye|chahta|chahti|chahye"
    r"|milta|milti|mil\s*sakta|mil\s*sakti"
    r"|mashwara|raabta|rabta"
    r"|kindly"
    r")\b",
    re.IGNORECASE,
)


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

    # Compositional trouble heuristic: loading / hang / stuck / "not
    # working" / "nothing happens" co-occurring with a UI action like
    # submit / click / button / form / page catches phrases that the
    # word-list alone misses, e.g. "your website is loading when I
    # click submit", "page keeps loading after submit".
    if _TROUBLE_COMPOSITIONAL.search(cleaned):
        return Intent.ASK_TROUBLESHOOTING

    if cleaned.endswith("?") or _KNOWLEDGE_SIGNALS.search(cleaned):
        return Intent.ASK_FACTUAL

    return Intent.FALLBACK


def is_chitchat(text: str) -> bool:
    """Backwards-compatible helper: True for social small-talk intents."""
    return classify_intent(text) in CHITCHAT_INTENTS
