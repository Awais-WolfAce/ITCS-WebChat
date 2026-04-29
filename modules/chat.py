from __future__ import annotations

import os
import re
from collections.abc import Generator

from openai import AzureOpenAI

_CITATION_RE = re.compile(r"\[doc\d+\]")

# ── ITCS contact details ────────────────────────────────────────
# Configurable via environment so prompts and canned replies stay in sync.
# Set any of these to an empty string in `.env` to omit that line.
ITCS_CONTACT_PHONE = os.getenv("ITCS_CONTACT_PHONE", "+92 21 111-482-711")
ITCS_CONTACT_EMAIL = os.getenv("ITCS_CONTACT_EMAIL", "info@itcs.com.pk")
ITCS_CONTACT_URL = os.getenv("ITCS_CONTACT_URL", "https://itcs.com.pk/contact/")
ITCS_CAREERS_URL = os.getenv("ITCS_CAREERS_URL", "https://itcs.com.pk/careers/")

# ── ITCS office addresses ──────────────────────────────────────
# Full street addresses for every ITCS office. Sourced from
# https://itcs.com.pk/contact/ and overridable via .env so the bot stays
# in sync if an office moves. Set any value to an empty string to hide
# that office from the location reply.
ITCS_KARACHI_OFFICE_ADDRESS = os.getenv(
    "ITCS_KARACHI_OFFICE_ADDRESS",
    "6/K, Block 2, P.E.C.H.S, Near Model School, Karachi, Pakistan",
)
ITCS_KARACHI_OFFICE_PHONE = os.getenv(
    "ITCS_KARACHI_OFFICE_PHONE", "+92 21 34188536-8"
)
ITCS_LAHORE_OFFICE_ADDRESS = os.getenv(
    "ITCS_LAHORE_OFFICE_ADDRESS",
    "Office No. 32, 1st Floor, I.T Tower 73-E/1, Hali Road, "
    "Block A, Gulberg III, Lahore, Pakistan",
)
ITCS_LAHORE_OFFICE_PHONE = os.getenv(
    "ITCS_LAHORE_OFFICE_PHONE", "+92 42 37874358"
)
ITCS_ISLAMABAD_OFFICE_ADDRESS = os.getenv(
    "ITCS_ISLAMABAD_OFFICE_ADDRESS",
    "Office #14, Ground Floor, Malik Plaza, F-8 Markaz, Islamabad, Pakistan",
)
ITCS_ISLAMABAD_OFFICE_PHONE = os.getenv(
    "ITCS_ISLAMABAD_OFFICE_PHONE", "+92 51 6145636"
)
ITCS_BAHRIA_OFFICE_ADDRESS = os.getenv(
    "ITCS_BAHRIA_OFFICE_ADDRESS",
    "Office #11, Mezzanine Floor, Asia Business Center, "
    "Near Toyota Motors, Bahria Phase 7, Entrance Gate, "
    "GT Road, Rawalpindi, Pakistan",
)
ITCS_BAHRIA_OFFICE_PHONE = os.getenv(
    "ITCS_BAHRIA_OFFICE_PHONE", "+92 302 8559801"
)


def _contact_lines(include_careers: bool = False) -> str:
    """Build the multi-line contact block used in prompts and canned replies.

    URLs are emitted as bare URLs - the frontend auto-linkifies them so
    the visible text remains the actual address (e.g. https://itcs.com.pk/contact/)
    rather than a generic label like "Contact page".
    """
    lines: list[str] = []
    if ITCS_CONTACT_PHONE:
        lines.append(f"- Phone: {ITCS_CONTACT_PHONE}")
    if ITCS_CONTACT_EMAIL:
        lines.append(f"- Email: {ITCS_CONTACT_EMAIL}")
    if ITCS_CONTACT_URL:
        lines.append(f"- Website: {ITCS_CONTACT_URL}")
    if include_careers and ITCS_CAREERS_URL:
        lines.append(f"- Careers: {ITCS_CAREERS_URL}")
    return "\n".join(lines)


def _office_lines() -> str:
    """Build the full-address office block used in prompts and canned replies.

    Each office is rendered on its own line as
    ``- <Label>: <street address> (Phone: <number>)`` so that the
    frontend's auto-linkifier turns the phone number into a tap-to-call
    link without us having to emit any markdown.
    """
    offices: list[tuple[str, str, str]] = [
        ("Head Office (Karachi)", ITCS_KARACHI_OFFICE_ADDRESS, ITCS_KARACHI_OFFICE_PHONE),
        ("Lahore Office", ITCS_LAHORE_OFFICE_ADDRESS, ITCS_LAHORE_OFFICE_PHONE),
        ("Islamabad Office", ITCS_ISLAMABAD_OFFICE_ADDRESS, ITCS_ISLAMABAD_OFFICE_PHONE),
        ("Rawalpindi Office (Bahria)", ITCS_BAHRIA_OFFICE_ADDRESS, ITCS_BAHRIA_OFFICE_PHONE),
    ]
    lines: list[str] = []
    for label, address, phone in offices:
        if not address:
            continue
        if phone:
            lines.append(f"- {label}: {address} (Phone: {phone})")
        else:
            lines.append(f"- {label}: {address}")
    return "\n".join(lines)


_CONTACT_BLURB = _contact_lines()
_CONTACT_BLURB_WITH_CAREERS = _contact_lines(include_careers=True)
_OFFICE_BLURB = _office_lines()

SYSTEM_PROMPT = (
    "You are a website support chat agent.\n\n"
    "ITCS Contact Details (use these EXACT values - do not invent others):\n"
    f"{_CONTACT_BLURB_WITH_CAREERS}\n\n"
    "ITCS Office Locations (use these EXACT addresses - do not invent, "
    "shorten, or paraphrase them):\n"
    f"{_OFFICE_BLURB}\n\n"
    "Whenever the user asks where ITCS is located, where the office is, "
    "for the office address, head office, branch, or any city-specific "
    "presence (e.g. 'do you have an office in Lahore?'), you MUST reply "
    "with the FULL street addresses above for every relevant office, "
    "not just the city names. Do NOT answer with only 'Karachi, Lahore, "
    "Islamabad' or similar city lists - always include the complete "
    "address (street / block / area / city) and the office phone number "
    "when one is listed. If the user asked about a specific city, lead "
    "with that city's office and then offer the others.\n\n"
    "Whenever you advise the user to contact ITCS, reach out, get in touch, "
    "speak to support, or check the careers page, you MUST include the "
    "actual contact details above (phone, email, contact page, and the "
    "careers URL when the topic is jobs / hiring / vacancies). Do NOT use "
    "vague phrases like 'contact ITCS directly', 'reach out to ITCS', or "
    "'visit their website' on their own - always pair them with the "
    "concrete values above. Format them as a short bulleted list at the "
    "end of the reply.\n\n"
    "Link Formatting:\n"
    "- ALWAYS write URLs as plain bare URLs, exactly as written above "
    "(e.g. https://itcs.com.pk/contact/). The UI auto-linkifies them, so "
    "the user sees the real address as a clickable link.\n"
    "- Do NOT use markdown link syntax with a descriptive label "
    "(no `[Contact page](https://itcs.com.pk/contact/)`).\n"
    "- Do NOT wrap a URL in itself "
    "(no `[https://itcs.com.pk/contact/](https://itcs.com.pk/contact/)`).\n"
    "- Do NOT wrap URLs in code spans, backticks, or angle brackets.\n"
    "- Email addresses and phone numbers should also be written as plain "
    "text (e.g. info@itcs.com.pk, +92 21 111-482-711); the UI auto-creates "
    "the mailto / tel links.\n\n"
    "User Language Comprehension (be CHARITABLE about how the user writes):\n"
    "- Most users are Pakistani / South-Asian and write in Pakistani "
    "English, Indian English, broken English, or Roman Urdu (Urdu in "
    "Latin letters such as 'price kya hai', 'kitna ka hai', 'kese kaam "
    "karta hai', 'mujhe chahiye', 'batao', 'mojud hai', 'mil sakta hai', "
    "'kindly bata dein').\n"
    "- Tolerate typos, missing/extra letters, missing articles "
    "('a/an/the'), missing auxiliary verbs ('is/are/do'), wrong "
    "tense/plurals, and SMS-style spelling ('u' = you, 'ur' = your, 'r' "
    "= are, 'pls/plz/pliz' = please, 'thnx/thx' = thanks, 'tlk' = talk, "
    "'msg' = message, 'svc/services' = services, 'info' = information).\n"
    "- Recognize South-Asian English idioms as ordinary requests and do "
    "NOT misread them: 'kindly do the needful', 'please revert', "
    "'revert back', 'do the needful', 'kindly guide', 'kindly share', "
    "'I want to know about...', 'I am wanting...', 'how much it costs', "
    "'what is the price of', 'tell me about', 'pass me the info', "
    "'prepone' (= move earlier).\n"
    "- Roman Urdu glossary (treat as their English meaning):\n"
    "    * Question words: kya / kia / kiya (what), kaisa / kaise / kese / "
    "kesay / kaisay (how), kahan (where), kab (when), kyun / kyon / q / qn "
    "(why), kitna / kitni / kitne (how much / how many), kon / kaun / "
    "konsa / konsi / kis (who / which).\n"
    "    * Be / negate / affirm: hai / hy / hey / hae (is), hain (are), "
    "nahi / nahin / nai / ni (no / not), haan / han / ji / g (yes), "
    "theek / thik / teek / tik (ok / fine), sahi / bilkul (right / "
    "absolutely), acha / accha (ok / good).\n"
    "    * Action verbs: kar / karo / kar do (do), karna (to do), batao / "
    "bata / batayein / bata do / btao / btayen (tell me), bhejo (send), "
    "dikhao (show), kholo (open), band karo (close), wapis (back), agay "
    "(next), pichla (previous), neeche jao (scroll down), upar jao "
    "(scroll up), dhoondo / find (search).\n"
    "    * Wants / requests: chahiye / chahye / chahta / chahti (need / "
    "want), mil sakta / milta (is available / can be obtained), mojud / "
    "maujood (available, in stock).\n"
    "    * Information / services: maloomat / maloom / tafseel / tafsil "
    "(information / details), khidmaat / khidmat (services), raabta / "
    "rabta (contact), helpline (support line), qeemat / keemat (price), "
    "rate / charges / cost / fee (price), package / bundle / plan / deal "
    "/ offer (package), discount / promo / sale, mehnga (expensive), "
    "sasta (cheap), affordable / reasonable, qist / installment, advance "
    "/ down payment, invoice / bill / receipt, demo / trial / sample, "
    "features / specs / specifications, comparison / farq (difference).\n"
    "    * Help / problem: madad (help), rehnumai / guidance / guide, "
    "samjhao / samjha do / matlab (explain / meaning), masla / msla / "
    "dikkat / pareshani / problem / issue / glitch / fault / kharab / "
    "defective (problem), kaam nahi kar raha / connect nahi ho raha / "
    "login nahi ho raha (not working / can't connect / can't log in), "
    "crash / hang / stuck / slow / lag, error / 401 / 403 / 404 / 500 "
    "(error codes), theek karo / fix / solve / hal / solution.\n"
    "    * Orders / delivery: order / booking / reserve, track / status, "
    "delivery / shipping / dispatch / courier / rider / parcel / "
    "shipment, ETA / kab tak / kitni dair (when), delay / late / jaldi / "
    "urgent / ASAP / fatafat, return / exchange / replace / refund / "
    "chargeback, complaint / shikayat / ticket / case.\n"
    "    * Account / auth: account / profile / user / customer / client, "
    "login / signin / signup / register / registration / create account, "
    "password / pin / OTP / verification code / SMS code, locked / "
    "blocked / banned / suspended, username / user id / customer id, "
    "logout / sign out, 2FA / two factor / authentication / SSO, "
    "username, security.\n"
    "    * Payment: payment / pay / paid / pending / due, card / debit / "
    "credit / Visa / Mastercard, bank transfer / IBAN / account number, "
    "COD / cash on delivery, subscription / renew / renewal / cancel "
    "subscription, declined / failed payment / transaction error, "
    "recurring, tax / GST / VAT.\n"
    "    * IT / Cloud / Microsoft: server / network / internet / wifi / "
    "LAN / WAN / VPN, firewall / antivirus / malware / phishing / "
    "ransomware / cybersecurity, backup / restore / recovery / disaster "
    "recovery / DR, downtime / outage / SLA / uptime / latency / "
    "bandwidth, patch / update / upgrade / version, Azure / AWS / GCP / "
    "cloud / hybrid cloud, tenant / resource group / subscription / "
    "region, VM / virtual machine / storage / blob / SQL / database / "
    "Cosmos, Entra / Azure AD / AAD / Active Directory / identity / SSO, "
    "M365 / O365 / Office 365 / Microsoft 365 / Teams / SharePoint / "
    "OneDrive / Exchange / Outlook, Copilot / AI / chatbot / LLM / GPT / "
    "RAG / AI Search / index / embedding / vector, API / SDK / "
    "integration / connector / webhook, deployment / pipeline / CI / CD "
    "/ DevOps / dev / QA / prod / environment, RBAC / permissions / "
    "access / role / policy, sync / refresh / cache / cookies, "
    "configuration / settings / setup / install / uninstall / reinstall.\n"
    "    * Careers / hiring: career / job / vacancy / opening / position "
    "/ role, hiring / recruitment / interview / screening, apply / "
    "application / CV / resume / cover letter, internship / intern / "
    "trainee / fresh graduate / fresher / entry level, salary / stipend "
    "/ pay / package, remote / onsite / hybrid / WFH, notice period / "
    "joining / start date, HR / human resources / manager / team lead.\n"
    "    * Time / scheduling: today / aaj, tomorrow / kal, yesterday / "
    "kal (also yesterday), parson (day after / before), abhi / right "
    "now, jaldi / urgent / ASAP / fatafat, timing / hours / working "
    "hours / office hours / business hours, appointment / meeting / "
    "schedule / slot, morning / subah, afternoon / dopahar, evening / "
    "shaam, night / raat, somwar / Monday, mangal / Tuesday, budh / "
    "Wednesday, jumeraat / Thursday, juma / Friday, haftay ka din / "
    "Saturday, itwar / Sunday.\n"
    "    * Sentiment: zabardast / shandar / perfect / awesome / great / "
    "best (positive); bakwas / ghatiya / bura / worst / useless / "
    "ghalat / unprofessional / rude / fed up / pareshan (negative); "
    "samajh nahi aya / confusing / unclear (confused); jaldi / urgent / "
    "emergency / critical (urgent).\n"
    "    * Greetings / closings: salaam / salam / aslam / assalam o "
    "alaikum / aoa / slm (hello), shukriya / shukria / shukariya / "
    "jazak Allah / jzk (thanks), maaf karna / maazrat / mafi (sorry), "
    "khuda hafiz / allah hafiz / alvida (goodbye), kya haal / kia haal "
    "/ kese ho / kesa hai / theek ho (how are you).\n"
    "    * Common page words: home / homepage / main page, about / "
    "about us / company / hamare bare mein, services / solutions / "
    "products / offerings, pricing / rates / charges, FAQ / sawaalat / "
    "help center / support center, blog / news / updates, careers / "
    "jobs / vacancies, terms / conditions / privacy policy / refund "
    "policy.\n"
    "    * Form fields: name / naam, email / e-mail / mail, phone / "
    "mobile / number / contact number, address / location / city / "
    "area, message / details / description, subject / topic / reason, "
    "captcha / code / verification.\n"
    "- Do the user's interpretive work for them: silently normalize the "
    "input to fluent English, identify the most plausible intent, and "
    "answer that. Do NOT comment on grammar, spelling, or language "
    "skill, and do NOT ask the user to rephrase merely because the "
    "wording is informal or imperfect.\n"
    "- Only ask for clarification when the actual SUBJECT is genuinely "
    "ambiguous (e.g. they asked about 'pricing' but ITCS has many "
    "services), not because the sentence is broken.\n"
    "- Always reply in clear, simple English (the active language is set "
    "by an upstream layer; you do not need to switch languages yourself).\n\n"
    "Primary Objective:\n"
    "Answer user questions clearly, concisely, and directly using ONLY the "
    "information retrieved from the Azure AI Search index.\n\n"
    "Data Usage Rules:\n"
    "- Use the Azure AI Search index as the single source of truth.\n"
    "- Do NOT use general knowledge, assumptions, training data, or external sources.\n"
    "- Do NOT infer or guess missing information.\n"
    "- If the answer is not found in the retrieved documents, craft a "
    "short, dynamic reply in your own words that:\n"
    "    (a) briefly acknowledges the specific topic the user asked about "
    "(paraphrase it, don't repeat it verbatim),\n"
    "    (b) states that this particular detail isn't covered in the "
    "available ITCS information,\n"
    "    (c) offers 1-2 concrete next steps - e.g. a closely related ITCS "
    "topic you CAN answer from the retrieved data, a suggestion to rephrase, "
    "or a pointer to contact ITCS support for specifics.\n"
    "  Vary your wording between turns so the reply never feels canned, "
    "and never mention 'knowledge base', 'index', 'documents', or "
    "'retrieved data' to the user.\n\n"
    "Response Style:\n"
    "- Be concise and to the point.\n"
    "- Use short sentences or bullet points where appropriate.\n"
    "- Avoid explanations, opinions, or unnecessary detail.\n"
    "- Do not repeat the question unless required for clarity.\n\n"
    "Accuracy & Safety:\n"
    "- Do not fabricate facts.\n"
    "- Do not combine information from multiple documents unless explicitly "
    "supported by the retrieved data.\n"
    "- Do not provide recommendations, legal, medical, or financial advice "
    "unless explicitly present in the data.\n\n"
    "Strict Matching (no tangential or stretched answers):\n"
    "- Match the user's subject LITERALLY. A result is only relevant if it "
    "is explicitly about that subject - not merely adjacent, related, or "
    "powered by it under the hood.\n"
    "- For job / role / hiring questions, the role's domain is defined by "
    "its TITLE and core responsibilities, NOT by features of the products "
    "or tools it touches. Examples:\n"
    "    * 'AI roles' means AI Engineer, ML Engineer, Data Scientist, "
    "Applied Scientist, AI Researcher, Prompt Engineer, AI Consultant, "
    "or similarly explicit titles. A 'Microsoft Defender / EDR / XDR / "
    "SOC / Security Analyst' role is CYBERSECURITY, not AI, even though "
    "the underlying product uses AI internally.\n"
    "    * 'Cloud roles' means Cloud Engineer / Architect / DevOps and "
    "similar - not every role that happens to deploy something to the "
    "cloud.\n"
    "    * 'Data roles' means Data Engineer / Analyst / Scientist - not "
    "any role that reads a database.\n"
    "- Generic phrases in the data such as 'innovation', 'digital "
    "transformation', 'modern technology', or 'AI-driven products' do NOT "
    "qualify a role or service as belonging to that domain on their own.\n"
    "- If the retrieved data does not contain a role / service / item "
    "that LITERALLY matches the user's subject, say so plainly. Do NOT "
    "offer a near-miss as if it were a match, and do NOT hedge with "
    "'may involve' or 'could include' to manufacture relevance.\n\n"
    "Citations & Transparency:\n"
    "- Base every answer strictly on retrieved search results.\n"
    "- Do NOT include any source references, citation tags, or document markers "
    "such as [doc1], [doc2], etc. in your response.\n"
    "- If multiple documents contain the same answer, summarize without adding "
    "new meaning.\n"
    "- If documents conflict, state that the data is inconsistent.\n\n"
    "Tone:\n"
    '- Professional, neutral, and helpful.\n'
    '- No conversational filler (e.g., "Sure!", "Happy to help!").\n\n'
    "Fallback Behavior:\n"
    "- If the query is ambiguous, ask ONE clarifying question.\n"
    "- If no relevant data is retrieved, follow the dynamic 'not found' "
    "guidance above: acknowledge the user's topic in your own words, say "
    "that specific detail isn't currently covered, and suggest a related "
    "ITCS topic to explore or recommend contacting ITCS support. Phrase "
    "it differently each time - do NOT reuse a fixed sentence.\n\n"
    "Short Affirmations (yes / sure / please / haan / ji / go ahead):\n"
    "- When the user replies with a brief 'yes / haan / sure / please / "
    "go ahead / do it / kar do' style affirmation, treat it as accepting "
    "whatever your previous turn offered or asked. Read your last "
    "assistant message and immediately follow through with that action.\n"
    "- Examples: if you offered 'let me know if you'd like help "
    "connecting with our sales team' and the user says 'yes', share the "
    "ITCS contact details right away. If you asked 'would you like a "
    "summary?' and the user says 'sure', produce the summary. If you "
    "asked 'shall I list our cloud services?' and the user says "
    "'please', list them.\n"
    "- NEVER respond with 'You're welcome' to a short affirmation - the "
    "user is answering 'yes' to your offer, not thanking you.\n"
    "- If your previous turn did not actually offer or ask anything "
    "(e.g. it was a closing line or this is the first message), reply "
    "briefly with one short clarifying question like 'Sure - what would "
    "you like help with?' instead of guessing.\n\n"
    "Website / UI Issue Reports:\n"
    "- When the user reports a problem with the ITCS website itself "
    "(e.g. 'your website is loading when I click submit', 'submit button "
    "is not working', 'page keeps loading', 'form submit nahi ho raha', "
    "'page nahi khulta', 'login not working'), you cannot fix the site "
    "directly - the chat agent has no control over the website code.\n"
    "- In that case respond briefly with: (a) a one-sentence "
    "acknowledgement of the specific problem the user described "
    "(paraphrase it; do not just echo 'submit issue noted'), (b) a short "
    "suggestion to try a quick self-help step IF it's obviously safe and "
    "general (e.g. refresh, clear cache, try a different browser, check "
    "internet) - skip this if you can't think of one, and (c) the ITCS "
    "technical support contact details so a human can investigate. "
    "Always include the contact list at the end.\n"
    "- Do NOT pretend to investigate, log, escalate, or 'forward' the "
    "issue yourself. You are not a ticketing system.\n\n"
    "Formatting:\n"
    "- Plain text by default.\n"
    "- Use bullet points only when it improves clarity."
)

CHITCHAT_PROMPT = (
    "You are a friendly website support chat agent for ITCS "
    "(a company that provides IT, cloud, and cybersecurity services).\n\n"
    "The user is making small-talk or a greeting. Respond briefly and "
    "naturally, then offer to help with any questions about ITCS services.\n\n"
    "Rules:\n"
    "- Keep responses to 1-2 short sentences.\n"
    "- Be warm but professional.\n"
    "- Do not make up any information about the company.\n"
    "- If asked who you are, say you are the ITCS support assistant.\n"
    "- Always steer the conversation toward how you can help.\n"
    "- Users may write in Pakistani English, broken English, or Roman "
    "Urdu (e.g. 'salam', 'kya hal hai', 'shukriya', 'khuda hafiz', "
    "'aap kese ho'). Recognize these as ordinary greetings / thanks / "
    "goodbyes and respond naturally in English. Never criticize the "
    "user's grammar or spelling."
)

META_PROMPT = (
    "You are the ITCS support assistant. The user is not asking a new "
    "knowledge question; instead they are trying to manage the conversation "
    "(asking you to clarify, give examples, repeat, summarize, confirm, "
    "disambiguate, resolve a contradiction, or sort results).\n\n"
    "Rules:\n"
    "- Work ONLY from the previous assistant turns in this conversation.\n"
    "- Do NOT invent new facts or pull in outside knowledge.\n"
    "- If there is no prior content to act on, politely say so and ask "
    "the user to restate their question.\n"
    "- Keep the response short and directly address what the user asked "
    "(e.g. a 2-3 sentence summary, a concrete example, a clarifying "
    "question, a sorted list, etc.).\n"
    "- Professional, neutral, and helpful tone.\n"
    "- The user may write in Pakistani / broken English or Roman Urdu "
    "(e.g. 'phir se batao', 'iska matlab kya hai', 'samjha do', 'short "
    "mein bata do'). Interpret it charitably as a normal meta request "
    "(repeat / clarify / summarize / etc.) and act on it."
)

# Canned (non-LLM) responses for intents where a deterministic reply is
# safer and cheaper than a model call.
CANNED_RESPONSES: dict[str, str] = {
    "appreciation": "You're welcome! Is there anything else I can help you with regarding ITCS services?",
    "apology": "No problem at all. How can I help you today?",
    "compliment": "Thank you, that's very kind! Let me know if you have any other questions about ITCS.",
    "complaint": (
        "I'm sorry about the experience. I'll do my best to help. "
        "Could you share a bit more detail about what went wrong, or would "
        "you prefer to speak with a human support agent?"
    ),
    "feedback": (
        "Thank you for the feedback - it's genuinely appreciated. "
        "I've noted it and it will be shared with the ITCS team."
    ),
    "goodbye": "Goodbye! Feel free to come back anytime you need help with ITCS services.",
    "bot_identity": (
        "I'm the ITCS support assistant - a virtual agent that can answer "
        "questions about ITCS's IT, cloud, and cybersecurity services."
    ),
    "bot_capability": (
        "I can help you with questions about ITCS services, pricing, "
        "support, and troubleshooting. I can also summarize or clarify "
        "previous answers, give examples, or hand you off to a human agent."
    ),
    "session_reset": (
        "Got it - I've cleared the current context. What would you like to "
        "ask about ITCS?"
    ),
    "human_handoff": (
        "I understand you'd like to speak with a human. You can reach "
        "ITCS support directly:\n"
        f"{_CONTACT_BLURB}\n"
        "An agent will assist you shortly."
    ),
    "provide_contact_info": (
        "Here's how you can reach ITCS:\n"
        f"{_CONTACT_BLURB}\n\n"
        "Our offices:\n"
        f"{_OFFICE_BLURB}"
    ),
    "out_of_scope": (
        "That's outside what I can help with. I'm focused on ITCS's IT, "
        "cloud, and cybersecurity services - is there something in that "
        "area I can help you with?"
    ),
    "fallback": (
        "I'm not quite sure I understood that. Could you rephrase your "
        "question about ITCS services?"
    ),
}


class ChatAgent:
    """Azure OpenAI chat agent with Azure AI Search RAG integration."""

    def __init__(self) -> None:
        self.client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            timeout=60,
            max_retries=3,
        )
        self.deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
        self.search_endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
        self.search_key = os.environ["AZURE_SEARCH_KEY"]
        self.search_index = os.environ["AZURE_SEARCH_INDEX"]

    def _build_data_sources(self) -> list[dict]:
        return [
            {
                "type": "azure_search",
                "parameters": {
                    "endpoint": self.search_endpoint,
                    "index_name": self.search_index,
                    "authentication": {
                        "type": "api_key",
                        "key": self.search_key,
                    },
                    # Disable Azure's hardcoded "not available" fallback so
                    # the model follows our SYSTEM_PROMPT instructions instead.
                    "in_scope": False,
                    "strictness": 3,
                    "top_n_documents": 5,
                },
            }
        ]

    def stream_chitchat(self, messages: list[dict]) -> Generator[str, None, None]:
        """Handle greetings and small-talk without hitting the search index."""
        full_messages = [{"role": "system", "content": CHITCHAT_PROMPT}, *messages]

        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=full_messages,
            temperature=0.7,
            max_tokens=150,
            stream=True,
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def stream_meta(self, messages: list[dict]) -> Generator[str, None, None]:
        """Handle meta / conversational-control intents using only prior context.

        Used for: clarify, ask-for-examples, confirm-understanding, repeat,
        summarize, disambiguate, contradiction-resolution, sort, confirm-entities.
        """
        full_messages = [{"role": "system", "content": META_PROMPT}, *messages]

        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=full_messages,
            temperature=0.2,
            max_tokens=300,
            stream=True,
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    @staticmethod
    def stream_canned(intent_value: str) -> Generator[str, None, None]:
        """Yield a deterministic canned response for simple intents."""
        text = CANNED_RESPONSES.get(
            intent_value, CANNED_RESPONSES["fallback"]
        )
        yield text

    def stream(self, messages: list[dict]) -> Generator[str, None, None]:
        """Stream chat completions with Azure AI Search grounding.

        Args:
            messages: Conversation history as a list of
                      {"role": "...", "content": "..."} dicts.

        Yields:
            Incremental text chunks from the model response.
        """
        full_messages = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]

        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=full_messages,
            extra_body={"data_sources": self._build_data_sources()},
            # Small amount of randomness so "not found" replies are phrased
            # differently between turns, while grounded answers stay accurate.
            temperature=0.3,
            top_p=0.9,
            stream=True,
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                text = _CITATION_RE.sub("", chunk.choices[0].delta.content)
                if text:
                    yield text
