from __future__ import annotations

import os
import re
from collections.abc import Generator

from openai import AzureOpenAI

_CITATION_RE = re.compile(r"\[doc\d+\]")

SYSTEM_PROMPT = (
    "You are a website support chat agent.\n\n"
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
    "- Always steer the conversation toward how you can help."
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
    "- Professional, neutral, and helpful tone."
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
        "I understand you'd like to speak with a human. Please reach ITCS "
        "support using the contact details on the Contact page, and an "
        "agent will assist you shortly."
    ),
    "provide_contact_info": (
        "You can reach ITCS through the Contact page on this website, "
        "which lists the current phone number, email address, and office "
        "locations."
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
