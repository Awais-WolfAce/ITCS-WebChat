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
    "- If the answer is not found in the retrieved documents, respond with: "
    '"I don\'t have that information in the available data."\n\n'
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
    "- If no relevant data is retrieved, say: "
    '"This information is not available in the current knowledge base."\n\n'
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


class ChatAgent:
    """Azure OpenAI chat agent with Azure AI Search RAG integration."""

    def __init__(self) -> None:
        self.client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
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
            temperature=0,
            stream=True,
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                text = _CITATION_RE.sub("", chunk.choices[0].delta.content)
                if text:
                    yield text
