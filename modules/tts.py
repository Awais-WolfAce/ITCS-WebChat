"""Text-to-Speech module using Azure Speech REST API.

Converts text to MP3 audio using Azure neural voices, with
language-aware voice selection for English and Urdu.
"""

from __future__ import annotations

import logging
import os
from xml.sax.saxutils import escape

import httpx

logger = logging.getLogger(__name__)

_VOICE_MAP: dict[str, tuple[str, str]] = {
    "en": ("en-US", "en-US-JennyNeural"),
    "ur": ("ur-PK", "ur-PK-UzmaNeural"),
}

_DEFAULT_VOICE = ("en-US", "en-US-JennyNeural")


class TextToSpeech:
    """Azure Text-to-Speech client (REST API, no native SDK required)."""

    def __init__(self) -> None:
        self.key = os.environ["AZURE_SPEECH_KEY"]
        self.region = os.environ["AZURE_SPEECH_REGION"]
        self._url = (
            f"https://{self.region}.tts.speech.microsoft.com"
            "/cognitiveservices/v1"
        )
        self._client = httpx.Client(timeout=20)

    def synthesize(self, text: str, lang: str = "en") -> bytes:
        """Convert *text* to MP3 audio bytes using the appropriate voice.

        Args:
            text: The text to speak.
            lang: Language code ("en" or "ur").

        Returns:
            Raw MP3 audio bytes.
        """
        locale, voice = _VOICE_MAP.get(lang, _DEFAULT_VOICE)

        ssml = (
            '<speak version="1.0" '
            'xmlns="http://www.w3.org/2001/10/synthesis" '
            f'xml:lang="{locale}">'
            f'<voice name="{voice}">{escape(text)}</voice>'
            "</speak>"
        )

        headers = {
            "Ocp-Apim-Subscription-Key": self.key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "audio-16khz-128kbitrate-mono-mp3",
        }

        resp = self._client.post(self._url, headers=headers, content=ssml)
        resp.raise_for_status()

        logger.info(
            "TTS: %d bytes (lang=%s, voice=%s)", len(resp.content), lang, voice
        )
        return resp.content
