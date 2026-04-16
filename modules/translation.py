"""Azure Translator module for language detection and translation.

Uses the Azure Cognitive Services Translator REST API (v3.0).
Requires AZURE_TRANSLATOR_ENDPOINT, AZURE_TRANSLATOR_KEY, and
AZURE_TRANSLATOR_REGION to be set in the environment.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

_API_VERSION = "3.0"

_SUPPORTED_LANGUAGES = {"en", "ur"}


@dataclass(frozen=True)
class DetectionResult:
    language: str
    confidence: float


class Translator:
    """Reusable Azure Translator client."""

    def __init__(self) -> None:
        self.endpoint = os.environ["AZURE_TRANSLATOR_ENDPOINT"].rstrip("/")
        self.key = os.environ["AZURE_TRANSLATOR_KEY"]
        self.region = os.environ["AZURE_TRANSLATOR_REGION"]
        self._headers = {
            "Ocp-Apim-Subscription-Key": self.key,
            "Ocp-Apim-Subscription-Region": self.region,
            "Content-Type": "application/json",
        }
        self._client = httpx.Client(timeout=10)

    # ── public API ───────────────────────────────────────────

    def detect_language(self, text: str) -> DetectionResult:
        """Detect the language of *text* and return language code + confidence."""
        url = f"{self.endpoint}/detect?api-version={_API_VERSION}"
        body = [{"Text": text}]
        resp = self._client.post(url, headers=self._headers, json=body)
        resp.raise_for_status()
        data = resp.json()[0]
        return DetectionResult(
            language=data["language"],
            confidence=data["score"],
        )

    def translate_to_english(self, text: str) -> tuple[str, str]:
        """Translate *text* to English, auto-detecting the source language.

        Returns:
            (translated_text, detected_source_language)
        """
        url = (
            f"{self.endpoint}/translate"
            f"?api-version={_API_VERSION}&to=en"
        )
        body = [{"Text": text}]
        resp = self._client.post(url, headers=self._headers, json=body)
        resp.raise_for_status()
        data = resp.json()[0]
        detected_lang = data["detectedLanguage"]["language"]
        confidence = data["detectedLanguage"]["score"]
        translated = data["translations"][0]["text"]

        base_lang = detected_lang.split("-")[0]
        resolved = base_lang if base_lang in _SUPPORTED_LANGUAGES else "ur"

        logger.info(
            "Detected: %s → resolved: %s (confidence: %.2f)",
            detected_lang, resolved, confidence,
        )

        return translated, resolved

    def translate_from_english(self, text: str, target_lang: str) -> str:
        """Translate English *text* into *target_lang*.

        Args:
            text: The English source text.
            target_lang: BCP-47 language code (e.g. "ur", "fr", "zh-Hans").

        Returns:
            Translated string in the target language.
        """
        if target_lang == "en":
            return text

        logger.info("Translating response to: %s", target_lang)

        url = (
            f"{self.endpoint}/translate"
            f"?api-version={_API_VERSION}&from=en&to={target_lang}"
        )
        body = [{"Text": text}]
        resp = self._client.post(url, headers=self._headers, json=body)
        resp.raise_for_status()
        return resp.json()[0]["translations"][0]["text"]
