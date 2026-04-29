"""Azure Translator module for language detection and translation.

Uses the Azure Cognitive Services Translator REST API (v3.0).
Requires AZURE_TRANSLATOR_ENDPOINT, AZURE_TRANSLATOR_KEY, and
AZURE_TRANSLATOR_REGION to be set in the environment.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

_API_VERSION = "3.0"

_SUPPORTED_LANGUAGES = {"en", "ur"}

# Arabic-script ranges (covers Urdu, including Arabic Presentation Forms).
_ARABIC_SCRIPT_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)
_LATIN_LETTER_RE = re.compile(r"[A-Za-z]")
# Confidence below this is treated as unreliable for short/ambiguous input.
_MIN_DETECT_CONFIDENCE = 0.5


def _script_hint(text: str) -> str | None:
    """Return 'en' or 'ur' based purely on script analysis, else None.

    Strong signal: Urdu is written in Arabic script, English in Latin script.
    Returns None when the text mixes scripts or contains no letters.
    """
    has_arabic = bool(_ARABIC_SCRIPT_RE.search(text))
    has_latin = bool(_LATIN_LETTER_RE.search(text))
    if has_arabic and not has_latin:
        return "ur"
    if has_latin and not has_arabic:
        return "en"
    return None


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
        self._client = httpx.Client(timeout=30)

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
        # Fast path: if the script is unambiguously Latin, it's English. Skip
        # the translate call entirely — Azure's detector misclassifies short
        # English phrases (e.g. "introduce yourself") into other languages.
        script = _script_hint(text)
        if script == "en":
            logger.info("Script hint: en (skipping translation)")
            return text, "en"

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

        if script == "ur":
            # Arabic-script text → Urdu, regardless of detector noise.
            resolved = "ur"
        elif base_lang in _SUPPORTED_LANGUAGES and confidence >= _MIN_DETECT_CONFIDENCE:
            resolved = base_lang
        else:
            # Low-confidence or unsupported language → default to English so
            # we don't surprise the user with an Urdu response to English input.
            resolved = "en"

        logger.info(
            "Detected: %s → resolved: %s (confidence: %.2f, script: %s)",
            detected_lang, resolved, confidence, script,
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
