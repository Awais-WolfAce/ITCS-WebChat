"""Speech-to-Text module using Azure Speech REST API.

Transcribes WAV audio by trying multiple candidate languages in parallel
and returning the result with the highest confidence.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

_CANDIDATE_LANGUAGES = ["en-US", "ur-PK"]


@dataclass(frozen=True)
class RecognitionResult:
    text: str
    language: str
    confidence: float


class SpeechToText:
    """Azure Speech-to-Text client (REST API, no native SDK required)."""

    def __init__(self) -> None:
        self.key = os.environ["AZURE_SPEECH_KEY"]
        self.region = os.environ["AZURE_SPEECH_REGION"]
        self._base_url = (
            f"https://{self.region}.stt.speech.microsoft.com"
            "/speech/recognition/conversation/cognitiveservices/v1"
        )
        self._client = httpx.Client(timeout=20)

    def transcribe(self, audio_bytes: bytes) -> RecognitionResult:
        """Transcribe WAV audio, auto-detecting language from candidates.

        Runs recognition for each candidate language in parallel and
        returns the result with the highest confidence score.
        """
        with ThreadPoolExecutor(max_workers=len(_CANDIDATE_LANGUAGES)) as pool:
            futures = {
                pool.submit(self._recognize, audio_bytes, lang): lang
                for lang in _CANDIDATE_LANGUAGES
            }
            results: list[RecognitionResult] = []
            for future in futures:
                lang = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception:
                    logger.warning("STT recognition failed for %s", lang, exc_info=True)

        if not results:
            logger.info("STT: no speech recognised in any language")
            return RecognitionResult(text="", language="en", confidence=0.0)

        best = max(results, key=lambda r: r.confidence)
        logger.info(
            "STT best: lang=%s conf=%.2f text='%s'",
            best.language, best.confidence, best.text[:80],
        )
        return best

    def _recognize(
        self, audio_bytes: bytes, language: str
    ) -> RecognitionResult | None:
        url = f"{self._base_url}?language={language}&format=detailed"
        headers = {
            "Ocp-Apim-Subscription-Key": self.key,
            "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000",
        }
        resp = self._client.post(url, headers=headers, content=audio_bytes)
        resp.raise_for_status()
        data = resp.json()

        if data.get("RecognitionStatus") != "Success":
            return None

        nbest = data.get("NBest", [])
        if not nbest:
            return None

        top = nbest[0]
        return RecognitionResult(
            text=top["Display"],
            language=language.split("-")[0],
            confidence=top["Confidence"],
        )
