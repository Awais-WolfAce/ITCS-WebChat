from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel

from modules.chat import ChatAgent
from modules.intent import (
    CHITCHAT_INTENTS,
    META_INTENTS,
    Intent,
    classify_intent,
)
from modules.session import SessionManager
from modules.stt import SpeechToText
from modules.translation import Translator
from modules.tts import TextToSpeech

# Intents that get a deterministic canned reply instead of calling the LLM.
CANNED_INTENTS: frozenset[Intent] = frozenset(
    {
        Intent.APPRECIATION,
        Intent.APOLOGY,
        Intent.GOODBYE,
        Intent.BOT_IDENTITY,
        Intent.BOT_CAPABILITY,
        Intent.COMPLAINT,
        Intent.FEEDBACK,
        Intent.SESSION_RESET,
        Intent.HUMAN_HANDOFF,
        Intent.PROVIDE_CONTACT_INFO,
        Intent.OUT_OF_SCOPE,
        Intent.FALLBACK,
    }
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    sessions.start_sweeper()
    try:
        yield
    finally:
        await sessions.stop_sweeper()
        # Flush any still-active sessions so their final email goes out
        # before the process exits.
        sessions.end_all(reason="shutdown")


app = FastAPI(title="ITCS Chat Agent", lifespan=lifespan)


class NoCacheStaticMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static"):
            response.headers["Cache-Control"] = "no-store"
        return response


app.add_middleware(NoCacheStaticMiddleware)

agent = ChatAgent()
translator = Translator()
stt = SpeechToText()
tts = TextToSpeech()
sessions = SessionManager()


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    user_id: str | None = None


class TTSRequest(BaseModel):
    text: str
    lang: str = "en"


class SessionEndRequest(BaseModel):
    user_id: str


@app.get("/")
async def index():
    return FileResponse(
        "static/index.html",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/stt")
async def speech_to_text(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    result = stt.transcribe(audio_bytes)
    return {"text": result.text, "lang": result.language}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    messages = [m.model_dump() for m in request.messages]

    latest_user_text = messages[-1]["content"]
    english_text, source_lang = translator.translate_to_english(latest_user_text)

    english_messages = messages.copy()
    english_messages[-1] = {"role": "user", "content": english_text}

    intent = classify_intent(english_text)
    # Resolve / mint the conversation id BEFORE streaming so the client
    # can pick it up on the very first turn (or whenever its stored id
    # was already swept due to idle timeout).
    user_id = sessions.get_or_create_user_id(request.user_id)
    logger.info(
        "Intent: %s | lang: %s | userId: %s",
        intent.value,
        source_lang,
        user_id,
    )

    def _select_stream():
        if intent in CANNED_INTENTS:
            return agent.stream_canned(intent.value)
        if intent in CHITCHAT_INTENTS:
            return agent.stream_chitchat(english_messages)
        if intent in META_INTENTS:
            return agent.stream_meta(english_messages)
        # Knowledge intents (ask_*) and anything else -> RAG over the search index.
        return agent.stream(english_messages)

    def event_stream():
        full_response = ""
        try:
            for chunk in _select_stream():
                full_response += chunk

            if source_lang != "en":
                translated = translator.translate_from_english(
                    full_response, source_lang
                )
            else:
                translated = full_response

            payload = {
                "content": translated,
                "lang": source_lang,
                "user_id": user_id,
            }
            yield f"data: {json.dumps(payload)}\n\n"
        except Exception as exc:
            logger.exception("Chat stream error")
            yield f"data: {json.dumps({'error': str(exc), 'user_id': user_id})}\n\n"
            full_response = ""  # don't log a partial/failed reply
        finally:
            if full_response:
                # Record the turn (and fire the per-row Power Automate
                # POST) only after the final, translated reply has been
                # produced. Failures are swallowed inside the logger so
                # the chat stream cannot be broken by a flow outage.
                try:
                    sessions.record_turn(
                        user_id=user_id,
                        user_input=latest_user_text,
                        bot_output=translated,
                        language=source_lang,
                    )
                except Exception:
                    logger.exception("Failed to record chat turn")
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/chat/end")
async def chat_end(request: Request):
    """Tab-close / explicit end notification from the browser.

    Accepts both a JSON body (regular fetch) and an
    ``application/x-www-form-urlencoded`` body, because
    ``navigator.sendBeacon`` defaults to the latter content type and we
    want to support both delivery paths.
    """

    user_id: str | None = None
    content_type = (request.headers.get("content-type") or "").lower()
    try:
        if "application/json" in content_type:
            body = await request.json()
            user_id = (body or {}).get("user_id")
        else:
            # sendBeacon with a Blob of type text/plain or form-encoded
            # arrives here. Try form first, then raw text as JSON.
            raw = await request.body()
            if not raw:
                user_id = None
            else:
                text = raw.decode("utf-8", errors="ignore").strip()
                try:
                    user_id = (json.loads(text) or {}).get("user_id")
                except json.JSONDecodeError:
                    # Fallback: form-encoded "user_id=..."
                    from urllib.parse import parse_qs

                    parsed = parse_qs(text)
                    vals = parsed.get("user_id") or []
                    user_id = vals[0] if vals else None
    except Exception:
        logger.exception("Failed to parse /api/chat/end body")

    if user_id:
        sessions.end_session(user_id, reason="client-end")
    return Response(status_code=204)


@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    audio_bytes = tts.synthesize(request.text, request.lang)
    return Response(content=audio_bytes, media_type="audio/mpeg")


app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
