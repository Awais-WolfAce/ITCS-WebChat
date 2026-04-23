from __future__ import annotations

import json
import logging

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
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

app = FastAPI(title="ITCS Chat Agent")


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


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


class TTSRequest(BaseModel):
    text: str
    lang: str = "en"


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
    logger.info("Intent: %s | lang: %s", intent.value, source_lang)

    def _select_stream():
        if intent in CANNED_INTENTS:
            return agent.stream_canned(intent.value)
        if intent in CHITCHAT_INTENTS:
            return agent.stream_chitchat(english_messages)
        if intent in META_INTENTS:
            return agent.stream_meta(english_messages)
        # Knowledge intents (ask_*) and anything else → RAG over the search index.
        return agent.stream(english_messages)

    def event_stream():
        try:
            full_response = ""
            for chunk in _select_stream():
                full_response += chunk

            if source_lang != "en":
                translated = translator.translate_from_english(
                    full_response, source_lang
                )
            else:
                translated = full_response

            yield f"data: {json.dumps({'content': translated, 'lang': source_lang})}\n\n"
        except Exception as exc:
            logger.exception("Chat stream error")
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    audio_bytes = tts.synthesize(request.text, request.lang)
    return Response(content=audio_bytes, media_type="audio/mpeg")


app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
