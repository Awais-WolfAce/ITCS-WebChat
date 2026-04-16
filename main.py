from __future__ import annotations

import json
import logging

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from modules.chat import ChatAgent
from modules.intent import is_chitchat
from modules.stt import SpeechToText
from modules.translation import Translator
from modules.tts import TextToSpeech

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ITCS Chat Agent")

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
    return FileResponse("static/index.html")


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

    chitchat = is_chitchat(english_text)
    logger.info("Intent: %s | lang: %s", "chitchat" if chitchat else "knowledge", source_lang)

    def event_stream():
        try:
            stream_fn = agent.stream_chitchat if chitchat else agent.stream
            full_response = ""
            for chunk in stream_fn(english_messages):
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
