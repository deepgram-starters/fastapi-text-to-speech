"""
FastAPI Text-to-Speech Starter

Simple TTS API endpoint returning binary audio data.
"""

import os
from typing import Optional
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from deepgram import DeepgramClient
from dotenv import load_dotenv
import toml

load_dotenv(override=False)

CONFIG = {
    "port": int(os.environ.get("PORT", 8081)),
    "host": os.environ.get("HOST", "0.0.0.0"),
    "frontend_port": int(os.environ.get("FRONTEND_PORT", 8080)),
}

def load_api_key():
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        raise ValueError("DEEPGRAM_API_KEY required")
    return api_key

api_key = load_api_key()
deepgram = DeepgramClient(api_key=api_key)

app = FastAPI(title="Deepgram TTS API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        f"http://localhost:{CONFIG['frontend_port']}",
        f"http://127.0.0.1:{CONFIG['frontend_port']}",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TTSRequest(BaseModel):
    text: str

@app.post("/tts/synthesize")
async def synthesize(
    body: TTSRequest,
    model: str = "aura-asteria-en",
    x_request_id: Optional[str] = Header(None)
):
    """POST /tts/synthesize - Convert text to speech"""
    try:
        if not body.text or not body.text.strip():
            raise HTTPException(status_code=400, detail="Text required")

        audio_generator = deepgram.speak.v1.audio.generate(
            text=body.text,
            model=model
        )

        audio_data = b"".join(audio_generator)
        
        headers = {"Content-Type": "audio/mpeg"}
        if x_request_id:
            headers["X-Request-Id"] = x_request_id
        
        return Response(content=audio_data, media_type="audio/mpeg", headers=headers)

    except Exception as e:
        print(f"TTS Error: {e}")
        raise HTTPException(status_code=500, detail="TTS synthesis failed")

@app.get("/api/metadata")
async def get_metadata():
    try:
        with open('deepgram.toml', 'r') as f:
            config = toml.load(f)
        return JSONResponse(content=config.get('meta', {}))
    except:
        raise HTTPException(status_code=500, detail="Metadata read failed")

if __name__ == "__main__":
    import uvicorn
    print(f"\n🚀 FastAPI TTS Server: http://localhost:{CONFIG['port']}\n")
    uvicorn.run(app, host=CONFIG["host"], port=CONFIG["port"])
