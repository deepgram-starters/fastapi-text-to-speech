"""
FastAPI Text-to-Speech Starter

Simple TTS API endpoint returning binary audio data.
"""

import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, JSONResponse
from fastapi.exceptions import RequestValidationError
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

# Exception handlers to ensure contract-compliant error responses
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Convert HTTPException to contract format"""
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": "SynthesisError",
                "code": "SYNTHESIS_FAILED",
                "message": str(exc.detail)
            }
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "type": "ValidationError",
                "code": "INVALID_INPUT",
                "message": "Invalid request"
            }
        }
    )

class TTSRequest(BaseModel):
    text: str

@app.post("/tts/synthesize")
async def synthesize(
    body: TTSRequest,
    model: str = "aura-asteria-en"
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
        
        return Response(content=audio_data, media_type="audio/mpeg")

    except Exception as e:
        print(f"TTS Error: {e}")
        error_msg = str(e).lower()

        # Check if it's a Deepgram text length error
        if any(keyword in error_msg for keyword in ['too long', 'length', 'limit', 'exceed']):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "type": "ValidationError",
                        "code": "TEXT_TOO_LONG",
                        "message": "Text exceeds maximum allowed length"
                    }
                }
            )

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
