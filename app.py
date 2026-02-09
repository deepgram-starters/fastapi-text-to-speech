"""
FastAPI Text-to-Speech Starter

Simple TTS API endpoint returning binary audio data.

Key Features:
- Single API endpoint: POST /api/text-to-speech
- JWT session auth with page nonce (production only)
- Returns binary audio data
"""

import os
import secrets
import time

import jwt
from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.responses import Response, JSONResponse, HTMLResponse
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
}

# ============================================================================
# SESSION AUTH - JWT tokens with page nonce for production security
# ============================================================================

SESSION_SECRET = os.environ.get("SESSION_SECRET") or secrets.token_hex(32)
REQUIRE_NONCE = bool(os.environ.get("SESSION_SECRET"))

# In-memory nonce store: nonce -> expiry timestamp
session_nonces = {}
NONCE_TTL = 5 * 60  # 5 minutes
JWT_EXPIRY = 3600  # 1 hour


def generate_nonce():
    """Generates a single-use nonce and stores it with an expiry."""
    nonce = secrets.token_hex(16)
    session_nonces[nonce] = time.time() + NONCE_TTL
    return nonce


def consume_nonce(nonce):
    """Validates and consumes a nonce (single-use). Returns True if valid."""
    expiry = session_nonces.pop(nonce, None)
    if expiry is None:
        return False
    return time.time() < expiry


def cleanup_nonces():
    """Remove expired nonces."""
    now = time.time()
    expired = [k for k, v in session_nonces.items() if now >= v]
    for k in expired:
        del session_nonces[k]


# Read frontend/dist/index.html template for nonce injection
_index_html_template = None
try:
    with open(os.path.join(os.path.dirname(__file__), "frontend", "dist", "index.html")) as f:
        _index_html_template = f.read()
except FileNotFoundError:
    pass  # No built frontend (dev mode)


def require_session(authorization: str = Header(None)):
    """FastAPI dependency for JWT session validation."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "type": "AuthenticationError",
                    "code": "MISSING_TOKEN",
                    "message": "Authorization header with Bearer token is required",
                }
            }
        )
    token = authorization[7:]
    try:
        jwt.decode(token, SESSION_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "type": "AuthenticationError",
                    "code": "INVALID_TOKEN",
                    "message": "Session expired, please refresh the page",
                }
            }
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "type": "AuthenticationError",
                    "code": "INVALID_TOKEN",
                    "message": "Invalid session token",
                }
            }
        )


# ============================================================================
# API KEY LOADING
# ============================================================================

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
    allow_origins=["*"],
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

# ============================================================================
# SESSION ROUTES - Auth endpoints (unprotected)
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve index.html with injected session nonce (production only)."""
    if not _index_html_template:
        raise HTTPException(status_code=404, detail="Frontend not built. Run make build first.")
    cleanup_nonces()
    nonce = generate_nonce()
    html = _index_html_template.replace(
        "</head>",
        f'<meta name="session-nonce" content="{nonce}">\n</head>'
    )
    return HTMLResponse(content=html)


@app.get("/api/session")
async def get_session(x_session_nonce: str = Header(None)):
    """Issues a JWT. In production, requires valid nonce via X-Session-Nonce header."""
    if REQUIRE_NONCE:
        if not x_session_nonce or not consume_nonce(x_session_nonce):
            raise HTTPException(
                status_code=403,
                detail={
                    "error": {
                        "type": "AuthenticationError",
                        "code": "INVALID_NONCE",
                        "message": "Valid session nonce required. Please refresh the page.",
                    }
                }
            )
    token = jwt.encode(
        {"iat": int(time.time()), "exp": int(time.time()) + JWT_EXPIRY},
        SESSION_SECRET,
        algorithm="HS256",
    )
    return JSONResponse(content={"token": token})


# ============================================================================
# API ROUTES
# ============================================================================

class TTSRequest(BaseModel):
    text: str

@app.post("/api/text-to-speech")
async def synthesize(
    body: TTSRequest,
    model: str = "aura-asteria-en",
    _auth=Depends(require_session)
):
    """POST /api/text-to-speech - Convert text to speech"""
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
    nonce_status = " (nonce required)" if REQUIRE_NONCE else ""
    print(f"\n🚀 FastAPI TTS Server starting on {CONFIG['host']}:{CONFIG['port']}")
    print(f"   GET  /api/session{nonce_status}")
    print(f"   POST /api/text-to-speech (auth required)")
    print(f"   GET  /api/metadata\n")
    uvicorn.run(app, host=CONFIG["host"], port=CONFIG["port"])
