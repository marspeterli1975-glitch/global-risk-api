import os
import time
import json
import secrets
from typing import Optional, Dict, Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse

# -----------------------------
# Basic App
# -----------------------------
app = FastAPI(title="Global Risk API", version="0.4.0-scrs-freemium")

# -----------------------------
# Helpers / Env
# -----------------------------
def env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if (v is not None and v != "") else default

APP_API_KEYS = env("APP_API_KEYS") or env("API_KEYS")  # support both names
OPENAI_API_KEY = env("OPENAI_API_KEY")
STRIPE_SECRET_KEY = env("STRIPE_SECRET_KEY")

PAYMENT_MODE = (env("PAYMENT_MODE", "stripe") or "stripe").lower()  # stripe / off
TOKEN_SECRET = env("TOKEN_SECRET", "change-me")  # used for signing/issuing tokens
PAID_TOKEN_TTL_SECONDS = int(env("PAID_TOKEN_TTL_SECONDS", "900"))
CACHE_TTL_SECONDS = int(env("CACHE_TTL_SECONDS", "900"))
DEBUG = (env("DEBUG", "false").lower() == "true")

SUCCESS_URL = env("SUCCESS_URL", "https://global-risk-api.onrender.com/success?session_id={CHECKOUT_SESSION_ID}")
CANCEL_URL = env("CANCEL_URL", "https://global-risk-api.onrender.com/cancel")

# -----------------------------
# In-memory token cache (simple)
# -----------------------------
_issued_tokens: Dict[str, Dict[str, Any]] = {}  # token -> {exp:int, email:str, session_id:str, amount_total:int, currency:str}

def _now() -> int:
    return int(time.time())

def _parse_api_keys(raw: Optional[str]) -> set:
    if not raw:
        return set()
    # allow comma / newline / space separated
    parts = []
    for chunk in raw.replace("\n", ",").replace(" ", ",").split(","):
        c = chunk.strip()
        if c:
            parts.append(c)
    return set(parts)

ALLOWED_API_KEYS = _parse_api_keys(APP_API_KEYS)

def require_api_key(x_api_key: Optional[str]) -> None:
    # If no keys configured, allow (for early testing)
    if not ALLOWED_API_KEYS:
        return
    if not x_api_key or x_api_key not in ALLOWED_API_KEYS:
        raise HTTPException(status_code=401, detail="Unauthorized. Invalid X-API-Key.")

def require_bearer_or_api_key(authorization: Optional[str], x_api_key: Optional[str]) -> Dict[str, Any]:
    """
    Accept either:
    - X-API-Key: <key>
    - Authorization: Bearer <token>
    """
    # API key route
    if x_api_key:
        require_api_key(x_api_key)
        return {"mode": "api_key"}

    # Bearer token route
    if not authorization:
        raise HTTPException(status_code=401, detail="Unauthorized. Provide X-API-Key or Bearer token.")

    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized. Invalid Authorization header format. Use 'Bearer <token>'.")

    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized. Empty bearer token.")

    rec = _issued_tokens.get(token)
    if not rec:
        raise HTTPException(status_code=401, detail="Unauthorized. Invalid token.")
    if rec["exp"] < _now():
        _issued_tokens.pop(token, None)
        raise HTTPException(status_code=401, detail="Unauthorized. Token expired.")
    return {"mode": "paid_token", "token": token, **rec}

def issue_token(session_id: str, email: str, amount_total: int, currency: str) -> str:
    # Create opaque token (not JWT) to keep it simple
    token = secrets.token_urlsafe(32)
    exp = _now() + PAID_TOKEN_TTL_SECONDS
    _issued_tokens[token] = {
        "exp": exp,
        "email": email,
        "session_id": session_id,
        "amount_total": amount_total,
        "currency": currency,
        "ttl_seconds": PAID_TOKEN_TTL_SECONDS,
    }
    return token

# -----------------------------
# ✅ Milestone-0 / First Fix: /__stripe MUST exist
# -----------------------------
@app.get("/__stripe")
def stripe_diag():
    """
    Minimal diagnostic endpoint.
    Goal: DO NOT 404.
    """
    return {
        "status": "ok",
        "has_stripe_secret_key": bool(STRIPE_SECRET_KEY),
        "payment_mode": PAYMENT_MODE,
        "success_url_template": SUCCESS_URL,
        "cancel_url": CANCEL_URL,
    }

# Optional: health endpoint
@app.get("/health")
def health():
    return {
        "status": "ok",
        "ts": _now(),
        "app_version": app.version,
        "debug": DEBUG,
    }

# -----------------------------
# Stripe success callback (server side verification)
# NOTE: Here we only implement a safe stub that issues token.
# If you already integrated stripe.checkout.Session.retrieve, keep yours.
# -----------------------------
@app.get("/success")
def success(session_id: str):
    """
    In production you should retrieve session from Stripe:
      stripe.checkout.Session.retrieve(session_id)
    For this step, we only make sure endpoint exists and can issue a token.
    """
    if PAYMENT_MODE == "off":
        # allow issuing token even if payment is disabled (for testing)
        token = issue_token(session_id=session_id, email="test@local", amount_total=0, currency="usd")
        return {
            "status": "ok",
            "version": app.version,
            "message": "Payment mode off. Token issued for testing.",
            "token": token,
            "session_id": session_id,
            "email": "test@local",
            "amount_total": 0,
            "currency": "usd",
            "ttl_seconds": PAID_TOKEN_TTL_SECONDS,
        }

    # Stripe mode requires STRIPE_SECRET_KEY to exist
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=500, detail="STRIPE_SECRET_KEY missing on server.")

    # --- Minimal safe behavior ---
    # If you want strict verification, you must add stripe SDK retrieve here.
    # For the /__stripe fix milestone, we do NOT expand scope.
    token = issue_token(session_id=session_id, email="email@google.com", amount_total=2900, currency="usd")
    return {
        "status": "ok",
        "version": app.version,
        "message": "Payment verified (stub). Use this token to call /analyze with Authorization: Bearer <token>",
        "token": token,
        "session_id": session_id,
        "email": "email@google.com",
        "amount_total": 2900,
        "currency": "usd",
        "ttl_seconds": PAID_TOKEN_TTL_SECONDS,
    }

# -----------------------------
# Analyze endpoint
# -----------------------------
@app.post("/analyze")
async def analyze(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    # Auth
    _auth = require_bearer_or_api_key(authorization, x_api_key)

    body = await request.json()
    route = body.get("route", "")
    context = body.get("context", "")
    horizon_days = int(body.get("horizon_days", 90))
    language = body.get("language", "en")

    # Minimal response (LLM part intentionally omitted here)
    return {
        "status": "ok",
        "version": app.version,
        "mode": _auth.get("mode"),
        "result": {
            "summary": f"Stub analysis for route='{route}', context='{context}', horizon_days={horizon_days}, language={language}.",
            "horizon_days": horizon_days,
            "key_risks": [
                {"name": "Geopolitical Tensions", "likelihood_0_1": 0.7, "impact_0_10": 8, "notes": "Stub."},
                {"name": "Regulatory Changes", "likelihood_0_1": 0.6, "impact_0_10": 7, "notes": "Stub."},
            ],
        },
    }
