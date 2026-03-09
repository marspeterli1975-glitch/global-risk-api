import os
import time
import json
import secrets
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import stripe
from openai import OpenAI


# =========================
# Env helpers
# =========================

def _get_env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    return v if v is not None else (default or "")


def _get_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _split_keys(v: str) -> List[str]:
    if not v:
        return []
    raw = v.replace("\n", ",").replace(" ", ",")
    return [x.strip() for x in raw.split(",") if x.strip()]


def _now() -> int:
    return int(time.time())


# =========================
# Configuration
# =========================

APP_NAME = _get_env("APP_NAME", "global-risk-api")
DEBUG = _get_bool("DEBUG", False)

STRIPE_SECRET_KEY = _get_env("STRIPE_SECRET_KEY", "")
OPENAI_API_KEY = _get_env("OPENAI_API_KEY", "")
LLM_MODEL = _get_env("LLM_MODEL", "gpt-4o-mini")

API_KEYS = _split_keys(_get_env("API_KEYS", ""))
API_KEYS += _split_keys(_get_env("APP_API_KEYS", ""))
API_KEYS = list(dict.fromkeys(API_KEYS))

TOKEN_SECRET = _get_env("TOKEN_SECRET", "change-me")
PAID_TOKEN_TTL_SECONDS = int(_get_env("PAID_TOKEN_TTL_SECONDS", "900") or "900")

SUCCESS_URL = _get_env(
    "SUCCESS_URL",
    "https://global-risk-api.onrender.com/success?session_id={CHECKOUT_SESSION_ID}"
)
CANCEL_URL = _get_env(
    "CANCEL_URL",
    "https://global-risk-api.onrender.com/cancel"
)

PAYMENT_MODE = _get_env("PAYMENT_MODE", "stripe").lower()

if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

openai_client: Optional[OpenAI] = None
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# FastAPI app
# =========================

app = FastAPI(title=APP_NAME, version="0.5.0-scrs-llm")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Schemas
# =========================

class AnalyzeRequest(BaseModel):
    route: str = Field(..., example="Shanghai -> Tokyo")
    context: str = Field(..., example="Lithium battery materials, CIF")
    horizon_days: int = Field(90, ge=1, le=365)
    language: str = Field("en", pattern="^(en|zh)$")


class AnalyzeResponse(BaseModel):
    status: str
    version: str
    mode: str
    result: Dict[str, Any]


# =========================
# Token storage
# =========================

_issued_tokens: Dict[str, Dict[str, Any]] = {}


def issue_token(session_id: str, email: str, amount_total: int, currency: str) -> str:
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


# =========================
# Auth helpers
# =========================

def _unauthorized(detail: str = "Unauthorized. Provide X-API-Key or Bearer token."):
    raise HTTPException(status_code=401, detail=detail)


def _extract_bearer(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.strip().split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    if len(parts) == 1 and parts[0]:
        return parts[0].strip()
    return None


def _verify_static_key(token_or_key: str) -> bool:
    return bool(token_or_key) and (token_or_key in API_KEYS)


def require_auth(
    authorization: Optional[str],
    x_api_key: Optional[str]
) -> Dict[str, Any]:
    if x_api_key and _verify_static_key(x_api_key):
        return {"mode": "api_key", "key": x_api_key}

    bearer = _extract_bearer(authorization)
    if bearer:
        if _verify_static_key(bearer):
            return {"mode": "api_key", "key": bearer}

        rec = _issued_tokens.get(bearer)
        if not rec:
            _unauthorized("Unauthorized. Invalid token.")
        if rec["exp"] < _now():
            _issued_tokens.pop(bearer, None)
            _unauthorized("Unauthorized. Token expired.")
        return {"mode": "paid_token", "token": bearer, **rec}

    _unauthorized()


# =========================
# Diagnostic routes
# =========================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "app": APP_NAME,
        "version": app.version,
        "has_openai_api_key": bool(OPENAI_API_KEY),
        "has_stripe_secret_key": bool(STRIPE_SECRET_KEY),
    }


@app.get("/__stripe")
def stripe_diag():
    return {
        "status": "ok",
        "has_stripe_secret_key": bool(STRIPE_SECRET_KEY),
        "payment_mode": PAYMENT_MODE,
        "success_url_template": SUCCESS_URL,
        "cancel_url": CANCEL_URL,
    }


@app.get("/cancel")
def cancel():
    return {"status": "ok", "message": "Payment cancelled. You can close this page."}


# =========================
# Stripe success callback
# =========================

@app.get("/success")
def success(session_id: str):
    if PAYMENT_MODE == "off":
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

    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=500, detail="STRIPE_SECRET_KEY missing on server.")

    if not session_id or not session_id.startswith("cs_"):
        raise HTTPException(status_code=400, detail="Invalid session_id (expected cs_...).")

    try:
        sess = stripe.checkout.Session.retrieve(session_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid session_id or Stripe error: {str(e)}")

    payment_status = getattr(sess, "payment_status", None) or sess.get("payment_status")
    status = getattr(sess, "status", None) or sess.get("status")

    if payment_status != "paid" and status != "complete":
        raise HTTPException(
            status_code=402,
            detail=f"Payment not completed. payment_status={payment_status}, status={status}"
        )

    customer_details = getattr(sess, "customer_details", None) or sess.get("customer_details") or {}
    email = customer_details.get("email") or ""

    amount_total = getattr(sess, "amount_total", None) or sess.get("amount_total") or 0
    currency = getattr(sess, "currency", None) or sess.get("currency") or "usd"

    token = issue_token(
        session_id=session_id,
        email=email,
        amount_total=amount_total,
        currency=currency
    )

    return {
        "status": "ok",
        "version": app.version,
        "message": "Payment verified. Use this token to call /analyze with Authorization: Bearer <token>",
        "token": token,
        "session_id": session_id,
        "email": email,
        "amount_total": amount_total,
        "currency": currency,
        "ttl_seconds": PAID_TOKEN_TTL_SECONDS,
    }


# =========================
# GPT analysis
# =========================

def run_analysis_with_llm(route: str, context: str, horizon_days: int, language: str) -> Dict[str, Any]:
    if not openai_client:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY missing on server.")

    system_prompt = """
You are a supply-chain risk analyst.

Return STRICT JSON only.
Do not use markdown.
Do not include any text outside JSON.

Required JSON structure:
{
  "summary": "string",
  "horizon_days": 90,
  "key_risks": [
    {
      "name": "string",
      "likelihood_0_1": 0.0,
      "impact_0_10": 0,
      "notes": "string"
    }
  ]
}

Rules:
- Give 3 to 5 key risks.
- likelihood_0_1 must be a decimal between 0 and 1.
- impact_0_10 must be an integer between 1 and 10.
- Keep notes concise and practical.
- Use the same language as requested by user: en or zh.
- Focus on route risk, regulatory risk, geopolitical risk, logistics risk, and supply disruption risk.
"""

    user_prompt = f"""
Analyze this supply-chain route risk case.

route: {route}
context: {context}
horizon_days: {horizon_days}
language: {language}

Return only valid JSON.
"""

    try:
        resp = openai_client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        content = resp.choices[0].message.content or ""
        parsed = json.loads(content)

        if "summary" not in parsed or "key_risks" not in parsed:
            raise ValueError("Missing required fields in LLM response.")

        parsed["horizon_days"] = horizon_days
        return parsed

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="LLM returned invalid JSON.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM analysis failed: {str(e)}")


# =========================
# Analyze route
# =========================

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(
    payload: AnalyzeRequest,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    auth_info = require_auth(authorization, x_api_key)
    mode = auth_info["mode"]

    result = run_analysis_with_llm(
        route=payload.route,
        context=payload.context,
        horizon_days=payload.horizon_days,
        language=payload.language,
    )

    return {
        "status": "ok",
        "version": app.version,
        "mode": mode,
        "result": result,
    }
