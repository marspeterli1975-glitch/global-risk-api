import os
import time
import json
import hmac
import hashlib
import secrets
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# =====================================================
# Basic App
# =====================================================
APP_VERSION = "0.5.1-scrs-schema"
APP_SECRET = os.getenv("APP_SECRET", "change-this-in-render-env")

app = FastAPI(title="SCRS API", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# In-memory token store
# NOTE:
# This is acceptable for MVP testing.
# On redeploy/restart, tokens will reset.
# =====================================================
ISSUED_TOKENS: Dict[str, Dict[str, Any]] = {}


# =====================================================
# Helpers
# =====================================================
def create_token(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    sig = hmac.new(APP_SECRET.encode("utf-8"), raw, hashlib.sha256).hexdigest()
    token = f"{secrets.token_urlsafe(24)}.{sig[:16]}"
    ISSUED_TOKENS[token] = {
        "payload": payload,
        "created_at": int(time.time())
    }
    return token


def verify_bearer_token(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")

    token = authorization.replace("Bearer ", "", 1).strip()

    if token not in ISSUED_TOKENS:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return token


def score_to_level(score: float) -> str:
    if score < 20:
        return "Low"
    elif score < 40:
        return "Moderate"
    elif score < 60:
        return "Elevated"
    elif score < 80:
        return "High"
    return "Critical"


# =====================================================
# Request Models
# =====================================================
class SCRSRequest(BaseModel):
    origin_country: str = Field(..., description="Country where shipment originates")
    destination_country: str = Field(..., description="Destination country")

    supplier_years: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="Years supplier has been operating"
    )

    supplier_financial_level: Optional[str] = Field(
        default="unknown",
        description="Supplier financial stability: strong / adequate / weak / distressed / unknown"
    )

    supplier_dependency_pct: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="Buyer dependency percentage on supplier"
    )

    transport_mode: str = Field(
        ...,
        description="Transport mode: sea / air / rail / road / multimodal"
    )

    route_region: str = Field(
        ...,
        description="Shipping route region"
    )

    product_type: str = Field(
        ...,
        description="Product category"
    )

    payment_terms: str = Field(
        ...,
        description="Payment terms"
    )


# =====================================================
# Health
# =====================================================
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "service": "scrs-api"
    }


# =====================================================
# Success
# NOTE:
# This endpoint issues a token for MVP flow.
# It does NOT verify Stripe server-to-server here.
# It preserves your existing "success -> token" usage pattern.
# =====================================================
@app.get("/success", response_class=HTMLResponse)
async def success(session_id: Optional[str] = None):
    issued_at = int(time.time())

    payload = {
        "session_id": session_id or f"manual-{issued_at}",
        "scope": "analyze",
        "issued_at": issued_at
    }

    token = create_token(payload)

    html = f"""
    <html>
      <head>
        <title>SCRS Payment Success</title>
        <meta charset="utf-8" />
      </head>
      <body style="font-family: Arial, sans-serif; padding: 32px;">
        <h2>Payment Success</h2>
        <p>Your access token has been issued.</p>
        <p><strong>Token:</strong></p>
        <textarea rows="5" cols="100" readonly>{token}</textarea>
        <p style="margin-top: 16px;">Use this token in your API request header:</p>
        <pre>Authorization: Bearer {token}</pre>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


# =====================================================
# Analyze
# STEP 1A goal only:
# - Accept structured SCRS request
# - Validate token
# - Return structured echo + placeholder result
# We are NOT building the scoring engine yet in this step.
# =====================================================
@app.post("/analyze")
async def analyze(data: SCRSRequest, authorization: Optional[str] = Header(default=None)):
    token = verify_bearer_token(authorization)

    # Placeholder output for STEP 1A validation only
    # This confirms:
    # 1) token works
    # 2) schema works
    # 3) /analyze accepts SCRS structured input
    # Scoring engine comes in STEP 1B
    response = {
        "status": "accepted",
        "version": APP_VERSION,
        "message": "SCRS structured request received successfully.",
        "authorized": True,
        "token_preview": token[:12] + "...",
        "input_received": data.dict(),
        "next_step": "STEP1B_SCORING_ENGINE"
    }

    return JSONResponse(content=response)


# =====================================================
# Root
# =====================================================
@app.get("/")
async def root():
    return {
        "name": "SCRS API",
        "version": APP_VERSION,
        "status": "running"
    }
