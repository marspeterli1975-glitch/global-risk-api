import os
import time
import json
import hashlib
import hmac
import base64

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import stripe
from dotenv import load_dotenv

# -----------------------------
# Load env
# -----------------------------
load_dotenv()

# -----------------------------
# ENV
# -----------------------------
APP_ENV = os.getenv("APP_ENV", "production")

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")

TOKEN_SECRET = os.getenv("TOKEN_SECRET", "riskatlas-secret")
SUCCESS_URL = os.getenv("SUCCESS_URL", "")
CANCEL_URL = os.getenv("CANCEL_URL", "")

ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")

PAID_TOKEN_TTL_SECONDS = int(os.getenv("PAID_TOKEN_TTL_SECONDS", "3600"))

stripe.api_key = STRIPE_SECRET_KEY

# -----------------------------
# App
# -----------------------------
app = FastAPI(
    title="RiskAtlas API",
    version="1.0.0"
)

# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Memory storage
# -----------------------------
SCANS = {}
PAID = {}

# -----------------------------
# Models
# -----------------------------
class RiskScanRequest(BaseModel):
    company_name: str
    country: str
    industry: str
    logistics_complexity: int = Field(..., ge=1, le=5)
    event_disruption: int = Field(..., ge=1, le=5)
    email: str


class CheckoutRequest(BaseModel):
    session_id: str
    amount_cents: int
    currency: str = "aud"


# -----------------------------
# Helpers
# -----------------------------
def now():
    return int(time.time())


def generate_session_id(payload):
    raw = json.dumps(payload, sort_keys=True).encode()
    digest = hashlib.sha256(raw + str(now()).encode()).hexdigest()
    return f"scan_{digest[:24]}"


def grade(score):
    if score <= 20:
        return "A"
    if score <= 40:
        return "B"
    if score <= 60:
        return "C"
    if score <= 80:
        return "D"
    return "E"


def risk_engine(req):
    country_risk = 25
    industry_risk = 20
    logistics_risk = req.logistics_complexity * 10
    event_risk = req.event_disruption * 10

    total = min(country_risk + industry_risk + logistics_risk + event_risk, 100)

    return {
        "score": total,
        "grade": grade(total)
    }


def sign_token(session_id, email):

    payload = {
        "sid": session_id,
        "email": email,
        "exp": now() + PAID_TOKEN_TTL_SECONDS
    }

    body = json.dumps(payload).encode()

    sig = hmac.new(
        TOKEN_SECRET.encode(),
        body,
        hashlib.sha256
    ).digest()

    token = base64.urlsafe_b64encode(body).decode() + "." + base64.urlsafe_b64encode(sig).decode()

    return token


def verify_token(token):

    try:
        p, s = token.split(".")
    except:
        raise HTTPException(401)

    body = base64.urlsafe_b64decode(p.encode())
    sig = base64.urlsafe_b64decode(s.encode())

    expected = hmac.new(
        TOKEN_SECRET.encode(),
        body,
        hashlib.sha256
    ).digest()

    if not hmac.compare_digest(sig, expected):
        raise HTTPException(401)

    payload = json.loads(body)

    if payload["exp"] < now():
        raise HTTPException(401)

    return payload


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
async def root():
    return {"ok": True, "service": "RiskAtlas API"}


@app.get("/health")
async def health():
    return {"status": "ok", "time": now()}


@app.post("/scan")
async def scan(req: RiskScanRequest):

    analysis = risk_engine(req)

    session_id = generate_session_id(req.dict())

    SCANS[session_id] = {
        "session_id": session_id,
        "email": req.email,
        "analysis": analysis,
        "created": now(),
        "paid": False
    }

    return {
        "session_id": session_id,
        "analysis": analysis
    }


@app.post("/create-checkout-session")
async def checkout(req: CheckoutRequest):

    scan = SCANS.get(req.session_id)

    if not scan:
        raise HTTPException(404)

    session = stripe.checkout.Session.create(
        mode="payment",
        success_url=SUCCESS_URL,
        cancel_url=CANCEL_URL,
        customer_email=scan["email"],
        line_items=[
            {
                "price_data": {
                    "currency": req.currency,
                    "product_data": {"name": "RiskAtlas Report"},
                    "unit_amount": req.amount_cents,
                },
                "quantity": 1
            }
        ],
        metadata={
            "riskatlas_session_id": req.session_id
        }
    )

    return {
        "checkout_url": session.url
    }


@app.post("/webhook")
async def webhook(request: Request):

    payload = await request.body()

    sig = request.headers.get("stripe-signature")

    try:

        event = stripe.Webhook.construct_event(
            payload,
            sig,
            STRIPE_WEBHOOK_SECRET
        )

    except Exception:
        raise HTTPException(400)

    if event["type"] == "checkout.session.completed":

        session = event["data"]["object"]

        sid = session["metadata"]["riskatlas_session_id"]

        scan = SCANS.get(sid)

        if not scan:
            return {"status": "unknown"}

        scan["paid"] = True

        token = sign_token(sid, scan["email"])

        PAID[sid] = {
            "token": token,
            "time": now()
        }

    return {"status": "ok"}


@app.get("/report")
async def report(token: str):

    payload = verify_token(token)

    sid = payload["sid"]

    scan = SCANS.get(sid)

    if not scan:
        raise HTTPException(404)

    return {
        "title": "Supply Chain Risk Report",
        "score": scan["analysis"]["score"],
        "grade": scan["analysis"]["grade"]
    })
