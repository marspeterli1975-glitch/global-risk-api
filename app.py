import os
import time
import json
import hmac
import base64
import hashlib
from typing import Optional, Dict, Any

import stripe
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

# -----------------------------
# App config
# -----------------------------
APP_ENV = os.getenv("APP_ENV", "production")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
SUCCESS_URL = os.getenv("SUCCESS_URL", "https://eastrion-site.vercel.app/success")
CANCEL_URL = os.getenv("CANCEL_URL", "https://eastrion-site.vercel.app/cancel")
TOKEN_SECRET = os.getenv("TOKEN_SECRET", "change-this-token-secret")
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "https://eastrion-site.vercel.app")
PAYMENT_MODE = os.getenv("PAYMENT_MODE", "payment")
PAID_TOKEN_TTL_SECONDS = int(os.getenv("PAID_TOKEN_TTL_SECONDS", "3600"))

# Stripe
if not STRIPE_SECRET_KEY:
    raise RuntimeError("Missing STRIPE_SECRET_KEY")
if not STRIPE_WEBHOOK_SECRET:
    # 允许先跑服务，但 webhook 会不可用
    print("WARNING: STRIPE_WEBHOOK_SECRET is missing. Webhook verification will fail.")

stripe.api_key = STRIPE_SECRET_KEY

app = FastAPI(
    title="RiskAtlas API",
    version="1.0.0",
    debug=DEBUG,
)

# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# -----------------------------
# In-memory stores
# Replace with DB in production later
# -----------------------------
SCAN_SESSIONS: Dict[str, Dict[str, Any]] = {}
PAID_SESSIONS: Dict[str, Dict[str, Any]] = {}
PROCESSED_WEBHOOK_EVENTS: Dict[str, float] = {}


# -----------------------------
# Models
# -----------------------------
class RiskScanRequest(BaseModel):
    company_name: str = Field(..., min_length=1, max_length=100)
    country: str = Field(..., min_length=1, max_length=60)
    industry: str = Field(..., min_length=1, max_length=80)
    logistics_complexity: int = Field(..., ge=1, le=5)
    event_disruption: int = Field(..., ge=1, le=5)
    email: str = Field(..., min_length=5, max_length=120)


class CheckoutRequest(BaseModel):
    session_id: str = Field(..., min_length=8, max_length=120)
    amount_cents: int = Field(..., ge=100, le=500000)
    currency: str = Field(default="aud", min_length=3, max_length=3)
    product_name: str = Field(default="RiskAtlas Exposure Report", min_length=3, max_length=120)


# -----------------------------
# Helpers
# -----------------------------
def _now_ts() -> int:
    return int(time.time())


def _make_session_id(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(raw + str(_now_ts()).encode("utf-8")).hexdigest()
    return f"scan_{digest[:24]}"


def _grade_from_score(score: int) -> str:
    if score <= 20:
        return "A"
    if score <= 40:
        return "B"
    if score <= 60:
        return "C"
    if score <= 80:
        return "D"
    return "E"


def _deterministic_risk_score(req: RiskScanRequest) -> Dict[str, Any]:
    """
    这里只是安全基线版示例。
    你后面可以把它替换成你原来的真实评分引擎。
    """
    country_risk = 25 if req.country.lower() not in {"australia", "singapore", "japan"} else 10
    industry_risk = 20 if req.industry.lower() in {"battery", "chemicals", "mining"} else 10
    logistics_risk = req.logistics_complexity * 10
    event_risk = req.event_disruption * 10

    total = min(country_risk + industry_risk + logistics_risk + event_risk, 100)
    grade = _grade_from_score(total)

    return {
        "score": total,
        "grade": grade,
        "dimensions": {
            "country_risk": country_risk,
            "industry_sensitivity": industry_risk,
            "logistics_complexity": logistics_risk,
            "event_disruption": event_risk,
        },
        "summary": f"RiskAtlas initial exposure score is {total}/100 with grade {grade}.",
    }


def _sign_download_token(session_id: str, email: str, ttl_seconds: int) -> str:
    exp = _now_ts() + ttl_seconds
    payload = {"sid": session_id, "email": email, "exp": exp}
    payload_b = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    sig = hmac.new(TOKEN_SECRET.encode("utf-8"), payload_b, hashlib.sha256).digest()
    token = base64.urlsafe_b64encode(payload_b).decode("utf-8") + "." + base64.urlsafe_b64encode(sig).decode("utf-8")
    return token


def _verify_download_token(token: str) -> Dict[str, Any]:
    try:
        payload_b64, sig_b64 = token.split(".")
        payload_b = base64.urlsafe_b64decode(payload_b64.encode("utf-8"))
        provided_sig = base64.urlsafe_b64decode(sig_b64.encode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid token format")

    expected_sig = hmac.new(TOKEN_SECRET.encode("utf-8"), payload_b, hashlib.sha256).digest()
    if not hmac.compare_digest(provided_sig, expected_sig):
        raise HTTPException(status_code=401, detail="Invalid token signature")

    try:
        payload = json.loads(payload_b.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid token payload")

    if payload.get("exp", 0) < _now_ts():
        raise HTTPException(status_code=401, detail="Token expired")

    return payload


def _build_mock_report(session_id: str) -> Dict[str, Any]:
    """
    这里先返回 JSON 报告占位。
    你后面可以在这里接入真实 PDF 生成逻辑。
    """
    paid = PAID_SESSIONS.get(session_id)
    if not paid:
        raise HTTPException(status_code=404, detail="Paid report not found")

    scan = SCAN_SESSIONS.get(session_id, {})
    analysis = scan.get("analysis", {})

    return {
        "report_type": "RiskAtlas Consulting Report",
        "session_id": session_id,
        "company_name": scan.get("company_name"),
        "email": scan.get("email"),
        "country": scan.get("country"),
        "industry": scan.get("industry"),
        "score": analysis.get("score"),
        "grade": analysis.get("grade"),
        "dimensions": analysis.get("dimensions"),
        "summary": analysis.get("summary"),
        "payment_verified": True,
        "generated_at": _now_ts(),
    }


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
async def root():
    return {"ok": True, "service": "RiskAtlas API", "env": APP_ENV}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "stripe_configured": bool(STRIPE_SECRET_KEY),
        "webhook_configured": bool(STRIPE_WEBHOOK_SECRET),
        "time": _now_ts(),
    }


@app.post("/scan")
async def create_scan(req: RiskScanRequest):
    analysis = _deterministic_risk_score(req)
    session_id = _make_session_id(req.model_dump())

    SCAN_SESSIONS[session_id] = {
        "session_id": session_id,
        "company_name": req.company_name,
        "country": req.country,
        "industry": req.industry,
        "email": req.email,
        "analysis": analysis,
        "created_at": _now_ts(),
        "payment_status": "created",
    }

    return {
        "session_id": session_id,
        "analysis": analysis,
        "payment_status": "created",
        "next_step": "create_checkout",
    }


@app.post("/create-checkout-session")
async def create_checkout_session(req: CheckoutRequest):
    scan = SCAN_SESSIONS.get(req.session_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Session not found")

    # 防止重复为已支付订单创建新的 checkout
    if scan.get("payment_status") == "paid":
        raise HTTPException(status_code=400, detail="This session is already paid")

    try:
        checkout_session = stripe.checkout.Session.create(
            mode=PAYMENT_MODE,
            success_url=f"{SUCCESS_URL}?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=CANCEL_URL,
            customer_email=scan["email"],
            line_items=[
                {
                    "price_data": {
                        "currency": req.currency.lower(),
                        "product_data": {
                            "name": req.product_name,
                        },
                        "unit_amount": req.amount_cents,
                    },
                    "quantity": 1,
                }
            ],
            metadata={
                "riskatlas_session_id": req.session_id,
                "company_name": scan["company_name"],
                "country": scan["country"],
                "industry": scan["industry"],
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stripe checkout creation failed: {str(e)}")

    scan["payment_status"] = "checkout_created"
    scan["checkout_session_id"] = checkout_session.id

    return {
        "checkout_url": checkout_session.url,
        "checkout_session_id": checkout_session.id,
        "payment_status": "checkout_created",
    }


@app.post("/webhook")
async def stripe_webhook(request: Request):
    """
    生产关键点：
    1. 只认 Stripe webhook，不认前端 success 页面
    2. 必须验签
    3. 必须防止重复处理
    """
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    if not sig_header:
        raise HTTPException(status_code=400, detail="Missing Stripe-Signature header")

    try:
        event = stripe.Webhook.construct_event(
            payload=payload,
            sig_header=sig_header,
            secret=STRIPE_WEBHOOK_SECRET,
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    event_id = event.get("id")
    if event_id in PROCESSED_WEBHOOK_EVENTS:
        return {"status": "already_processed", "event_id": event_id}

    PROCESSED_WEBHOOK_EVENTS[event_id] = time.time()

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        payment_status = session.get("payment_status")
        checkout_session_id = session.get("id")
        metadata = session.get("metadata", {}) or {}
        riskatlas_session_id = metadata.get("riskatlas_session_id")

        if not riskatlas_session_id:
            raise HTTPException(status_code=400, detail="Missing riskatlas_session_id in metadata")

        if payment_status != "paid":
            return {"status": "ignored", "reason": f"payment_status={payment_status}"}

        scan = SCAN_SESSIONS.get(riskatlas_session_id)
        if not scan:
            raise HTTPException(status_code=404, detail="Referenced scan session not found")

        download_token = _sign_download_token(
            session_id=riskatlas_session_id,
            email=scan["email"],
            ttl_seconds=PAID_TOKEN_TTL_SECONDS,
        )

        scan["payment_status"] = "paid"
        scan["paid_at"] = _now_ts()

        PAID_SESSIONS[riskatlas_session_id] = {
            "session_id": riskatlas_session_id,
            "checkout_session_id": checkout_session_id,
            "stripe_event_id": event_id,
            "email": scan["email"],
            "download_token": download_token,
            "paid_at": _now_ts(),
        }

        print("Stripe payment success")
        print("riskatlas_session_id:", riskatlas_session_id)
        print("checkout_session_id:", checkout_session_id)
        print("email:", scan["email"])

    return {"status": "success"}


@app.get("/payment-status/{session_id}")
async def payment_status(session_id: str):
    scan = SCAN_SESSIONS.get(session_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Session not found")

    result = {
        "session_id": session_id,
        "payment_status": scan.get("payment_status", "unknown"),
    }

    if scan.get("payment_status") == "paid":
        paid = PAID_SESSIONS.get(session_id, {})
        result["download_token"] = paid.get("download_token")

    return result


@app.get("/report")
async def get_report(token: str):
    """
    这里先返回 JSON 报告占位。
    后面你把它接成真实 PDF 下载即可。
    """
    payload = _verify_download_token(token)
    session_id = payload["sid"]

    report = _build_mock_report(session_id)
    return JSONResponse(content=report)


# -----------------------------
# Optional custom error handler
# -----------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    if DEBUG:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {str(exc)}"},
        )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
