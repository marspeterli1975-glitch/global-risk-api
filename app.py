import os
import json
import time
import hashlib
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

import stripe
from openai import OpenAI


# =========================
# Config
# =========================
APP_VERSION = os.getenv("APP_VERSION", "0.4.0-scrs-freemium")
DEBUG = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes", "y")

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

PAYMENT_MODE = os.getenv("PAYMENT_MODE", "stripe").strip().lower()  # stripe / off / free
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "").strip()
# STRIPE_WEBHOOK_SECRET 可选，如果你未来要用 webhook 来可靠发放权限
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "").strip()

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))  # 默认 1 天
MAX_SIGNALS = int(os.getenv("MAX_SIGNALS", "8"))

# 你自己的 API Key 白名单（逗号分隔）
APP_API_KEYS_RAW = os.getenv("APP_API_KEYS", "").strip()
APP_API_KEYS = [x.strip() for x in APP_API_KEYS_RAW.split(",") if x.strip()]

# 用于生成 token 的服务端盐（不设置就用 STRIPE_SECRET_KEY 的 hash 兜底）
TOKEN_SALT = os.getenv("TOKEN_SALT", "").strip()
if not TOKEN_SALT:
    TOKEN_SALT = hashlib.sha256((STRIPE_SECRET_KEY or "fallback").encode("utf-8")).hexdigest()

# Stripe 初始化
stripe_configured = False
if PAYMENT_MODE == "stripe" and STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY
    stripe_configured = True


# OpenAI 初始化
openai_configured = False
client: Optional[OpenAI] = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
    openai_configured = True


# =========================
# In-memory cache (Render 重启会丢；后续可升级到 DB/Redis)
# =========================
# paid_sessions[session_id] = {"paid": True, "created": ts, "email": "...", "amount_total": 2900, "currency": "usd"}
paid_sessions: Dict[str, Dict[str, Any]] = {}

# issued_tokens[token] = {"session_id": "...", "created": ts}
issued_tokens: Dict[str, Dict[str, Any]] = {}


def _now() -> int:
    return int(time.time())


def _cleanup_cache() -> None:
    """简单 TTL 清理"""
    cutoff = _now() - CACHE_TTL_SECONDS
    for sid in list(paid_sessions.keys()):
        if int(paid_sessions[sid].get("created", 0)) < cutoff:
            paid_sessions.pop(sid, None)
    for tok in list(issued_tokens.keys()):
        if int(issued_tokens[tok].get("created", 0)) < cutoff:
            issued_tokens.pop(tok, None)


def _make_access_token(session_id: str) -> str:
    raw = f"{session_id}:{TOKEN_SALT}:{_now()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _extract_bearer(request: Request) -> str:
    auth = request.headers.get("authorization", "").strip()
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    return ""


def _require_api_key_or_paid_token(request: Request) -> Dict[str, Any]:
    """
    允许两种方式通过：
    1) X-API-Key: <your_key>  (在 APP_API_KEYS 白名单中)
    2) Authorization: Bearer <token>  (由 /success 发放)
    """
    _cleanup_cache()

    # 方式1：自定义 API Key
    api_key = request.headers.get("x-api-key", "").strip()
    if api_key and (not APP_API_KEYS or api_key in APP_API_KEYS):
        return {"mode": "api_key", "api_key": api_key}

    # 方式2：Stripe 支付后 token
    token = _extract_bearer(request)
    if token and token in issued_tokens:
        return {"mode": "paid_token", "token": token, "session_id": issued_tokens[token]["session_id"]}

    raise HTTPException(status_code=401, detail="Unauthorized. Provide X-API-Key or Bearer token.")


def _stripe_retrieve_session(session_id: str) -> Dict[str, Any]:
    if not stripe_configured:
        raise HTTPException(status_code=500, detail="Stripe not configured. Check STRIPE_SECRET_KEY / PAYMENT_MODE.")
    try:
        sess = stripe.checkout.Session.retrieve(session_id)
        # sess 是 stripe object，转 dict
        return json.loads(str(sess))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid session_id or Stripe error: {str(e)}")


def _stripe_is_paid(session: Dict[str, Any]) -> bool:
    # Stripe checkout session 常用字段：
    # payment_status: "paid" / "unpaid"
    return str(session.get("payment_status", "")).lower() == "paid"


# =========================
# Pydantic models
# =========================
class AnalyzeRequest(BaseModel):
    """你问的 AnalyzeRequest 在这里（Pydantic Model）"""
    route: str = Field(..., description="Route or scenario description, e.g. 'Shanghai->Mumbai, lithium battery materials'")
    context: Optional[str] = Field(None, description="Extra context: ports, suppliers, HS codes, Incoterms, etc.")
    horizon_days: int = Field(90, ge=1, le=365, description="Risk horizon in days")
    max_signals: int = Field(MAX_SIGNALS, ge=1, le=30, description="How many signals to return")


class AnalyzeResponse(BaseModel):
    status: str
    version: str
    mode: str
    result: Dict[str, Any]


# =========================
# App
# =========================
app = FastAPI(title="Global Risk API", version=APP_VERSION)


@app.get("/")
def root():
    # 你访问主域名时不再是 404
    return {
        "status": "ok",
        "service": "global-risk-api",
        "version": APP_VERSION,
        "endpoints": ["/health", "/analyze", "/success", "/cancel"],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "openai_configured": bool(openai_configured),
        "payment_mode": PAYMENT_MODE,
        "stripe_configured": bool(stripe_configured),
    }


# =========================
# Stripe return URLs
# =========================
@app.get("/success")
def stripe_success(session_id: str):
    """
    Stripe Payment Link 的确认页 URL 建议设置到：
    https://global-risk-api.onrender.com/success?session_id={CHECKOUT_SESSION_ID}

    这个接口会：
    1) 去 Stripe 拉取 session
    2) 校验 payment_status == paid
    3) 生成一个 Bearer token（有效期受 CACHE_TTL_SECONDS 影响）
    """
    _cleanup_cache()

    if PAYMENT_MODE != "stripe":
        return {"status": "ok", "note": "PAYMENT_MODE is not stripe", "version": APP_VERSION}

    sess = _stripe_retrieve_session(session_id)
    if not _stripe_is_paid(sess):
        raise HTTPException(status_code=402, detail="Payment not completed (payment_status != paid).")

    email = (sess.get("customer_details") or {}).get("email")
    amount_total = sess.get("amount_total")
    currency = sess.get("currency")

    paid_sessions[session_id] = {
        "paid": True,
        "created": _now(),
        "email": email,
        "amount_total": amount_total,
        "currency": currency,
    }

    token = _make_access_token(session_id)
    issued_tokens[token] = {"session_id": session_id, "created": _now()}

    # 你可以改成 Redirect 到你的前端页面：
    # return RedirectResponse(url=f"https://your-frontend.com/paid?token={token}")
    return {
        "status": "ok",
        "version": APP_VERSION,
        "message": "Payment verified. Use this token to call /analyze with Authorization: Bearer <token>",
        "token": token,
        "session_id": session_id,
        "email": email,
        "amount_total": amount_total,
        "currency": currency,
        "ttl_seconds": CACHE_TTL_SECONDS,
    }


@app.get("/cancel")
def stripe_cancel():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "message": "Payment canceled or not completed.",
    }


# （可选）如果你后续要做 webhook：更可靠、不会因 Render 重启丢失状态
@app.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="STRIPE_WEBHOOK_SECRET not set.")

    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Webhook signature verification failed: {str(e)}")

    # 你可以在这里处理 checkout.session.completed 等事件
    return {"received": True, "type": event.get("type")}


# =========================
# Business endpoint
# =========================
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest, request: Request):
    """
    调用方式：
    - 付费后：Authorization: Bearer <token>
    - 或者你自己给客户发：X-API-Key: <key>
    """
    auth = _require_api_key_or_paid_token(request)

    if not openai_configured or client is None:
        raise HTTPException(status_code=500, detail="OpenAI not configured. Set OPENAI_API_KEY.")

    # 简单示例：让模型输出结构化风险要点（你后续可以换成你的 risk engine）
    prompt = f"""
You are a supply-chain risk analyst.
Return a JSON object with:
- summary (string)
- horizon_days (int)
- key_risks (array of objects: name, likelihood_0_1, impact_0_10, notes)
- suggested_actions (array of strings)
Constraints:
- max key_risks = {min(req.max_signals, 30)}
Route/scenario: {req.route}
Context: {req.context or ""}
Horizon days: {req.horizon_days}
"""

    try:
        rsp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "Return strictly valid JSON, no markdown, no extra text."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        content = rsp.choices[0].message.content or "{}"
        result = json.loads(content)
    except json.JSONDecodeError:
        # 兜底：避免模型偶发不严格 JSON
        result = {
            "summary": "Model output was not valid JSON; returning raw text.",
            "raw": (rsp.choices[0].message.content if 'rsp' in locals() else None),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analyze failed: {str(e)}")

    return AnalyzeResponse(
        status="ok",
        version=APP_VERSION,
        mode=auth["mode"],
        result=result,
    )
