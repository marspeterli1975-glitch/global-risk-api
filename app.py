import os
import json
import time
import hashlib
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from fastapi import FastAPI, Request as FastAPIRequest, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field

# third-party
from openai import OpenAI  # openai>=1.40.0
import stripe  # stripe>=10.x

APP_VERSION = "0.4.0-scrs-freemium"

# ---------------------------
# Env helpers
# ---------------------------
def _env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None else str(v).strip()

def _env_int(name: str, default: int) -> int:
    v = _env_str(name, "")
    if not v:
        return default
    try:
        return int(v)
    except Exception:
        return default

def _env_bool(name: str, default: bool = False) -> bool:
    v = _env_str(name, "")
    if not v:
        return default
    return v.lower() in ("1", "true", "yes", "y", "on")

def _split_keys(raw: str) -> List[str]:
    # supports: "k1,k2,k3" or "k1; k2" or newline
    if not raw:
        return []
    parts = []
    for sep in [",", ";", "\n", "\r\n"]:
        if sep in raw:
            raw = raw.replace(sep, ",")
    for p in raw.split(","):
        p = p.strip()
        if p:
            parts.append(p)
    return parts

# ---------------------------
# Config
# ---------------------------
DEBUG = _env_bool("DEBUG", False)

PAYMENT_MODE = _env_str("PAYMENT_MODE", "stripe").lower()  # stripe | off
STRIPE_SECRET_KEY = _env_str("STRIPE_SECRET_KEY", "")
OPENAI_API_KEY = _env_str("OPENAI_API_KEY", "")
LLM_MODEL = _env_str("LLM_MODEL", "gpt-4o-mini")
CACHE_TTL_SECONDS = _env_int("CACHE_TTL_SECONDS", 3600)
MAX_SIGNALS = _env_int("MAX_SIGNALS", 12)

APP_API_KEYS = _split_keys(_env_str("APP_API_KEYS", ""))  # optional: protect API

# token salt for redeeming a paid session into an API token
TOKEN_SALT = _env_str("TOKEN_SALT", "change-me-please")

# init stripe if configured
if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

# init openai if configured
_openai_client: Optional[OpenAI] = None
if OPENAI_API_KEY:
    _openai_client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Global Risk API", version=APP_VERSION)

# in-memory cache (simple)
_cache: Dict[str, Dict[str, Any]] = {}

# ---------------------------
# Models
# ---------------------------
class AnalyzeRequest(BaseModel):
    # 你之前问的 AnalyzeRequest 就在这里
    route: str = Field(..., description="Route string, e.g. Shanghai -> Auckland or 'CN-SHA to NZ-AKL'")
    cargo: Optional[str] = Field(None, description="Cargo description")
    incoterm: Optional[str] = Field(None, description="INCOTERM, e.g. FOB/CIF/DDP")
    amount_usd: Optional[float] = Field(None, description="Order amount in USD")
    horizon_days: int = Field(90, ge=7, le=365, description="Risk horizon in days")
    notes: Optional[str] = Field(None, description="Any extra notes")

class AnalyzeResponse(BaseModel):
    ok: bool
    mode: str
    paid: bool
    version: str
    result: Dict[str, Any]

# ---------------------------
# Auth / helpers
# ---------------------------
def _require_api_key(req: FastAPIRequest):
    if not APP_API_KEYS:
        return
    # allow either header or query
    k = req.headers.get("x-api-key") or req.query_params.get("api_key")
    if not k or k not in APP_API_KEYS:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid API key")

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _make_unlock_token(session_id: str) -> str:
    # deterministic token bound to session_id
    return _sha256(f"{session_id}::{TOKEN_SALT}")

def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    hit = _cache.get(key)
    if not hit:
        return None
    if time.time() - hit["ts"] > CACHE_TTL_SECONDS:
        _cache.pop(key, None)
        return None
    return hit["data"]

def _cache_set(key: str, data: Dict[str, Any]):
    _cache[key] = {"ts": time.time(), "data": data}

# ---------------------------
# Stripe verification
# ---------------------------
def stripe_is_ready() -> bool:
    return bool(STRIPE_SECRET_KEY)

def verify_stripe_session_paid(session_id: str) -> Dict[str, Any]:
    """
    Returns dict with:
      paid: bool
      currency, amount_total
      customer_details (email) if available
    """
    if not stripe_is_ready():
        raise HTTPException(status_code=500, detail="Stripe is not configured (missing STRIPE_SECRET_KEY).")

    cache_key = f"stripe_session::{session_id}"
    cached = _cache_get(cache_key)
    if cached:
        return cached

    try:
        # Payment Links create Checkout Sessions
        sess = stripe.checkout.Session.retrieve(
            session_id,
            expand=["line_items", "customer_details"]
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid session_id: {str(e)}")

    paid = (sess.get("payment_status") == "paid") or (sess.get("status") == "complete")

    data = {
        "paid": bool(paid),
        "currency": sess.get("currency"),
        "amount_total": sess.get("amount_total"),
        "customer_email": (sess.get("customer_details") or {}).get("email"),
        "id": sess.get("id"),
    }
    _cache_set(cache_key, data)
    return data

def is_unlock_token_valid(token: str, session_id: str) -> bool:
    return token == _make_unlock_token(session_id)

# ---------------------------
# Routes
# ---------------------------
@app.get("/")
def root():
    # Render 常会 HEAD / 探活；给 200，避免 404 造成误解
    return {"status": "ok", "service": "global-risk-api", "version": APP_VERSION}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "openai_configured": bool(_openai_client),
        "payment_mode": PAYMENT_MODE,
        "stripe_configured": stripe_is_ready(),
        "max_signals": MAX_SIGNALS,
    }

@app.get("/cancel")
def cancel():
    # 你可以在 Stripe Payment Link 里把 cancel_url 指到这里
    return HTMLResponse(
        "<h3>Payment cancelled</h3><p>You cancelled the payment. You can close this page.</p>"
    )

@app.get("/success")
def success(session_id: str):
    """
    Stripe Payment Link 成功后跳转到：
    https://global-risk-api.onrender.com/success?session_id={CHECKOUT_SESSION_ID}

    注意：如果你这里之前 404，说明你部署的 app.py 根本没有这个路由生效。
    """
    info = verify_stripe_session_paid(session_id)
    if not info["paid"]:
        return HTMLResponse(
            f"<h3>Payment not completed</h3><p>session_id={quote(session_id)}</p>",
            status_code=402,
        )

    unlock_token = _make_unlock_token(session_id)

    # 返回一个清晰的“下一步怎么用”的页面
    html = f"""
    <html>
      <body style="font-family: Arial, sans-serif; padding: 24px;">
        <h2>✅ Payment Success</h2>
        <p><b>session_id</b>: {quote(session_id)}</p>
        <p><b>amount</b>: {info.get("amount_total")} {info.get("currency")}</p>
        <p><b>customer</b>: {info.get("customer_email") or "-"}</p>
        <hr/>
        <h3>Next Step</h3>
        <p>Use this unlock token to call the API:</p>
        <pre style="background:#f4f4f4;padding:12px;border-radius:8px;">{unlock_token}</pre>

        <p><b>Example request</b> (POST /v1/analyze):</p>
        <pre style="background:#f4f4f4;padding:12px;border-radius:8px;">
Headers:
  x-api-key: (optional, if you set APP_API_KEYS)
Body:
{json.dumps({"route":"Shanghai -> Auckland","cargo":"battery materials","incoterm":"CIF","amount_usd":100000,"horizon_days":90,"notes":"test"}, indent=2)}
Query:
  ?session_id={quote(session_id)}&unlock_token={unlock_token}
        </pre>

        <p>You can close this page.</p>
      </body>
    </html>
    """
    return HTMLResponse(html)

@app.post("/v1/analyze", response_model=AnalyzeResponse)
async def analyze(req: FastAPIRequest, payload: AnalyzeRequest):
    """
    Freemium 逻辑：
    - PAYMENT_MODE=off: 不校验支付，直接返回
    - PAYMENT_MODE=stripe: 允许两种方式解锁：
        A) session_id + unlock_token (来自 /success 页面)
        B) session_id (服务端实时验证 paid，unlock_token 可选但建议用)
    """
    _require_api_key(req)

    paid = False
    mode = PAYMENT_MODE

    session_id = req.query_params.get("session_id", "").strip()
    unlock_token = req.query_params.get("unlock_token", "").strip()

    if mode == "stripe":
        if not session_id:
            # 没给 session_id：走 freemium（有限结果）
            paid = False
        else:
            # 有 session_id：校验
            info = verify_stripe_session_paid(session_id)
            if info["paid"]:
                # 如果提供 unlock_token，则再多一层一致性校验（防止别人随便拿 session_id 调用）
                if unlock_token and not is_unlock_token_valid(unlock_token, session_id):
                    raise HTTPException(status_code=401, detail="Invalid unlock_token for this session_id.")
                paid = True
            else:
                paid = False

    # -----------------------
    # Core logic (demo)
    # -----------------------
    # freemium 限制：只返回少量信号，不调用 LLM（或可调用但建议省钱）
    if not paid:
        result = {
            "tier": "freemium",
            "route": payload.route,
            "horizon_days": payload.horizon_days,
            "signals": [
                {"key": "port_congestion", "level": "medium"},
                {"key": "fx_volatility", "level": "medium"},
                {"key": "geopolitical", "level": "low"},
            ][: max(1, min(3, MAX_SIGNALS))],
            "note": "Freemium result. Provide a paid session_id to unlock full report.",
        }
        return AnalyzeResponse(ok=True, mode=mode, paid=False, version=APP_VERSION, result=result)

    # paid：允许调用 OpenAI 生成更完整报告（如果没配置 OpenAI 也能返回结构化结果）
    base = {
        "tier": "paid",
        "route": payload.route,
        "cargo": payload.cargo,
        "incoterm": payload.incoterm,
        "amount_usd": payload.amount_usd,
        "horizon_days": payload.horizon_days,
        "dimensions": [
            "geopolitical", "regulatory", "port", "carrier", "weather",
            "fx", "supplier", "fraud"
        ],
    }

    if not _openai_client:
        base["note"] = "Paid verified, but OpenAI not configured. Returning non-LLM report scaffold."
        base["risk_score"] = 62
        base["recommendations"] = [
            "Add buffer days to ETA and define fallback port.",
            "Add payment terms risk premium or use LC/DP where needed.",
            "Lock FX rate for 30–60 days if margin is thin."
        ]
        return AnalyzeResponse(ok=True, mode=mode, paid=True, version=APP_VERSION, result=base)

    # LLM call (kept minimal; you can harden later)
    prompt = f"""
You are a supply-chain risk analyst.
Return a JSON object ONLY.

Input:
route={payload.route}
cargo={payload.cargo}
incoterm={payload.incoterm}
amount_usd={payload.amount_usd}
horizon_days={payload.horizon_days}
notes={payload.notes}

Required JSON keys:
risk_score (0-100 int),
top_risks (array of 8 items: key, level, rationale),
recommendations (array of 5 items),
assumptions (array).
"""

    try:
        resp = _openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You output strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        text = resp.choices[0].message.content or "{}"
        # be defensive
        llm_json = json.loads(text)
    except Exception as e:
        llm_json = {"error": f"LLM failure: {str(e)}"}

    result = {**base, **llm_json}
    return AnalyzeResponse(ok=True, mode=mode, paid=True, version=APP_VERSION, result=result)

# ---------------------------
# Error handler (optional)
# ---------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(_: FastAPIRequest, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
