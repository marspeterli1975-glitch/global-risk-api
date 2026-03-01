"""
app.py — SCRS FastAPI (full replace)

Features
- /health : health check
- /analyze : protected (X-API-Key OR Authorization token/Bearer token)
- /success : Stripe redirect landing — verifies session_id, issues short-lived paid token, returns HTML that stores token and can auto-call /analyze
- Optional /create-checkout-session : create Stripe checkout session (useful if you later want your own pay button)

Auth rules
- Provide either:
  A) Header: X-API-Key: <your_api_key>
  OR
  B) Header: Authorization: Bearer <paid_token>   (also accepts Authorization: <paid_token>)

ENV you should set on Render
- APP_VERSION="0.4.0-scrs-freemium"
- TOKEN_SECRET="a_long_random_secret"          # REQUIRED for paid token signing
- API_KEYS="peterbingli-202602-risk-001"       # optional; comma-separated
- STRIPE_SECRET_KEY="sk_test_..."              # REQUIRED for /success (verify checkout session)
- STRIPE_PRICE_ID="price_..."                  # optional; for /create-checkout-session
- SUCCESS_URL="https://global-risk-api.onrender.com/success?session_id={CHECKOUT_SESSION_ID}"
- CANCEL_URL="https://global-risk-api.onrender.com/cancel"
- PAID_TOKEN_TTL_SECONDS="900"                 # default 900s
- CORS_ALLOW_ORIGINS="*"                       # or comma-separated origins

Run (Render)
- Start command: uvicorn app:app --host 0.0.0.0 --port 10000
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

# Stripe is optional at import time; /success needs it
try:
    import stripe  # type: ignore
except Exception:
    stripe = None


# =========================
# Config
# =========================

APP_VERSION = os.getenv("APP_VERSION", "0.4.0-scrs-freemium")

TOKEN_SECRET = os.getenv("TOKEN_SECRET", "")  # REQUIRED for signing paid tokens
API_KEYS_RAW = os.getenv("API_KEYS", "").strip()
API_KEYS = {k.strip() for k in API_KEYS_RAW.split(",") if k.strip()}

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "").strip()
STRIPE_PRICE_ID = os.getenv("STRIPE_PRICE_ID", "").strip()

SUCCESS_URL = os.getenv(
    "SUCCESS_URL",
    "https://global-risk-api.onrender.com/success?session_id={CHECKOUT_SESSION_ID}",
).strip()
CANCEL_URL = os.getenv("CANCEL_URL", "https://global-risk-api.onrender.com/cancel").strip()

PAID_TOKEN_TTL_SECONDS = int(os.getenv("PAID_TOKEN_TTL_SECONDS", "900").strip() or "900")

CORS_ALLOW_ORIGINS_RAW = os.getenv("CORS_ALLOW_ORIGINS", "*").strip()
if CORS_ALLOW_ORIGINS_RAW == "*":
    CORS_ALLOW_ORIGINS = ["*"]
else:
    CORS_ALLOW_ORIGINS = [o.strip() for o in CORS_ALLOW_ORIGINS_RAW.split(",") if o.strip()]


# =========================
# App
# =========================

app = FastAPI(title="SCRS API", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# In-memory token store
# =========================

@dataclass
class TokenRecord:
    token: str
    session_id: str
    email: str
    amount_total: int
    currency: str
    issued_at: int
    expires_at: int


# token -> record
ISSUED_TOKENS: Dict[str, TokenRecord] = {}


def _now() -> int:
    return int(time.time())


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64url_decode(s: str) -> bytes:
    pad = "=" * ((4 - (len(s) % 4)) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("utf-8"))


def _hmac_sha256(key: bytes, msg: bytes) -> bytes:
    return hmac.new(key, msg, hashlib.sha256).digest()


def _sign_token(payload: dict) -> str:
    """
    Token format: base64url(payload_json).base64url(sig)
    """
    if not TOKEN_SECRET:
        raise RuntimeError("TOKEN_SECRET is missing")

    payload_bytes = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    body = _b64url(payload_bytes).encode("utf-8")
    sig = _b64url(_hmac_sha256(TOKEN_SECRET.encode("utf-8"), body))
    return f"{body.decode('utf-8')}.{sig}"


def _verify_token(token: str) -> Tuple[bool, Optional[dict]]:
    if not TOKEN_SECRET:
        return False, None

    if "." not in token:
        return False, None

    body_b64, sig_b64 = token.split(".", 1)
    body_bytes = body_b64.encode("utf-8")
    expected_sig = _b64url(_hmac_sha256(TOKEN_SECRET.encode("utf-8"), body_bytes))

    # constant-time compare
    if not hmac.compare_digest(expected_sig, sig_b64):
        return False, None

    try:
        payload = json.loads(_b64url_decode(body_b64).decode("utf-8"))
        return True, payload
    except Exception:
        return False, None


def _clean_expired_tokens() -> None:
    now = _now()
    expired = [t for t, rec in ISSUED_TOKENS.items() if rec.expires_at <= now]
    for t in expired:
        ISSUED_TOKENS.pop(t, None)


def issue_paid_token(*, session_id: str, email: str, amount_total: int, currency: str) -> TokenRecord:
    _clean_expired_tokens()

    issued_at = _now()
    expires_at = issued_at + PAID_TOKEN_TTL_SECONDS

    payload = {
        "typ": "scrs_paid",
        "sid": session_id,
        "iat": issued_at,
        "exp": expires_at,
        "jti": uuid.uuid4().hex,
        "email": email or "",
        "amt": int(amount_total or 0),
        "cur": (currency or "").lower(),
        "v": APP_VERSION,
    }

    token = _sign_token(payload)

    rec = TokenRecord(
        token=token,
        session_id=session_id,
        email=email or "",
        amount_total=int(amount_total or 0),
        currency=(currency or "").lower(),
        issued_at=issued_at,
        expires_at=expires_at,
    )

    ISSUED_TOKENS[token] = rec
    return rec


def _extract_auth_token(authorization: Optional[str]) -> Optional[str]:
    """
    Accept:
    - "Bearer <token>"
    - "<token>" (raw)
    """
    if not authorization:
        return None
    auth = authorization.strip()
    if not auth:
        return None
    parts = auth.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return auth


def _require_auth(
    *,
    x_api_key: Optional[str],
    authorization: Optional[str],
) -> dict:
    """
    Returns auth context dict if authorized, else raises 401.
    """
    # 1) Master API key path
    if x_api_key and x_api_key.strip() and (not API_KEYS or x_api_key.strip() in API_KEYS):
        return {"mode": "api_key", "principal": "api_key"}

    # 2) Token path
    tok = _extract_auth_token(authorization)
    if not tok:
        raise HTTPException(status_code=401, detail="Unauthorized. Provide X-API-Key or Bearer token.")

    # First validate signature & expiry
    ok, payload = _verify_token(tok)
    if not ok or not payload:
        raise HTTPException(status_code=401, detail="Unauthorized. Invalid token.")

    exp = int(payload.get("exp") or 0)
    if exp <= _now():
        raise HTTPException(status_code=401, detail="Unauthorized. Token expired.")

    # Optional: require it exists in our issued store (prevents random signed token reuse if you rotate)
    rec = ISSUED_TOKENS.get(tok)
    if not rec:
        # If you want to allow stateless tokens only, comment out this block.
        raise HTTPException(status_code=401, detail="Unauthorized. Token not recognized.")

    return {
        "mode": "paid_token",
        "principal": payload.get("email") or "paid_user",
        "session_id": payload.get("sid"),
        "expires_at": exp,
    }


# =========================
# Helpers: risk analysis (simple baseline)
# =========================

def _simple_risk_engine(route: str, context: str, horizon_days: int, language: str = "en") -> dict:
    """
    Replace this function with your real risk engine / OpenAI calls.
    Keep deterministic-ish and structured.
    """
    route_l = (route or "").lower()
    context_l = (context or "").lower()
    horizon_days = int(horizon_days or 30)

    # Baseline factors
    base = 0.55
    if "lithium" in context_l or "battery" in context_l:
        base += 0.10
    if "cif" in context_l:
        base += 0.05
    if "shanghai" in route_l and "tokyo" in route_l:
        base += 0.05
    if horizon_days >= 90:
        base += 0.05

    base = max(0.05, min(0.95, base))

    # Build 3 risks
    risks = [
        {
            "name": "Geopolitical Tensions" if language == "en" else "地缘政治紧张",
            "likelihood_0_1": round(min(0.95, base + 0.10), 2),
            "impact_0_10": 8,
            "notes": "Potential trade restrictions / policy swings could affect routing and customs." if language == "en"
            else "潜在贸易限制/政策波动可能影响航线与通关。",
        },
        {
            "name": "Regulatory Changes" if language == "en" else "监管变动",
            "likelihood_0_1": round(min(0.95, base + 0.00), 2),
            "impact_0_10": 7,
            "notes": "Changes in import/export compliance (e.g., batteries/chemicals) may introduce delays." if language == "en"
            else "进出口合规（电池/化学品）要求变化可能带来延误。",
        },
        {
            "name": "Supply Disruptions" if language == "en" else "供应扰动",
            "likelihood_0_1": round(min(0.95, base - 0.05), 2),
            "impact_0_10": 7,
            "notes": "Port congestion, carrier blank sailings, or upstream disruptions can shift ETD/ETA." if language == "en"
            else "港口拥堵、停航或上游扰动会影响ETD/ETA。",
        },
    ]

    overall = sum(r["likelihood_0_1"] * (r["impact_0_10"] / 10) for r in risks) / len(risks)
    overall = round(overall, 2)

    if language == "en":
        summary = (
            f"The supply chain route '{route}' faces moderate-to-elevated risk over {horizon_days} days. "
            f"Key drivers include policy/regulatory uncertainty and disruption volatility."
        )
    else:
        summary = (
            f"航线“{route}”在未来{horizon_days}天面临中等偏高风险，主要驱动来自政策/监管不确定性与扰动波动。"
        )

    return {
        "summary": summary,
        "horizon_days": horizon_days,
        "route": route,
        "context": context,
        "overall_risk_0_1": overall,
        "key_risks": risks,
        "recommended_actions": (
            [
                "Pre-book capacity and set contingency routing." ,
                "Lock compliance checklist for dangerous goods / batteries.",
                "Build buffer stock for critical SKUs; monitor port/carrier alerts.",
            ]
            if language == "en"
            else
            [
                "提前订舱并准备备选航线。",
                "固化危险品/电池合规清单并前置审核。",
                "关键SKU设置安全库存缓冲，持续监测港口/船司预警。",
            ]
        ),
    }


# =========================
# Routes
# =========================

@app.get("/", response_class=JSONResponse)
def home() -> dict:
    # You previously saw {"detail":"Not Found"} on root; returning a minimal payload is usually nicer.
    return {
        "status": "ok",
        "service": "scrs-api",
        "version": APP_VERSION,
        "endpoints": ["/health", "/analyze", "/success"],
    }


@app.get("/health", response_class=JSONResponse)
def health() -> dict:
    stripe_configured = bool(STRIPE_SECRET_KEY)
    return {
        "status": "ok",
        "version": APP_VERSION,
        "payment_mode": "stripe",
        "stripe_configured": stripe_configured,
    }


@app.post("/analyze", response_class=JSONResponse)
async def analyze(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> dict:
    auth_ctx = _require_auth(x_api_key=x_api_key, authorization=authorization)

    body = await request.json()
    # Accept both old and new param names
    route = body.get("route") or body.get("location") or ""
    context = body.get("context") or body.get("notes") or ""
    horizon_days = body.get("horizon_days") or body.get("horizonDays") or 90
    language = body.get("language") or "en"

    req_id = uuid.uuid4().hex[:12]

    result = _simple_risk_engine(
        route=str(route),
        context=str(context),
        horizon_days=int(horizon_days),
        language=str(language).lower(),
    )

    return {
        "status": "ok",
        "version": APP_VERSION,
        "mode": auth_ctx.get("mode"),
        "request_id": req_id,
        "auth": {
            "principal": auth_ctx.get("principal"),
            "session_id": auth_ctx.get("session_id"),
            "expires_at": auth_ctx.get("expires_at"),
        },
        "result": result,
    }


@app.get("/success", response_class=JSONResponse)
def success(session_id: str) -> Any:
    """
    Stripe redirects here as:
    /success?session_id={CHECKOUT_SESSION_ID}

    We verify checkout session, issue paid token, then return an HTML page:
    - stores token in localStorage
    - offers button to POST /analyze
    - can auto-run analyze if desired
    """
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing session_id")

    if stripe is None:
        raise HTTPException(status_code=500, detail="Stripe library not installed")

    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=500, detail="STRIPE_SECRET_KEY not configured")

    stripe.api_key = STRIPE_SECRET_KEY

    try:
        # Expand customer_details for email; expand payment_intent if you want
        sess = stripe.checkout.Session.retrieve(
            session_id,
            expand=["customer_details"],
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid session_id or Stripe error: {str(e)}")

    # Stripe test/live differs, but generally: status == "complete" and/or payment_status == "paid"
    status = getattr(sess, "status", None) or ""
    payment_status = getattr(sess, "payment_status", None) or ""

    if (status != "complete") and (payment_status != "paid"):
        raise HTTPException(
            status_code=402,
            detail=f"Payment not complete. status={status}, payment_status={payment_status}",
        )

    customer_details = getattr(sess, "customer_details", None)
    email = ""
    if customer_details and getattr(customer_details, "email", None):
        email = customer_details.email or ""

    amount_total = int(getattr(sess, "amount_total", 0) or 0)
    currency = str(getattr(sess, "currency", "usd") or "usd")

    rec = issue_paid_token(
        session_id=session_id,
        email=email,
        amount_total=amount_total,
        currency=currency,
    )

    # IMPORTANT: You can choose JSON or HTML.
    # For end users, HTML is better. If you want JSON-only, return dict instead.
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Payment Verified</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 0; background:#0b0f14; color:#e6edf3; }}
    .wrap {{ max-width: 920px; margin: 40px auto; padding: 24px; }}
    .card {{ background:#111827; border:1px solid #243040; border-radius: 14px; padding: 18px; }}
    code, pre {{ background:#0b1220; border:1px solid #22314a; border-radius: 10px; padding: 10px; overflow:auto; color:#d1fae5; }}
    .row {{ display:flex; gap:12px; flex-wrap:wrap; }}
    .btn {{ background:#2563eb; border:none; color:white; padding:10px 14px; border-radius: 10px; cursor:pointer; font-weight:600; }}
    .btn2 {{ background:#10b981; }}
    .muted {{ color:#9ca3af; font-size: 13px; }}
    input, textarea {{ width:100%; background:#0b1220; border:1px solid #22314a; color:#e6edf3; border-radius: 10px; padding:10px; }}
    label {{ display:block; margin:10px 0 6px; font-weight:600; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h2>✅ Payment verified</h2>
    <p class="muted">Token issued (TTL {PAID_TOKEN_TTL_SECONDS}s). This page stores it locally for calling <code>/analyze</code>.</p>

    <div class="card">
      <div class="row">
        <div style="flex:1; min-width: 280px;">
          <div><b>session_id</b></div>
          <code>{rec.session_id}</code>
        </div>
        <div style="flex:1; min-width: 280px;">
          <div><b>email</b></div>
          <code>{rec.email or "-"}</code>
        </div>
      </div>

      <label>Route</label>
      <input id="route" value="Shanghai -> Tokyo" />

      <label>Context</label>
      <input id="context" value="Lithium battery materials, CIF" />

      <label>Horizon days</label>
      <input id="horizon" value="90" />

      <label>Language (en/zh)</label>
      <input id="lang" value="en" />

      <div style="height:12px"></div>
      <div class="row">
        <button class="btn btn2" onclick="runAnalyze()">Run analyze now</button>
        <button class="btn" onclick="copyToken()">Copy token</button>
      </div>

      <div style="height:12px"></div>
      <div><b>Response</b></div>
      <pre id="out">Ready.</pre>
    </div>

    <div style="height:16px"></div>
    <div class="card">
      <div><b>How to call with Hoppscotch / Postman</b></div>
      <p class="muted">Set header <code>Authorization: Bearer &lt;token&gt;</code> (or just the token) and POST JSON to <code>/analyze</code>.</p>
      <pre>POST {str(Request).split("'")[1] if False else ""}https://global-risk-api.onrender.com/analyze
Content-Type: application/json
Authorization: Bearer {rec.token}

{{ "route":"Shanghai -> Tokyo", "context":"Lithium battery materials, CIF", "horizon_days":90, "language":"en" }}</pre>
    </div>
  </div>

<script>
  // Store token for later usage
  const token = {json.dumps(rec.token)};
  localStorage.setItem("SCRS_PAID_TOKEN", token);

  function copyToken() {{
    navigator.clipboard.writeText(token);
    alert("Token copied.");
  }}

  async function runAnalyze() {{
    const route = document.getElementById("route").value;
    const context = document.getElementById("context").value;
    const horizon_days = parseInt(document.getElementById("horizon").value || "90", 10);
    const language = document.getElementById("lang").value || "en";

    const payload = {{ route, context, horizon_days, language }};

    document.getElementById("out").textContent = "Calling /analyze ...";
    try {{
      const resp = await fetch("/analyze", {{
        method: "POST",
        headers: {{
          "Content-Type": "application/json",
          "Authorization": "Bearer " + token
        }},
        body: JSON.stringify(payload)
      }});
      const txt = await resp.text();
      document.getElementById("out").textContent = txt;
    }} catch (e) {{
      document.getElementById("out").textContent = "Error: " + (e && e.message ? e.message : String(e));
    }}
  }}

  // Optional: auto-run
  // runAnalyze();
</script>
</body>
</html>
"""
    return HTMLResponse(content=html, status_code=200)


@app.get("/cancel", response_class=HTMLResponse)
def cancel() -> str:
    return "<h3>Payment canceled.</h3>"


@app.post("/create-checkout-session", response_class=JSONResponse)
async def create_checkout_session(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> dict:
    """
    OPTIONAL endpoint.
    If you later want your own pay button from your landing page, call this to create a Stripe session.
    Protected by X-API-Key (master key) to prevent abuse.
    """
    if not x_api_key or (API_KEYS and x_api_key.strip() not in API_KEYS):
        raise HTTPException(status_code=401, detail="Unauthorized. Provide valid X-API-Key.")

    if stripe is None:
        raise HTTPException(status_code=500, detail="Stripe library not installed")

    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=500, detail="STRIPE_SECRET_KEY not configured")

    if not STRIPE_PRICE_ID:
        raise HTTPException(status_code=500, detail="STRIPE_PRICE_ID not configured")

    stripe.api_key = STRIPE_SECRET_KEY

    body = await request.json()
    # allow override quantity (default 1)
    qty = int(body.get("quantity") or 1)
    qty = max(1, min(10, qty))

    try:
        session = stripe.checkout.Session.create(
            mode="payment",
            line_items=[{"price": STRIPE_PRICE_ID, "quantity": qty}],
            success_url=SUCCESS_URL,
            cancel_url=CANCEL_URL,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Stripe error: {str(e)}")

    return {
        "status": "ok",
        "checkout_url": session.url,
        "session_id": session.id,
    }


# =========================
# Global error handler (optional)
# =========================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Avoid leaking secrets; keep it tight.
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})


# Note: Render uses `uvicorn app:app ...` so no __main__ needed.
