import os
import json
import time
import hmac
import math
import hashlib
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request as FastAPIRequest, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, conint, confloat

import stripe

APP_VERSION = "0.4.0-scrs-freemium"

# -----------------------------
# Env helpers
# -----------------------------
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except Exception:
        return default

def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

def _get_app_api_keys() -> List[str]:
    """
    APP_API_KEYS supports:
      - single key: "abc"
      - multiple keys: "abc,def,ghi"
      - JSON array: ["abc","def"]
    """
    raw = (os.getenv("APP_API_KEYS") or "").strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            arr = json.loads(raw)
            return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
    return [x.strip() for x in raw.split(",") if x.strip()]

def _require_app_key(req: FastAPIRequest) -> str:
    auth = req.headers.get("Authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing_api_key")
    token = auth.split(" ", 1)[1].strip()
    keys = _get_app_api_keys()
    if not keys:
        raise HTTPException(status_code=500, detail="APP_API_KEYS is not set on server.")
    if token not in keys:
        raise HTTPException(status_code=401, detail="invalid_api_key")
    return token

def _payment_mode() -> str:
    return (os.getenv("PAYMENT_MODE") or "stripe").strip().lower()

def _stripe_configured() -> bool:
    return bool((os.getenv("STRIPE_SECRET_KEY") or "").strip())

# -----------------------------
# Stripe helpers
# -----------------------------
def _stripe_init() -> None:
    sk = (os.getenv("STRIPE_SECRET_KEY") or "").strip()
    if not sk:
        raise RuntimeError("STRIPE_SECRET_KEY not set")
    stripe.api_key = sk

def verify_stripe_session_paid(session_id: str) -> Dict[str, Any]:
    """
    Returns:
      { ok: bool, paid: bool, session_id: str, amount_total: int|None, currency: str|None, customer_email: str|None }
    """
    _stripe_init()

    try:
        # Expand customer_details for email; amount_total is in cents
        session = stripe.checkout.Session.retrieve(
            session_id,
            expand=["customer_details"]
        )

        paid = (session.get("payment_status") == "paid")
        amount_total = session.get("amount_total")
        currency = session.get("currency")
        customer_email = None
        cd = session.get("customer_details") or {}
        if isinstance(cd, dict):
            customer_email = cd.get("email")

        return {
            "ok": True,
            "paid": bool(paid),
            "session_id": session_id,
            "amount_total": amount_total,
            "currency": currency,
            "customer_email": customer_email,
        }
    except Exception as e:
        return {
            "ok": False,
            "paid": False,
            "session_id": session_id,
            "error": str(e),
        }

# -----------------------------
# SCRS engine (MVP deterministic)
# -----------------------------
RISK_DIMENSIONS = [
    "geopolitics",
    "macro_fx",
    "port_congestion",
    "customs_trade",
    "weather_climate",
    "security_crime",
    "supplier_disruption",
    "compliance_regulatory",
]

DIM_WEIGHTS = {
    "geopolitics": 0.16,
    "macro_fx": 0.12,
    "port_congestion": 0.14,
    "customs_trade": 0.12,
    "weather_climate": 0.12,
    "security_crime": 0.10,
    "supplier_disruption": 0.14,
    "compliance_regulatory": 0.10,
}

def _clamp(n: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, n))

def _hash_to_unit_float(s: str) -> float:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    # take 12 hex chars -> int
    v = int(h[:12], 16)
    return (v % 10_000_000) / 10_000_000.0

def _time_window_bucket(days: int = 30) -> str:
    # Rolling bucket changes over time but stable within the bucket
    t = int(time.time())
    bucket = t // (days * 24 * 3600)
    return str(bucket)

def scrs_band(score_0_100: int) -> str:
    # 5-level system
    # 0-19: Very Low, 20-39: Low, 40-59: Medium, 60-79: High, 80-100: Very High
    if score_0_100 <= 19:
        return "very_low"
    if score_0_100 <= 39:
        return "low"
    if score_0_100 <= 59:
        return "medium"
    if score_0_100 <= 79:
        return "high"
    return "very_high"

def attention_amplification(base_score: float, origin: str, destination: str) -> Tuple[float, str]:
    """
    MVP: amplify score a bit if "attention signals" are high.
    No external fetch. Deterministic, time-bucketed.
    """
    bucket = _time_window_bucket(days=7)  # "attention" shifts faster
    u = _hash_to_unit_float(f"attn|{bucket}|{origin}|{destination}")
    # attention factor between 0.0 and 1.0
    attn = _clamp(u, 0.0, 1.0)

    # amplify up to +8 points when attention is high
    amplified = base_score + attn * 8.0

    if attn >= 0.75:
        note = "Attention elevated: recent signals density is higher than typical for similar routes."
    elif attn >= 0.45:
        note = "Attention moderate: some recent signals may be affecting short-term volatility."
    else:
        note = "Attention normal: no notable short-term amplification detected."

    return amplified, note

def compute_scrs_full(
    origin: str,
    destination: str,
    margin_pct: float,
    credit_days: int,
    currency: str
) -> Dict[str, Any]:
    """
    MVP relative model:
      - Baseline = env SCRS_BASELINE_MEAN (default 50)
      - Score derived from weighted deterministic per-dimension signals + attention layer
      - 30–90 day dynamic relative index: uses time bucket in hashing
      - Profit band simulation: linear model (MVP)
    """
    baseline_mean = _env_float("SCRS_BASELINE_MEAN", 50.0)

    # Use 30-day bucket for "index window"
    bucket = _time_window_bucket(days=30)
    route_key = f"{origin.strip().lower()}|{destination.strip().lower()}|{bucket}"

    breakdown: List[Dict[str, Any]] = []
    weighted_sum = 0.0

    for dim in RISK_DIMENSIONS:
        u = _hash_to_unit_float(f"{dim}|{route_key}")
        # map to roughly 25..85, centered around 55
        raw = 25.0 + u * 60.0
        # mild penalty for long credit days & thin margin (risk exposure)
        credit_adj = _clamp((credit_days - 30) / 120.0, 0.0, 1.0) * 6.0   # up to +6
        margin_adj = _clamp((20.0 - margin_pct) / 20.0, 0.0, 1.0) * 6.0  # up to +6
        score = _clamp(raw + credit_adj + margin_adj, 0.0, 100.0)

        w = DIM_WEIGHTS.get(dim, 0.0)
        weighted_sum += score * w

        breakdown.append({
            "dimension": dim,
            "score": int(round(score)),
            "weight": w,
            "note": "MVP deterministic signal proxy (no external sources)."
        })

    base_total = _clamp(weighted_sum, 0.0, 100.0)

    # attention layer
    total_with_attn, attn_note = attention_amplification(base_total, origin, destination)
    scrs_total = int(round(_clamp(total_with_attn, 0.0, 100.0)))
    band = scrs_band(scrs_total)

    deviation_pct = 0.0
    if baseline_mean > 0:
        deviation_pct = ((scrs_total - baseline_mean) / baseline_mean) * 100.0

    # Profit range simulation (linear MVP)
    # intuition: higher risk compresses realized margin, and longer credit days increases exposure.
    risk_drag = scrs_total / 100.0  # 0..1
    credit_drag = _clamp(credit_days / 180.0, 0.0, 1.0)  # 0..1

    # realized margin range: margin_pct +/- something, but skewed downward with risk
    # best case slightly below nominal, worst case more compressed
    best_case = margin_pct - (risk_drag * 1.5) - (credit_drag * 0.8)
    worst_case = margin_pct - (risk_drag * 7.0) - (credit_drag * 2.0)

    best_case = float(_clamp(best_case, -50.0, 80.0))
    worst_case = float(_clamp(worst_case, -50.0, 80.0))

    profit_range_pct = {
        "best_case": round(best_case, 2),
        "worst_case": round(worst_case, 2),
        "model": "linear_mvp",
        "notes": "Linear sensitivity to SCRS risk and credit-days exposure (MVP)."
    }

    return {
        "scrs_total": scrs_total,
        "scrs_band": band,
        "baseline_mean": round(baseline_mean, 2),
        "deviation_pct": round(deviation_pct, 2),
        "breakdown": breakdown,
        "profit_range_pct": profit_range_pct,
        "attention_note": attn_note,
        "assumptions": {
            "index_type": "relative",
            "window_days": "30–90 (MVP uses 30-day bucket + 7-day attention proxy)",
            "baseline_definition": "statistical mean of sample route set (MVP uses configured baseline_mean)",
            "data_policy": "No external data sources in MVP; deterministic proxy signals only.",
            "no_advisory": True
        }
    }

def fuzz_score(score: int, origin: str, destination: str, delta: int) -> int:
    """
    Freemium: return score ±delta (deterministic, so repeated calls stable within time bucket).
    """
    bucket = _time_window_bucket(days=7)
    u = _hash_to_unit_float(f"fuzz|{bucket}|{origin}|{destination}")
    shift = int(round((u * 2 - 1) * delta))  # -delta .. +delta
    return int(_clamp(score + shift, 0, 100))

# -----------------------------
# FastAPI models
# -----------------------------
class ScrsScanRequest(BaseModel):
    origin: str = Field(..., min_length=1, max_length=80)
    destination: str = Field(..., min_length=1, max_length=80)
    margin_pct: confloat(ge=-50.0, le=80.0) = Field(..., description="Gross margin percentage")
    credit_days: conint(ge=0, le=365) = Field(..., description="Credit terms in days")
    currency: str = Field("USD", min_length=3, max_length=8)

    # optional: client can pass Stripe session_id for paid unlock
    session_id: Optional[str] = Field(None, description="Stripe Checkout Session ID for paid unlock")

class SuccessResponse(BaseModel):
    ok: bool
    paid: bool
    session_id: Optional[str] = None
    amount_total: Optional[int] = None
    currency: Optional[str] = None
    customer_email: Optional[str] = None
    error: Optional[str] = None

# -----------------------------
# App init
# -----------------------------
app = FastAPI(
    title="SCRS – Supply Chain Route Risk Exposure Scan",
    version=APP_VERSION
)

# CORS for landing page integration
allowed_origins = (os.getenv("ALLOWED_ORIGINS") or "*").strip()
origins = ["*"] if allowed_origins == "*" else [x.strip() for x in allowed_origins.split(",") if x.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

DISCLAIMER_TEXT = (
    "SCRS provides a relative risk exposure index for informational purposes only. "
    "It is a product tool, not consulting or advisory. "
    "Outputs do not constitute financial, legal, compliance, or operational advice. "
    "Users remain responsible for decisions, validation, and outcomes."
)

# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "version": APP_VERSION,
        "payment_mode": _payment_mode(),
        "stripe_configured": _stripe_configured(),
    }

@app.get("/success", response_model=SuccessResponse)
async def success(session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Stripe Payment Link redirect target.
    Stripe will append: ?session_id={CHECKOUT_SESSION_ID}
    """
    if not session_id:
        return {"ok": False, "paid": False, "error": "missing_session_id"}

    if _payment_mode() != "stripe":
        return {"ok": False, "paid": False, "session_id": session_id, "error": "payment_mode_not_stripe"}

    if not _stripe_configured():
        return {"ok": False, "paid": False, "session_id": session_id, "error": "stripe_not_configured"}

    res = verify_stripe_session_paid(session_id)
    # Keep response minimal and JSON-only
    return res

@app.post("/scrs/scan")
async def scrs_scan(req: FastAPIRequest, body: ScrsScanRequest) -> Dict[str, Any]:
    """
    Main SCRS API.
    - Requires APP API key.
    - Freemium:
        unpaid -> {scrs_band, scrs_total (fuzz ±3), baseline_mean, deviation_pct, attention_note, disclaimer, assumptions}
        paid   -> full breakdown + profit_range_pct
    """
    _require_app_key(req)

    origin = body.origin.strip()
    destination = body.destination.strip()
    margin_pct = float(body.margin_pct)
    credit_days = int(body.credit_days)
    currency = (body.currency or "USD").strip().upper()
    session_id = (body.session_id or "").strip() or None

    full = compute_scrs_full(origin, destination, margin_pct, credit_days, currency)

    # Decide paid/unpaid
    paid = False
    payment_meta: Dict[str, Any] = {"mode": _payment_mode()}

    if _payment_mode() == "stripe" and session_id:
        if _stripe_configured():
            v = verify_stripe_session_paid(session_id)
            paid = bool(v.get("paid"))
            payment_meta.update({
                "session_id": session_id,
                "verified": bool(v.get("ok")),
                "paid": bool(v.get("paid")),
            })
        else:
            payment_meta.update({"session_id": session_id, "verified": False, "paid": False, "error": "stripe_not_configured"})
    else:
        payment_meta.update({"paid": False})

    # Freemium: unpaid response (no breakdown, no profit_range_pct)
    fuzz_delta = _env_int("SCRS_FUZZ_DELTA", 3)

    if not paid:
        fuzzed_total = fuzz_score(int(full["scrs_total"]), origin, destination, fuzz_delta)

        return {
            "ok": True,
            "paid": False,
            "input": {
                "origin": origin,
                "destination": destination,
                "margin_pct": margin_pct,
                "credit_days": credit_days,
                "currency": currency,
            },
            "output": {
                "scrs_band": full["scrs_band"],
                "scrs_total": fuzzed_total,  # fuzzed
                "baseline_mean": full["baseline_mean"],
                "deviation_pct": full["deviation_pct"],
                "attention_note": full["attention_note"],
                "assumptions": full["assumptions"],
                "disclaimer": DISCLAIMER_TEXT,
            },
            "meta": {
                "engine_version": APP_VERSION,
                "freemium": {"fuzz_delta": fuzz_delta},
                "payment": payment_meta,
            }
        }

    # Paid: full response
    return {
        "ok": True,
        "paid": True,
        "input": {
            "origin": origin,
            "destination": destination,
            "margin_pct": margin_pct,
            "credit_days": credit_days,
            "currency": currency,
        },
        "output": {
            "scrs_total": full["scrs_total"],
            "scrs_band": full["scrs_band"],
            "baseline_mean": full["baseline_mean"],
            "deviation_pct": full["deviation_pct"],
            "breakdown": full["breakdown"],
            "profit_range_pct": full["profit_range_pct"],
            "attention_note": full["attention_note"],
            "assumptions": full["assumptions"],
            "disclaimer": DISCLAIMER_TEXT,
        },
        "meta": {
            "engine_version": APP_VERSION,
            "payment": payment_meta,
        }
    }
