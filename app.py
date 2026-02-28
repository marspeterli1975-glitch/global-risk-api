import os
import json
import time
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote
from urllib.request import Request, urlopen
from email.utils import parsedate_to_datetime

from fastapi import FastAPI, Request as FastAPIRequest, HTTPException
from pydantic import BaseModel, Field

# OpenAI Python SDK v1.x
from openai import OpenAI

APP_VERSION = "0.3.1-scrs-freemium"

# -----------------------------
# Config helpers
# -----------------------------
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
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

# -----------------------------
# RSS fetch + cache
# -----------------------------
_CACHE: Dict[str, Tuple[float, Any]] = {}  # key -> (expires_at, value)

def _cache_get(key: str) -> Optional[Any]:
    now = time.time()
    item = _CACHE.get(key)
    if not item:
        return None
    exp, val = item
    if now >= exp:
        _CACHE.pop(key, None)
        return None
    return val

def _cache_set(key: str, val: Any, ttl: int) -> None:
    _CACHE[key] = (time.time() + ttl, val)

def _http_get(url: str, timeout: int = 15) -> bytes:
    req = Request(url, headers={"User-Agent": "global-risk-api/0.3"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()

def _google_news_rss_url(location: str, language: str) -> str:
    q = quote(location)
    return f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"

def _parse_rss_items(xml_bytes: bytes, max_items: int) -> List[Dict[str, Any]]:
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_bytes)
    channel = root.find("channel")
    if channel is None:
        return []

    out: List[Dict[str, Any]] = []
    for item in channel.findall("item")[:max_items]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        source = "Google News RSS"

        published_at = None
        if pub:
            try:
                dt = parsedate_to_datetime(pub)
                published_at = dt.isoformat()
            except Exception:
                published_at = pub

        out.append(
            {
                "title": title,
                "source": source,
                "url": link,
                "published_at": published_at,
            }
        )
    return out

def fetch_signals_live_rss(location: str, language: str, cache_ttl: int, max_signals: int) -> List[Dict[str, Any]]:
    ck = "rss:" + hashlib.sha1(f"{location}|{language}|{max_signals}".encode("utf-8")).hexdigest()
    cached = _cache_get(ck)
    if cached is not None:
        return cached

    url = _google_news_rss_url(location, language)
    xml_bytes = _http_get(url)
    items = _parse_rss_items(xml_bytes, max_items=max_signals)

    _cache_set(ck, items, ttl=cache_ttl)
    return items

# -----------------------------
# Fallback keyword scoring
# -----------------------------
def _count_hits(texts: List[str], keywords: List[str]) -> int:
    count = 0
    for t in texts:
        for k in keywords:
            if k in t:
                count += 1
    return count

def build_evidence(signals: List[Dict[str, Any]], keywords: List[str], max_items: int = 3) -> List[Dict[str, Any]]:
    kws = [k.lower() for k in keywords]
    out: List[Dict[str, Any]] = []
    for s in signals:
        t = (s.get("title") or "").lower()
        if any(k in t for k in kws):
            out.append(s)
        if len(out) >= max_items:
            break
    if not out:
        out = signals[:max_items]
    return out

def score_factors_keyword(location: str, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    texts = [(s.get("title") or "").lower() for s in signals]

    macro_kw = ["inflation", "recession", "gdp", "fx", "currency", "interest rate", "oil", "energy"]
    policy_kw = ["sanction", "regulation", "law", "ban", "visa", "tariff", "restriction"]
    security_kw = ["attack", "crime", "protest", "earthquake", "typhoon", "flood", "explosion"]

    macro_hits = _count_hits(texts, macro_kw)
    policy_hits = _count_hits(texts, policy_kw)
    security_hits = _count_hits(texts, security_kw)

    macro_score = min(100, 45 + macro_hits * 8)
    policy_score = min(100, 40 + policy_hits * 10)
    security_score = min(100, 45 + security_hits * 10)

    return [
        {"name": "macro", "score": macro_score, "evidence": build_evidence(signals, macro_kw, max_items=3)},
        {"name": "policy", "score": policy_score, "evidence": build_evidence(signals, policy_kw, max_items=3)},
        {"name": "security", "score": security_score, "evidence": build_evidence(signals, security_kw, max_items=3)},
    ]

# -----------------------------
# LLM scoring (Route A: RSS -> LLM)
# -----------------------------
def score_with_llm(location: str, language: str, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    model = (os.getenv("LLM_MODEL") or "gpt-4o-mini").strip()
    client = OpenAI(api_key=api_key)

    compact_signals = signals[: _env_int("MAX_SIGNALS", 12)]
    prompt = {
        "task": "You are a geopolitical & supply-chain risk analyst. Use the provided news signals to score risk for the location.",
        "location": location,
        "language": language,
        "signals": compact_signals,
        "output_schema": {
            "risk_score": "int 0-100",
            "summary": "short string",
            "factors": [
                {
                    "name": "macro|policy|security",
                    "score": "int 0-100",
                    "event": "string",
                    "severity": "low|medium|high",
                    "probability": "low|medium|high",
                    "impact": "low|medium|high",
                    "rationale": "1-2 sentences",
                    "evidence": [
                        {
                            "title": "string",
                            "source": "string",
                            "url": "string",
                            "published_at": "string|null"
                        }
                    ]
                }
            ]
        },
        "rules": [
            "Return STRICT JSON only, no markdown.",
            "Use only the provided signals; do not invent sources.",
            "If evidence is weak, keep scores closer to 40-60 and say uncertainty in rationale."
        ],
    }

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return STRICT JSON only."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        temperature=0.2,
    )

    content = resp.choices[0].message.content or ""
    data = json.loads(content)

    if "risk_score" not in data or "factors" not in data:
        raise RuntimeError("LLM response missing fields")
    return data

# -----------------------------
# FastAPI models (existing)
# -----------------------------
class AnalyzeRequest(BaseModel):
    location: str = Field(..., min_length=1)
    language: str = Field("en")

# -----------------------------
# SCRS models (new)
# -----------------------------
class SCRSScanRequest(BaseModel):
    origin: str = Field(..., min_length=2, max_length=80)
    destination: str = Field(..., min_length=2, max_length=80)
    margin_pct: float = Field(..., ge=-50.0, le=200.0)
    credit_days: int = Field(..., ge=0, le=365)
    currency: str = Field(..., min_length=3, max_length=6)
    window_days: int = Field(60, ge=30, le=90)
    stripe_session_id: Optional[str] = Field(None)

class SCRSProfitRange(BaseModel):
    min_pct: float
    max_pct: float

class SCRSAssumptions(BaseModel):
    model_type: str = "relative_index"
    baseline_definition: str = "statistical_mean_of_sample_route_set"
    window_days: int
    profit_model: str = "linear_mvp"
    attention_layer: str = "enabled_mvp"
    notes: List[str] = Field(default_factory=list)

class SCRSScanFullResponse(BaseModel):
    scrs_total: float = Field(..., ge=0, le=100)
    scrs_band: str
    baseline_mean: float = Field(..., ge=0, le=100)
    deviation_pct: float
    risk_breakdown: Dict[str, float]
    profit_range_pct: SCRSProfitRange
    attention_note: str
    assumptions: SCRSAssumptions
    disclaimer: str

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Global Risk API", version=APP_VERSION)

@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "version": APP_VERSION,
        "openai_configured": bool((os.getenv("OPENAI_API_KEY") or "").strip()),
        "payment_mode": (os.getenv("PAYMENT_MODE") or "stripe").strip(),
    }

# -----------------------------
# Existing endpoint: /engine/analyze
# -----------------------------
@app.post("/engine/analyze")
async def engine_analyze(req: FastAPIRequest, body: AnalyzeRequest) -> Dict[str, Any]:
    _require_app_key(req)

    location = body.location.strip()
    language = (body.language or "en").strip()

    data_mode = (os.getenv("DATA_MODE") or "live_rss").strip()
    cache_ttl = _env_int("CACHE_TTL_SECONDS", 900)
    max_signals = _env_int("MAX_SIGNALS", 12)
    debug = _env_bool("DEBUG", False)

    t0 = time.time()

    try:
        if data_mode != "live_rss":
            data_mode = "live_rss"

        signals = fetch_signals_live_rss(location, language, cache_ttl=cache_ttl, max_signals=max_signals)

        if not signals:
            factors = score_factors_keyword(location, [])
            return {
                "ok": True,
                "location": location,
                "language": language,
                "risk_score": 50,
                "summary": f"No fresh signals found for {location}. Returning neutral score.",
                "factors": factors,
                "meta": {
                    "engine_version": APP_VERSION,
                    "data_mode": data_mode,
                    "latency_ms": int((time.time() - t0) * 1000),
                },
            }

        llm_data = score_with_llm(location, language, signals)

        out = {
            "ok": True,
            "location": location,
            "language": language,
            "risk_score": int(llm_data.get("risk_score", 50)),
            "summary": llm_data.get("summary", ""),
            "factors": llm_data.get("factors", []),
            "meta": {
                "engine_version": APP_VERSION,
                "data_mode": data_mode,
                "model": (os.getenv("LLM_MODEL") or "gpt-4o-mini").strip(),
                "signals": len(signals),
                "latency_ms": int((time.time() - t0) * 1000),
            },
        }
        return out

    except Exception as e:
        try:
            signals = fetch_signals_live_rss(location, language, cache_ttl=cache_ttl, max_signals=max_signals)
        except Exception:
            signals = []

        factors = score_factors_keyword(location, signals)
        scores = [f.get("score", 50) for f in factors] or [50]
        risk_score = int(sum(scores) / len(scores))

        if debug:
            raise HTTPException(status_code=500, detail={"detail": "internal_error", "error": str(e)})

        return {
            "ok": True,
            "location": location,
            "language": language,
            "risk_score": risk_score,
            "summary": f"Fallback scoring for {location}. (LLM unavailable or failed.)",
            "factors": factors,
            "meta": {
                "engine_version": APP_VERSION,
                "data_mode": data_mode,
                "fallback": "keyword",
                "latency_ms": int((time.time() - t0) * 1000),
            },
        }

# -----------------------------
# SCRS engine (MVP Freemium)
# -----------------------------
SCRS_DISCLAIMER = (
    "SCRS is an automated relative risk exposure index generated from model assumptions and sample baselines. "
    "It is provided for informational purposes only and does not constitute consulting, advice, or a recommendation. "
    "Users are responsible for their own decisions and validation."
)

def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))

def _scrs_band(score: float) -> str:
    if score < 20: return "A"
    if score < 40: return "B"
    if score < 60: return "C"
    if score < 80: return "D"
    return "E"

def _blur_score(score: float) -> float:
    import random
    return round(_clamp(score + random.uniform(-3, 3)), 2)

def _baseline_mean(window_days: int) -> float:
    import random
    seed = int(os.getenv("SCRS_BASELINE_SEED", "42")) + int(window_days)
    rnd = random.Random(seed)
    sample_size = _env_int("SCRS_BASELINE_SAMPLE_SIZE", 200)
    samples = [rnd.uniform(30, 70) + (window_days - 60) * 0.05 for _ in range(sample_size)]
    mean = sum(samples) / len(samples)
    return round(_clamp(mean), 2)

def _compute_scrs_core(origin: str, destination: str, currency: str, window_days: int) -> Tuple[float, Dict[str, float], str]:
    import random
    seed = hash((origin.lower(), destination.lower(), currency.upper(), int(window_days))) & 0xFFFFFFFF
    rnd = random.Random(seed)

    breakdown = {
        "geopolitical": rnd.uniform(10, 90),
        "regulatory_trade": rnd.uniform(10, 85),
        "logistics_disruption": rnd.uniform(15, 90),
        "port_congestion": rnd.uniform(5, 80),
        "weather_climate": rnd.uniform(5, 85),
        "fx_volatility": rnd.uniform(5, 75),
        "supplier_concentration": rnd.uniform(5, 80),
        "security_fraud": rnd.uniform(5, 70),
    }
    breakdown = {k: round(_clamp(v), 2) for k, v in breakdown.items()}

    vals = list(breakdown.values())
    dispersion = (max(vals) - min(vals)) / 100.0
    window_factor = (window_days - 30) / 60.0
    attention_score = _clamp((dispersion * 60.0) + (window_factor * 10.0) + rnd.uniform(0, 15), 0, 100)
    amp = 1.0 + (attention_score / 100.0) * 0.18

    weights = {
        "geopolitical": 0.16,
        "regulatory_trade": 0.14,
        "logistics_disruption": 0.16,
        "port_congestion": 0.10,
        "weather_climate": 0.12,
        "fx_volatility": 0.10,
        "supplier_concentration": 0.12,
        "security_fraud": 0.10,
    }
    base_total = sum(breakdown[k] * w for k, w in weights.items())
    total = round(_clamp(base_total * amp), 2)

    attention_note = (
        f"Attention layer applied (score={attention_score:.1f}/100, multiplier={amp:.3f}). "
        "Index remains relative to baseline route set."
    )
    return total, breakdown, attention_note

def _profit_range_linear(margin_pct: float, credit_days: int, currency: str, scrs_total: float) -> SCRSProfitRange:
    currency = currency.upper().strip()
    currency_factor = {
        "USD": 1.00, "EUR": 1.02, "GBP": 1.03, "CNY": 0.98, "INR": 1.05,
        "AUD": 1.01, "JPY": 1.02, "SGD": 1.00, "HKD": 1.00, "CAD": 1.01,
    }.get(currency, 1.00)

    risk_drag = (scrs_total / 100.0) * 0.55 * currency_factor
    credit_drag = min(credit_days / 180.0, 1.0) * 0.20
    center = margin_pct * (1.0 - risk_drag - credit_drag)

    spread = abs(margin_pct) * (0.10 + (scrs_total / 100.0) * 0.20 + (credit_days / 365.0) * 0.10)
    return SCRSProfitRange(min_pct=round(center - spread, 2), max_pct=round(center + spread, 2))

def _is_paid(stripe_session_id: Optional[str]) -> bool:
    payment_mode = (os.getenv("PAYMENT_MODE") or "stripe").strip().lower()
    if payment_mode == "disabled":
        return True

    if not stripe_session_id:
        return False

    stripe_key = (os.getenv("STRIPE_SECRET_KEY") or "").strip()
    if not stripe_key:
        return False

    try:
        import stripe  # type: ignore
        stripe.api_key = stripe_key
        session = stripe.checkout.Session.retrieve(stripe_session_id)
        return getattr(session, "payment_status", None) == "paid"
    except Exception:
        return False

@app.post("/v1/scan")
async def scrs_scan(req: FastAPIRequest, body: SCRSScanRequest):
    _require_app_key(req)

    origin = body.origin.strip()
    destination = body.destination.strip()
    currency = body.currency.strip().upper()
    window_days = int(body.window_days)

    scrs_total, breakdown, attention_note = _compute_scrs_core(origin, destination, currency, window_days)
    band = _scrs_band(scrs_total)

    paid = _is_paid(body.stripe_session_id)

    if not paid:
        return {
            "preview": True,
            "scrs_band": band,
            "scrs_total": _blur_score(scrs_total),
            "message": "Full breakdown requires payment."
        }

    baseline_mean = _baseline_mean(window_days)
    deviation = 0.0 if baseline_mean <= 0 else round((scrs_total - baseline_mean) / baseline_mean * 100.0, 2)
    profit_range = _profit_range_linear(body.margin_pct, body.credit_days, currency, scrs_total)

    assumptions = SCRSAssumptions(
        window_days=window_days,
        notes=[
            "Baseline is the statistical mean of a sample route set (MVP simulated).",
            "Index is relative (not absolute) and designed for 30–90 day comparisons.",
            "Profit range is a linear MVP simulation and may not match realized outcomes."
        ],
    )

    full = SCRSScanFullResponse(
        scrs_total=scrs_total,
        scrs_band=band,
        baseline_mean=baseline_mean,
        deviation_pct=deviation,
        risk_breakdown=breakdown,
        profit_range_pct=profit_range,
        attention_note=attention_note,
        assumptions=assumptions,
        disclaimer=SCRS_DISCLAIMER,
    )

    return full.model_dump()
