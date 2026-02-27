# app.py
import os
import time
import json
import hashlib
import traceback
from typing import Optional, List, Literal, Dict, Any
from datetime import datetime, timezone

import requests
import xml.etree.ElementTree as ET

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ----------------------------
# App init
# ----------------------------
app = FastAPI(title="Global Risk API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://hoppscotch.io",
        "https://hoppscotch.com",
        "http://localhost",
        "http://localhost:3000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "global-risk-api/0.2.0 (+https://global-risk-api.onrender.com)",
        "Accept": "*/*",
    }
)

# Simple in-memory cache (good enough for Render single instance; resets on redeploy/sleep)
_CACHE: Dict[str, Dict[str, Any]] = {}


# ----------------------------
# Models
# ----------------------------
class AnalyzeRequest(BaseModel):
    location: str = Field(..., examples=["Tokyo"])
    language: Literal["en", "zh"] = Field("en", examples=["en"])


class EvidenceItem(BaseModel):
    title: str
    source: str
    url: str
    published_at: Optional[str] = None


class Factor(BaseModel):
    name: str
    score: int
    evidence: List[EvidenceItem]


class AnalyzeResponse(BaseModel):
    ok: bool
    location: str
    language: str
    risk_score: int
    summary: str
    factors: List[Factor]
    meta: dict


# ----------------------------
# API Key Gate
# ----------------------------
def _extract_api_key(request: Request) -> Optional[str]:
    api_key = request.headers.get("X-API-Key")
    if api_key and api_key.strip():
        return api_key.strip()
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
        return token if token else None
    return None


def _load_allowed_keys() -> set[str]:
    raw = os.getenv("APP_API_KEYS", "").strip()
    if not raw:
        return set()
    return {k.strip() for k in raw.split(",") if k.strip()}


def require_api_key(request: Request) -> str:
    api_key = _extract_api_key(request)
    if not api_key:
        raise HTTPException(status_code=401, detail="missing_api_key")
    allowed = _load_allowed_keys()
    if not allowed:
        raise HTTPException(status_code=503, detail="api_key_gate_not_configured")
    if api_key not in allowed:
        raise HTTPException(status_code=403, detail="invalid_api_key")
    return api_key


# ----------------------------
# Error handling
# ----------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    debug = os.getenv("DEBUG", "0") == "1"
    payload = {"detail": "internal_error", "error": str(exc)}
    if debug:
        payload["traceback_tail"] = traceback.format_exc().splitlines()[-40:]
    return JSONResponse(status_code=500, content=payload)


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/engine/analyze", response_model=AnalyzeResponse)
def engine_analyze(req: AnalyzeRequest, request: Request):
    require_api_key(request)

    t0 = time.time()
    result = analyze_risk(req.location, req.language)
    result["meta"]["latency_ms"] = int((time.time() - t0) * 1000)
    return result


# ----------------------------
# Risk engine (Live RSS)
# ----------------------------
def analyze_risk(location: str, language: str) -> dict:
    data_mode = os.getenv("DATA_MODE", "live_rss")
    ttl = int(os.getenv("CACHE_TTL_SECONDS", "900"))

    cache_key = _cache_key(location=location, language=language, mode=data_mode)
    cached = _cache_get(cache_key, ttl)
    if cached:
        return cached

    # 1) collect signals
    signals = collect_signals_rss(location)

    # 2) extract & score factors (simple heuristic for now)
    factors = score_factors(location, signals)

    # 3) aggregate risk score
    risk_score = int(round(sum(f["score"] for f in factors) / max(1, len(factors))))

    # 4) generate summary (optionally with OpenAI)
    summary = generate_summary(location, language, risk_score, factors)

    payload = {
        "ok": True,
        "location": location,
        "language": language,
        "risk_score": max(0, min(100, risk_score)),
        "summary": summary,
        "factors": factors,
        "meta": {
            "engine_version": "0.2.0",
            "data_mode": data_mode,
            "signals_count": len(signals),
            "latency_ms": 0,
        },
    }

    _cache_set(cache_key, payload)
    return payload


def collect_signals_rss(location: str) -> List[Dict[str, Any]]:
    """
    RSS sources:
      - Google News RSS for broad coverage (location-based query)
      - You can add more official/vertical sources later
    """
    q = location.strip()
    # Google News RSS (query)
    url = (
        "https://news.google.com/rss/search?"
        f"q={requests.utils.quote(q)}%20when%3A7d&hl=en-US&gl=US&ceid=US:en"
    )

    items = fetch_rss(url, source="Google News RSS", limit=12)
    return items


def fetch_rss(url: str, source: str, limit: int = 10) -> List[Dict[str, Any]]:
    resp = SESSION.get(url, timeout=20)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    channel = root.find("channel")
    if channel is None:
        return []

    out: List[Dict[str, Any]] = []
    for item in channel.findall("item")[:limit]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        out.append(
            {
                "title": title,
                "url": link,
                "published_at": pub_date,
                "source": source,
            }
        )
    return out


def score_factors(location: str, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Simple scoring (MVP):
      - macro: baseline + "inflation / recession / GDP / FX" keywords
      - policy: "sanction / regulation / law / ban / visa" keywords
      - security: "attack / crime / protest / earthquake / typhoon" keywords
    Each factor returns evidence items with URLs.
    """
    texts = [s.get("title", "").lower() for s in signals]

    macro_kw = ["inflation", "recession", "gdp", "fx", "currency", "interest rate", "oil", "energy"]
    policy_kw = ["sanction", "regulation", "law", "ban", "visa", "tariff", "restriction"]
    security_kw = ["attack", "crime", "protest", "earthquake", "typhoon", "flood", "explosion"]

    macro_hits = _count_hits(texts, macro_kw)
    policy_hits = _count_hits(texts, policy_kw)
    security_hits = _count_hits(texts, security_kw)

    # baseline scores
    macro_score = min(100, 45 + macro_hits * 8)
    policy_score = min(100, 40 + policy_hits * 10)
    security_score = min(100, 45 + security_hits * 10)

    macro_evi = build_evidence(signals, macro_kw, max_items=3)
    policy_evi = build_evidence(signals, policy_kw, max_items=3)
    security_evi = build_evidence(signals, security_kw, max_items=3)

    return [
        {"name": "macro", "score": macro_score, "evidence": macro_evi},
        {"name": "policy", "score": policy_score, "evidence": policy_evi},
        {"name": "security", "score": security_score, "evidence": security_evi},
    ]


def build_evidence(signals: List[Dict[str, Any]], keywords: List[str], max_items: int = 3) -> List[Dict[str, Any]]:
    out = []
    for s in signals:
        t = (s.get("title") or "").lower()
        if any(k in t for k in keywords):
            out.append(
                {
                    "title": s.get("title", ""),
                    "source": s.get("source", ""),
                    "url": s.get("url", ""),
                    "published_at": s.get("published_at"),
                }
            )
        if len(out) >= max_items:
            break
    # If no matches, still provide top headlines as “context”
    if not out:
        for s in signals[:max_items]:
            out.append(
                {
                    "title": s.get("title", ""),
                    "source": s.get("source", ""),
                    "url": s.get("url", ""),
                    "published_at": s.get("published_at"),
                }
            )
    return out


def generate_summary(location: str, language: str, risk_score: int, factors: List[Dict[str, Any]]) -> str:
    """
    If OPENAI_API_KEY exists, we can produce a cleaner summary.
    Otherwise return a deterministic summary.
    """
    if language == "zh":
        base = f"{location} 风险概览：综合评分 {risk_score}/100。"
        bullets = "；".join([f"{f['name']}={f['score']}" for f in factors])
        return base + f" 分项（越高风险越高）：{bullets}。证据来自公开新闻/RSS。"
    else:
        base = f"Risk overview for {location}: overall score {risk_score}/100."
        bullets = ", ".join([f"{f['name']}={f['score']}" for f in factors])
        return base + f" Factor scores (higher = riskier): {bullets}. Evidence sourced from public RSS/news."

# ----------------------------
# Cache utils
# ----------------------------
def _cache_key(**kwargs) -> str:
    raw = json.dumps(kwargs, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_get(key: str, ttl_seconds: int) -> Optional[dict]:
    rec = _CACHE.get(key)
    if not rec:
        return None
    if time.time() - rec["ts"] > ttl_seconds:
        _CACHE.pop(key, None)
        return None
    return rec["val"]


def _cache_set(key: str, val: dict):
    _CACHE[key] = {"ts": time.time(), "val": val}
