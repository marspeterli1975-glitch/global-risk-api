# app.py
import os
import time
import json
import hashlib
import traceback
from typing import Optional, List, Literal, Dict, Any

import requests
import xml.etree.ElementTree as ET

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from openai import OpenAI


# ----------------------------
# App init
# ----------------------------
app = FastAPI(title="Global Risk API", version="0.3.0-llm")

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
        "User-Agent": "global-risk-api/0.3.0 (+https://global-risk-api.onrender.com)",
        "Accept": "*/*",
    }
)

_CACHE: Dict[str, Dict[str, Any]] = {}


# ----------------------------
# Models
# ----------------------------
class AnalyzeRequest(BaseModel):
    location: str = Field(..., examples=["Tokyo"])
    language: Literal["en", "zh"] = Field("en", examples=["en"])


class AnalyzeResponse(BaseModel):
    ok: bool
    location: str
    language: str
    risk_score: int
    summary: str
    factors: List[Dict[str, Any]]
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
    return {"status": "ok", "version": "0.3.0-llm"}


@app.post("/engine/analyze", response_model=AnalyzeResponse)
def engine_analyze(req: AnalyzeRequest, request: Request):
    require_api_key(request)

    t0 = time.time()
    out = analyze_risk(req.location, req.language)
    out["meta"]["latency_ms"] = int((time.time() - t0) * 1000)
    return out


# ----------------------------
# Engine (RSS -> LLM -> Scores)
# ----------------------------
def analyze_risk(location: str, language: str) -> dict:
    data_mode = os.getenv("DATA_MODE", "live_rss_llm")
    ttl = int(os.getenv("CACHE_TTL_SECONDS", "900"))

    cache_key = _cache_key(location=location, language=language, mode=data_mode)
    cached = _cache_get(cache_key, ttl)
    if cached:
        return cached

    max_signals = int(os.getenv("MAX_SIGNALS", "12"))

    signals = collect_signals_rss(location, limit=max_signals)

    # LLM structured extraction
    events = llm_extract_events(location, language, signals)

    # Score aggregation
    factors = score_from_events(events, signals)

    risk_score = int(round(sum(f["score"] for f in factors) / max(1, len(factors))))
    risk_score = max(0, min(100, risk_score))

    summary = generate_summary(location, language, risk_score, factors, events)

    payload = {
        "ok": True,
        "location": location,
        "language": language,
        "risk_score": risk_score,
        "summary": summary,
        "factors": factors,
        "meta": {
            "engine_version": "0.3.0-llm",
            "data_mode": data_mode,
            "signals_count": len(signals),
            "events_count": len(events),
            "llm_model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
            "latency_ms": 0,
        },
    }

    _cache_set(cache_key, payload)
    return payload


def collect_signals_rss(location: str, limit: int = 12) -> List[Dict[str, Any]]:
    q = location.strip()
    url = (
        "https://news.google.com/rss/search?"
        f"q={requests.utils.quote(q)}%20when%3A7d&hl=en-US&gl=US&ceid=US:en"
    )
    return fetch_rss(url, source="Google News RSS", limit=limit)


def fetch_rss(url: str, source: str, limit: int = 10) -> List[Dict[str, Any]]:
    resp = SESSION.get(url, timeout=25)
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


# ----------------------------
# LLM extraction
# ----------------------------
def llm_extract_events(location: str, language: str, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        # No key: fallback to empty events (still works but less useful)
        return []

    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)

    # Compact input to control cost
    items = []
    for i, s in enumerate(signals):
        items.append(
            {
                "id": i,
                "title": s.get("title", ""),
                "source": s.get("source", ""),
                "published_at": s.get("published_at", ""),
                "url": s.get("url", ""),
            }
        )

    sys = (
        "You are a risk analyst. Extract structured risk events from news headlines. "
        "Return ONLY valid JSON. No markdown."
    )

    # We ask for structured output with simple schema.
    # category must be one of: macro, policy, security, logistics, finance
    # impact: -1..1 (negative is worse), probability: 0..1, severity: 0..100
    user = {
        "task": "extract_risk_events",
        "location": location,
        "language": language,
        "categories": ["macro", "policy", "security", "logistics", "finance"],
        "input_items": items,
        "output_schema": {
            "events": [
                {
                    "item_id": "int",
                    "category": "macro|policy|security|logistics|finance",
                    "event": "short string",
                    "impact": "number -1..1",
                    "probability": "number 0..1",
                    "severity": "int 0..100",
                    "rationale": "short string"
                }
            ]
        },
        "rules": [
            "Use title only; do not browse the URL.",
            "If headline is not about risk, either omit it or set severity low.",
            "Keep rationale under 25 words.",
        ],
    }

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        temperature=0.2,
    )

    content = resp.choices[0].message.content or ""
    data = _safe_json_loads(content)
    if not data or "events" not in data or not isinstance(data["events"], list):
        return []

    # sanitize events
    clean: List[Dict[str, Any]] = []
    for e in data["events"]:
        try:
            item_id = int(e.get("item_id"))
            category = str(e.get("category", "")).strip()
            if category not in {"macro", "policy", "security", "logistics", "finance"}:
                continue
            event = str(e.get("event", "")).strip()[:200]
            impact = float(e.get("impact", 0.0))
            prob = float(e.get("probability", 0.5))
            sev = int(e.get("severity", 0))
            rationale = str(e.get("rationale", "")).strip()[:220]

            impact = max(-1.0, min(1.0, impact))
            prob = max(0.0, min(1.0, prob))
            sev = max(0, min(100, sev))

            clean.append(
                {
                    "item_id": item_id,
                    "category": category,
                    "event": event,
                    "impact": impact,
                    "probability": prob,
                    "severity": sev,
                    "rationale": rationale,
                }
            )
        except Exception:
            continue

    return clean


def _safe_json_loads(s: str) -> Optional[dict]:
    s = s.strip()
    if not s:
        return None
    # Try direct parse
    try:
        return json.loads(s)
    except Exception:
        pass
    # Try to salvage JSON substring
    lb = s.find("{")
    rb = s.rfind("}")
    if lb != -1 and rb != -1 and rb > lb:
        try:
            return json.loads(s[lb : rb + 1])
        except Exception:
            return None
    return None


# ----------------------------
# Scoring from events
# ----------------------------
def score_from_events(events: List[Dict[str, Any]], signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert extracted events into factor scores (0..100) with explainable evidence.
    Score formula (simple & explainable):
      category_score = clamp( baseline + sum( severity * probability * max(0, -impact) / 10 ), 0..100 )
    impact negative => risk; positive => reduce risk
    """
    by_cat: Dict[str, List[Dict[str, Any]]] = {"macro": [], "policy": [], "security": [], "logistics": [], "finance": []}
    for e in events:
        by_cat[e["category"]].append(e)

    # Baselines: conservative
    baseline = {"macro": 45, "policy": 40, "security": 45, "logistics": 40, "finance": 42}

    factors: List[Dict[str, Any]] = []
    for cat, es in by_cat.items():
        score = baseline.get(cat, 40)

        # accumulate risk contributions
        contrib = 0.0
        for e in es:
            sev = float(e.get("severity", 0))
            prob = float(e.get("probability", 0.5))
            impact = float(e.get("impact", 0.0))
            risk_dir = max(0.0, -impact)  # only negative impact adds risk
            contrib += (sev * prob * risk_dir) / 10.0

        score = int(round(max(0.0, min(100.0, score + contrib))))

        evidence = build_evidence_from_events(cat, es, signals, max_items=3)

        # Only output the 3 core factors to keep response stable with your current UI
        # If you want all 5 categories, tell me and I'll expose all.
        if cat in {"macro", "policy", "security"}:
            factors.append({"name": cat, "score": score, "evidence": evidence})

    # Ensure order
    order = {"macro": 0, "policy": 1, "security": 2}
    factors.sort(key=lambda x: order.get(x["name"], 9))
    return factors


def build_evidence_from_events(category: str, events: List[Dict[str, Any]], signals: List[Dict[str, Any]], max_items: int = 3) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    # rank events by severity*probability
    ranked = sorted(events, key=lambda e: float(e.get("severity", 0)) * float(e.get("probability", 0.5)), reverse=True)

    for e in ranked[:max_items]:
        item_id = e.get("item_id")
        s = signals[item_id] if isinstance(item_id, int) and 0 <= item_id < len(signals) else None
        if not s:
            continue
        title = s.get("title", "")
        url = s.get("url", "")
        src = s.get("source", "")
        pub = s.get("published_at", None)

        out.append(
            {
                "title": title,
                "source": src,
                "url": url,
                "published_at": pub,
                "event": e.get("event"),
                "severity": e.get("severity"),
                "probability": e.get("probability"),
                "impact": e.get("impact"),
                "rationale": e.get("rationale"),
            }
        )

    # fallback: if LLM produced nothing for this category, show top headlines
    if not out:
        for s in signals[:max_items]:
            out.append(
                {
                    "title": s.get("title", ""),
                    "source": s.get("source", ""),
                    "url": s.get("url", ""),
                    "published_at": s.get("published_at"),
                    "event": None,
                    "severity": 0,
                    "probability": 0,
                    "impact": 0,
                    "rationale": "No structured events extracted; showing context headlines.",
                }
            )
    return out


def generate_summary(location: str, language: str, risk_score: int, factors: List[Dict[str, Any]], events: List[Dict[str, Any]]) -> str:
    top = sorted(events, key=lambda e: float(e.get("severity", 0)) * float(e.get("probability", 0.5)), reverse=True)[:3]
    if language == "zh":
        brief = "；".join([f"{t['category']}：{t['event']}" for t in top]) if top else "暂无高置信度事件。"
        bullets = "；".join([f"{f['name']}={f['score']}" for f in factors])
        return f"{location} 风险概览：综合评分 {risk_score}/100。分项（越高风险越高）：{bullets}。主要风险事件：{brief}。"
    else:
        brief = "; ".join([f"{t['category']}: {t['event']}" for t in top]) if top else "No high-confidence events extracted."
        bullets = ", ".join([f"{f['name']}={f['score']}" for f in factors])
        return f"Risk overview for {location}: overall score {risk_score}/100. Factor scores: {bullets}. Key events: {brief}."


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
