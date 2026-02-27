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

APP_VERSION = "0.3.0-llm"

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
    # comma separated
    return [x.strip() for x in raw.split(",") if x.strip()]

def _require_app_key(req: FastAPIRequest) -> str:
    auth = req.headers.get("Authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing_api_key")
    token = auth.split(" ", 1)[1].strip()
    keys = _get_app_api_keys()
    if not keys:
        # server misconfig
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
    # Google News RSS query
    # We keep it simple: query by location keyword; you can extend later.
    q = quote(location)
    # language is kept as informational in this RSS; GN RSS doesn’t strictly honor it always.
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
    # If not enough, just return top items as placeholder evidence
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

    # Keep payload small & deterministic
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

    # Minimal validation / normalization
    if "risk_score" not in data or "factors" not in data:
        raise RuntimeError("LLM response missing fields")
    return data

# -----------------------------
# FastAPI models
# -----------------------------
class AnalyzeRequest(BaseModel):
    location: str = Field(..., min_length=1)
    language: str = Field("en")

app = FastAPI(title="Global Risk API", version=APP_VERSION)

@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "version": APP_VERSION,
        "openai_configured": bool((os.getenv("OPENAI_API_KEY") or "").strip()),
    }

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
            # In this project, Route A = RSS. Keep strict.
            data_mode = "live_rss"

        signals = fetch_signals_live_rss(location, language, cache_ttl=cache_ttl, max_signals=max_signals)

        # If no signals, fallback to empty keyword scoring
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

        # LLM first
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
        # Hard fallback: keyword scoring so API stays up
        try:
            signals = fetch_signals_live_rss(location, language, cache_ttl=cache_ttl, max_signals=max_signals)
        except Exception:
            signals = []

        factors = score_factors_keyword(location, signals)

        # overall risk: avg factors
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
