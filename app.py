from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Any, Optional
import os
import openai
import json
import time
import re

app = FastAPI(title="Global Risk API", version="1.1.0")

# -----------------------------
# CORS (important for browsers / Coze / frontends)
# -----------------------------
# 生产环境建议把 "*" 换成你的前端域名白名单
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# OpenAI config
# -----------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT_SEC = float(os.getenv("OPENAI_TIMEOUT_SEC", "30"))
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))

Language = Literal["en", "zh"]
RiskLevel = Literal["Low", "Medium", "High"]

CATEGORIES = [
    "weather",
    "political",
    "military",
    "public_security",
    "health",
    "transport",
    "women_safety",
    "discrimination",
]


# -----------------------------
# Models
# -----------------------------
class RiskRequest(BaseModel):
    location: str = Field(..., min_length=1, max_length=80)
    language: Language = "en"


class RiskItem(BaseModel):
    category: str
    level: RiskLevel
    rationale: str
    practical_tips: str


class RiskResponse(BaseModel):
    location: str
    overall_risk_level: RiskLevel
    key_risks: List[RiskItem]


# -----------------------------
# Helpers
# -----------------------------
def _require_api_key():
    if not openai.api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is missing on server. Set it in Render Environment variables."
        )


def _extract_json_object(text: str) -> Dict[str, Any]:
    """
    Robust JSON extraction:
    - First try json.loads directly.
    - If model adds extra text, try to extract the first {...} block and parse.
    """
    text = text.strip()

    # 1) direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) extract first JSON object block
    # This is a pragmatic approach to recover from minor prompt violations.
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise HTTPException(
            status_code=502,
            detail="Model returned non-JSON output (no JSON object found). Please retry."
        )
    candidate = m.group(0)
    try:
        return json.loads(candidate)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Model returned malformed JSON. Parse error: {e}"
        )


def _validate_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enforce required fields and 8 categories exactly once.
    Also normalize category strings if needed.
    """
    for k in ("location", "overall_risk_level", "key_risks"):
        if k not in data:
            raise HTTPException(status_code=502, detail=f"Model JSON missing required field: {k}")

    if not isinstance(data["key_risks"], list) or len(data["key_risks"]) != 8:
        raise HTTPException(status_code=502, detail="key_risks must be a list of 8 items.")

    seen = []
    for item in data["key_risks"]:
        if not isinstance(item, dict):
            raise HTTPException(status_code=502, detail="Each key_risks item must be an object.")
        if "category" not in item or "level" not in item or "rationale" not in item or "practical_tips" not in item:
            raise HTTPException(status_code=502, detail="Each key_risks item must include category, level, rationale, practical_tips.")
        cat = str(item["category"]).strip()
        item["category"] = cat
        seen.append(cat)

    missing = [c for c in CATEGORIES if c not in seen]
    extra = [c for c in seen if c not in CATEGORIES]
    dup = [c for c in set(seen) if seen.count(c) > 1]

    if missing or extra or dup:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "key_risks categories must include all 8 categories exactly once.",
                "missing": missing,
                "extra": extra,
                "duplicated": dup,
                "expected": CATEGORIES,
                "got": seen,
            }
        )

    return data


def _chat(messages: List[Dict[str, str]], temperature: float = TEMPERATURE) -> str:
    _require_api_key()
    try:
        resp = openai.ChatCompletion.create(
            model=DEFAULT_MODEL,
            messages=messages,
            temperature=temperature,
            request_timeout=OPENAI_TIMEOUT_SEC,
        )
        return resp["choices"][0]["message"]["content"]
    except openai.error.RateLimitError as e:
        raise HTTPException(status_code=429, detail=f"OpenAI RateLimitError: {str(e)}")
    except openai.error.AuthenticationError as e:
        raise HTTPException(status_code=401, detail=f"OpenAI AuthenticationError: {str(e)}")
    except openai.error.Timeout as e:
        raise HTTPException(status_code=504, detail=f"OpenAI Timeout: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI call failed: {str(e)}")


# -----------------------------
# Optional: lightweight in-memory rate limit (per IP)
# -----------------------------
# Default: 30 requests per 60 seconds per IP (can adjust by env)
RATE_WINDOW_SEC = int(os.getenv("RATE_WINDOW_SEC", "60"))
RATE_MAX_REQ = int(os.getenv("RATE_MAX_REQ", "30"))
_ip_hits: Dict[str, List[float]] = {}


def _rate_limit(request: Request):
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    hits = _ip_hits.get(ip, [])
    hits = [t for t in hits if now - t <= RATE_WINDOW_SEC]
    if len(hits) >= RATE_MAX_REQ:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: max {RATE_MAX_REQ} requests per {RATE_WINDOW_SEC}s"
        )
    hits.append(now)
    _ip_hits[ip] = hits


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"status": "ok", "service": "Global Risk API"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=RiskResponse)
def analyze_risk(req: RiskRequest, request: Request):
    _rate_limit(request)
    start = time.time()

    if req.language == "zh":
        system = (
            "你是一个严谨的风险分析助手。"
            "你必须只输出严格 JSON（不要 Markdown、不要多余解释、不要代码块、不要前后缀文字）。"
            "输出字段必须完全符合给定 schema。"
        )
        user = (
            f"请对地点：{req.location} 做“旅行/差旅安全与公共风险”结构化评估。\n"
            "你必须严格输出 JSON，schema 如下（字段名必须一致）：\n\n"
            "{\n"
            '  "location": "string",\n'
            '  "overall_risk_level": "Low|Medium|High",\n'
            '  "key_risks": [\n'
            "    {\n"
            '      "category": "weather|political|military|public_security|health|transport|women_safety|discrimination",\n'
            '      "level": "Low|Medium|High",\n'
            '      "rationale": "string",\n'
            '      "practical_tips": "string"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "要求：\n"
            "- key_risks 必须包含上述 8 类，每类 1 条\n"
            "- rationale 要具体可解释，不要空话\n"
            "- practical_tips 给出可执行建议\n"
        )
    else:
        system = (
            "You are a rigorous risk analyst. "
            "You MUST output STRICT JSON only (no markdown, no extra text, no code fences). "
            "The output must match the provided schema exactly."
        )
        user = (
            f"Provide a structured travel/security risk assessment for: {req.location}.\n"
            "You MUST output STRICT JSON only. Schema (field names must match):\n\n"
            "{\n"
            '  "location": "string",\n'
            '  "overall_risk_level": "Low|Medium|High",\n'
            '  "key_risks": [\n'
            "    {\n"
            '      "category": "weather|political|military|public_security|health|transport|women_safety|discrimination",\n'
            '      "level": "Low|Medium|High",\n'
            '      "rationale": "string",\n'
            '      "practical_tips": "string"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Requirements:\n"
            "- key_risks must include ALL 8 categories exactly once\n"
            "- rationale must be specific and explainable\n"
            "- practical_tips must be actionable\n"
        )

    content = _chat(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=TEMPERATURE,
    )

    data = _extract_json_object(content)
    data = _validate_schema(data)

    # Add meta for debugging (safe for downstream; remove if you don't want)
    data["_meta"] = {
        "model": DEFAULT_MODEL,
        "latency_ms": int((time.time() - start) * 1000),
    }

    return data
