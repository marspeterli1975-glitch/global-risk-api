# app.py
import os
import time
import traceback
from typing import Optional, List, Literal

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ----------------------------
# App init
# ----------------------------
app = FastAPI(title="Global Risk API", version="0.1.0")

# CORS: 你用 Hoppscotch Browser 模式时会需要；Proxy/Agent/Extension 通常不需要
# 为了省事，先放开 hoppscotch 域名。你后续上线再收紧 allow_origins。
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
    allow_headers=["*"],  # 关键：允许 X-API-Key / Authorization
)


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
    factors: List[dict]
    meta: dict


# ----------------------------
# Helpers
# ----------------------------
def _extract_api_key(request: Request) -> Optional[str]:
    """
    Support:
      - X-API-Key: <key>
      - Authorization: Bearer <key>
    """
    api_key = request.headers.get("X-API-Key")
    if api_key and api_key.strip():
        return api_key.strip()

    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
        return token if token else None

    return None


def _load_allowed_keys() -> set[str]:
    """
    Read APP_API_KEYS (comma-separated).
    Example:
      APP_API_KEYS="key1,key2,key3"
    """
    raw = os.getenv("APP_API_KEYS", "").strip()
    if not raw:
        return set()
    return {k.strip() for k in raw.split(",") if k.strip()}


def require_api_key(request: Request) -> str:
    """
    API key gate for protected endpoints.
    - Missing key -> 401
    - Gate not configured -> 503
    - Invalid key -> 403
    """
    api_key = _extract_api_key(request)
    if not api_key:
        raise HTTPException(status_code=401, detail="missing_api_key")

    allowed = _load_allowed_keys()
    if not allowed:
        # 生产上这比 500 更合理：服务未配置好
        raise HTTPException(status_code=503, detail="api_key_gate_not_configured")

    if api_key not in allowed:
        raise HTTPException(status_code=403, detail="invalid_api_key")

    return api_key


# ----------------------------
# Error handling (JSON everywhere)
# ----------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    debug = os.getenv("DEBUG", "0") == "1"
    payload = {"detail": "internal_error", "error": str(exc)}
    if debug:
        payload["traceback_tail"] = traceback.format_exc().splitlines()[-30:]
    return JSONResponse(status_code=500, content=payload)


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/engine/analyze", response_model=AnalyzeResponse)
def engine_analyze(req: AnalyzeRequest, request: Request):
    # 1) API key gate
    _ = require_api_key(request)

    # 2) Business logic
    #    这里我先给你一个稳定的“可返回结构”，避免你还没接上模型/数据就报错。
    #    你后续把 analyze_risk(req) 的内部替换成真实逻辑（抓取、检索、归因、评分）。
    t0 = time.time()
    result = analyze_risk(req)
    result["meta"]["latency_ms"] = int((time.time() - t0) * 1000)
    return result


# ----------------------------
# Core logic (replace with your engine later)
# ----------------------------
def analyze_risk(req: AnalyzeRequest) -> dict:
    """
    Placeholder risk engine.
    Replace this with your real pipeline:
      - fetch signals/news
      - extract risk factors
      - score/normalize
      - generate summary
    """
    # 简单示例：用 location 做一个伪评分（保证永远不会崩）
    base = 50
    bump = 5 if req.location.lower() in {"tokyo", "japan"} else 0
    score = max(0, min(100, base + bump))

    factors = [
        {"name": "macro", "score": 55, "evidence": ["placeholder"]},
        {"name": "policy", "score": 45, "evidence": ["placeholder"]},
        {"name": "security", "score": 50, "evidence": ["placeholder"]},
    ]

    summary_en = f"Risk assessment for {req.location}: overall score {score}/100."
    summary_zh = f"{req.location} 风险评估：综合评分 {score}/100。"
    summary = summary_en if req.language == "en" else summary_zh

    return {
        "ok": True,
        "location": req.location,
        "language": req.language,
        "risk_score": score,
        "summary": summary,
        "factors": factors,
        "meta": {
            "engine_version": "0.1.0",
            "data_mode": "placeholder",
            "latency_ms": 0,
        },
    }
    return data
