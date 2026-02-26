from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, List, Optional, Dict, Any
import os
import openai
import json
import time

app = FastAPI(title="Global Risk API", version="1.0.0")

# ---- OpenAI key ----
openai.api_key = os.getenv("OPENAI_API_KEY")


# ---- Models ----
Language = Literal["en", "zh"]

class RiskRequest(BaseModel):
    location: str = Field(..., min_length=1, max_length=80)
    language: Language = "en"

class RiskItem(BaseModel):
    category: str
    level: Literal["Low", "Medium", "High"]
    rationale: str
    practical_tips: str

class RiskResponse(BaseModel):
    location: str
    overall_risk_level: Literal["Low", "Medium", "High"]
    key_risks: List[RiskItem]


# ---- Helpers ----
def _require_api_key():
    if not openai.api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is missing on server. Please set it in Render Environment variables."
        )

def _safe_json_loads(s: str) -> Dict[str, Any]:
    """
    Try to parse model output as JSON. If it fails, raise a clean error.
    """
    try:
        return json.loads(s)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Model returned non-JSON output. Please retry or tighten the prompt. Parse error: {e}"
        )

def _chat(messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    """
    OpenAI ChatCompletion call (openai==0.28.1).
    """
    _require_api_key()
    try:
        resp = openai.ChatCompletion.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            temperature=temperature,
        )
        return resp["choices"][0]["message"]["content"]
    except openai.error.RateLimitError as e:
        # 429 / quota / rate limit
        raise HTTPException(status_code=429, detail=f"OpenAI RateLimitError: {str(e)}")
    except openai.error.AuthenticationError as e:
        raise HTTPException(status_code=401, detail=f"OpenAI AuthenticationError: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI call failed: {str(e)}")


# ---- Routes ----
@app.get("/")
def root():
    return {"status": "ok", "service": "Global Risk API"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze", response_model=RiskResponse)
def analyze_risk(req: RiskRequest):
    """
    Returns a structured JSON response that downstream workflows can consume reliably.
    """
    start = time.time()

    # Prompt: enforce STRICT JSON output
    if req.language == "zh":
        system = (
            "你是一个严谨的风险分析助手。"
            "你必须只输出严格 JSON（不要 Markdown、不要多余解释、不要代码块、不要前后缀文字）。"
            "输出字段必须完全符合给定 schema。"
        )
        user = f"""
请对地点：{req.location} 做“旅行/差旅安全与公共风险”结构化评估。
你必须严格输出 JSON，schema 如下（字段名必须一致）：

{{
  "location": "string",
  "overall_risk_level": "Low|Medium|High",
  "key_risks": [
    {{
      "category": "weather|political|military|public_security|health|transport|women_safety|discrimination",
      "level": "Low|Medium|High",
      "rationale": "string",
      "practical_tips": "string"
    }}
  ]
}}

要求：
- key_risks 必须包含上述 8 类，每类 1 条
- rationale 要可解释，不要空话
- practical_tips 给出可执行建议
"""
    else:
        system = (
            "You are a rigorous risk analyst. "
            "You MUST output STRICT JSON only (no markdown, no extra text, no code fences). "
            "The output must match the provided schema exactly."
        )
        user = f"""
Provide a structured travel/security risk assessment for: {req.location}.
You MUST output STRICT JSON only. Schema (field names must match):

{{
  "location": "string",
  "overall_risk_level": "Low|Medium|High",
  "key_risks": [
    {{
      "category": "weather|political|military|public_security|health|transport|women_safety|discrimination",
      "level": "Low|Medium|High",
      "rationale": "string",
      "practical_tips": "string"
    }}
  ]
}}

Requirements:
- key_risks must include ALL 8 categories exactly once
- rationale must be specific and explainable
- practical_tips must be actionable
"""

    content = _chat(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0.2
    )

    data = _safe_json_loads(content)

    # Minimal server-side validation of required structure
    if "location" not in data or "overall_risk_level" not in data or "key_risks" not in data:
        raise HTTPException(status_code=502, detail="Model JSON missing required fields.")

    # Optional: attach server timing (if you want, uncomment)
    # data["_meta"] = {"latency_ms": int((time.time() - start) * 1000)}

    return data
