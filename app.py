from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import openai

# ===== App =====
app = FastAPI(title="Global Risk API", version="1.0.0")

# ===== CORS (Hoppscotch / Browser needs this) =====
# 最宽松写法：允许任意来源。后续你要上线再收紧到指定域名。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== OpenAI Key =====
openai.api_key = os.getenv("OPENAI_API_KEY")


class RiskRequest(BaseModel):
    location: str
    language: str = "en"


@app.get("/")
def root():
    return {"status": "ok", "service": "Global Risk API"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_risk(request: RiskRequest):
    # 1) Key 检查：不给 500“黑盒”，直接告诉你缺什么
    if not openai.api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not set on the server. Please add it in Render -> Settings -> Environment.",
        )

    # 2) Prompt
    prompt_en = f"""
Provide a structured global risk analysis for: {request.location}

Return JSON with the following fields:
- location
- overall_risk_level (Low/Medium/High)
- key_risks: array of objects with {{category, level, rationale, practical_tips}}
Categories must include:
weather, political, military, public_security, health, transport, women_safety, discrimination

Keep it concise but actionable.
"""

    prompt_cn = f"""
请对 {request.location} 做一份结构化“全球风险”分析。

请只输出 JSON，字段如下：
- location
- overall_risk_level（低/中/高）
- key_risks：数组，每项包含 {{category, level, rationale, practical_tips}}
category 必须包含：
weather（天气）、political（政治）、military（军事/冲突）、public_security（治安）、health（健康/疫情）、transport（交通）、women_safety（女性安全）、discrimination（歧视/族群）

要求：简洁但可执行。
"""

    user_prompt = prompt_cn if request.language.lower().startswith("zh") else prompt_en

    # 3) Call OpenAI (openai==0.28.1)
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a risk analysis assistant. Output must be valid JSON only."},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=800,
        )
        content = resp["choices"][0]["message"]["content"]
        return {"result": content}
    except Exception as e:
        # 把错误原因透出来，方便你在 Hoppscotch/Render logs 里直接定位
        raise HTTPException(status_code=500, detail=f"OpenAI call failed: {repr(e)}")
