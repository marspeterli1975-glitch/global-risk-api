from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import openai

# Read OpenAI key from environment variables (Render Environment Variables)
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="Global Risk API", version="1.0.0")

# ✅ CORS: allow Hoppscotch / browser calls (for testing, allow all)
# After it works, you can restrict allow_origins to specific domains.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RiskRequest(BaseModel):
    location: str
    language: str = "en"  # "en" or "zh"


@app.get("/")
def root():
    return {"status": "ok", "service": "Global Risk API"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/analyze")
async def analyze_risk(request: RiskRequest):
    if not openai.api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not set in environment variables.",
        )

    prompt_en = f"""
You are a risk analyst.
Provide a structured global risk analysis for: {request.location}

Output format:
1) Overall risk level: Low/Medium/High (one line)
2) Key risks (bullet list, 6-10 items)
3) Category breakdown (Weather / Political / Military / Public Security / Health / Transport) - short bullets
4) Practical safety advice (5-8 bullets)
5) Confidence & data limitations (2-4 bullets)
"""

    prompt_zh = f"""
你是一名风险分析师。
请对做结构化的全球风险分析。

输出格式：
1）总体风险等级：低/中/高（单行）
2）关键风险要点（6-10条要点）
3）分类拆解（天气/政治/军事/治安/健康/交通）每类给简短要点
4）实用安全建议（5-8条）
5）置信度与数据局限（2-4条）
"""

    prompt = prompt_zh if request.language.lower().startswith("zh") else prompt_en

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You produce concise, structured risk reports."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=700,
        )
        content = resp["choices"][0]["message"]["content"]
        return {
            "location": request.location,
            "language": request.language,
            "report": content,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI request failed: {str(e)}")
