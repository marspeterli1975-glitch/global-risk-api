from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os

app = FastAPI()

# ✅ CORS（解决浏览器 Network Error）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")


# 请求结构
class RiskRequest(BaseModel):
    location: str
    language: str = "en"


# 健康检查
@app.get("/")
def root():
    return {"status": "ok", "service": "Global Risk API"}


@app.get("/health")
def health():
    return {"status": "healthy"}


# 核心分析接口
@app.post("/analyze")
async def analyze_risk(request: RiskRequest):

    prompt_en = f"""
Provide a structured global risk analysis for {request.location}.

Include:

1. Weather risk
2. Political risk
3. Military risk
4. Public security
5. Health risk
6. Transport risk
7. Women safety
8. Racial issues

Then give:

- Overall risk level (Low / Medium / High)
- Short explanation
"""

    prompt_cn = f"""
请对 {request.location} 进行全球风险分析，包括：

1. 天气风险
2. 政治风险
3. 军事风险
4. 治安风险
5. 健康风险
6. 交通风险
7. 女性安全
8. 种族问题

并给出：

- 综合风险等级（低 / 中 / 高）
- 简要说明
"""

    prompt = prompt_cn if request.language == "zh" else prompt_en

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a global risk analysis expert."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    result = response["choices"][0]["message"]["content"]

    return {
        "location": request.location,
        "analysis": result
    }
