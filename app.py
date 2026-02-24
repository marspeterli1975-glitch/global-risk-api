from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import openai
import os

app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")

class RiskRequest(BaseModel):
    location: str
    language: str = "en"

@app.post("/analyze")
async def analyze_risk(request: RiskRequest):

    prompt_en = f"""
    Provide a structured global risk analysis for {request.location}.
    Cover:
    - Weather risk
    - Political risk
    - Military risk
    - Public security
    - Health risk
    - Transport risk
    - Women safety
    - Racial issues
    Give overall risk level (Low / Medium / High).
    """

    prompt_cn = f"""
    请对 {request.location} 进行全球风险分析，包括：
    - 天气风险
    - 政治风险
    - 军事风险
    - 治安情况
    - 健康风险
    - 交通风险
    - 女性安全
    - 种族问题
    并给出综合风险等级（低 / 中 / 高）。
    """

    prompt = prompt_cn if request.language == "cn" else prompt_en

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )

    return {
        "location": request.location,
        "analysis": response.choices[0].message["content"]
    }
