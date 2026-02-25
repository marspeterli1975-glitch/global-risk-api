from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class RiskRequest(BaseModel):
    location: str
    language: str = "en"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze_risk(request: RiskRequest):

    prompt = f"""
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

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a global risk analysis expert."},
            {"role": "user", "content": prompt}
        ]
    )

    return {
        "location": request.location,
        "analysis": response.choices[0].message.content
    }
