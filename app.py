from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import random

app = FastAPI(
    title="RiskAtlas SCRS API",
    version="0.7",
    description="Supply Chain Risk Scan Engine"
)

# -----------------------------
# Health Check
# -----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "0.7",
        "service": "scrs-api"
    }


# -----------------------------
# Request Model
# -----------------------------
class RiskRequest(BaseModel):
    supplier_country: str
    port: Optional[str] = None
    product: Optional[str] = None


# -----------------------------
# Risk Scan
# -----------------------------
@app.post("/risk")
def risk_scan(req: RiskRequest):

    risk_score = random.choice(["A", "B", "C"])

    return {
        "supplier_country": req.supplier_country,
        "port": req.port,
        "product": req.product,
        "risk_score": risk_score,
        "factors": [
            "geopolitical",
            "logistics",
            "currency",
            "climate"
        ]
    }


# -----------------------------
# Risk Report
# -----------------------------
@app.post("/report")
def risk_report(req: RiskRequest):

    risk_score = random.choice(["A", "B", "C"])

    report = {
        "title": "Supply Chain Risk Report",
        "supplier_country": req.supplier_country,
        "port": req.port,
        "product": req.product,
        "risk_score": risk_score,
        "key_risks": [
            "port congestion",
            "currency volatility",
            "regional instability"
        ],
        "recommendation": "Diversify suppliers and monitor logistics routes."
    }

    return report
