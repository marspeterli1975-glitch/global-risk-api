import os
import time
import json
import hmac
import hashlib
import secrets
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# =====================================================
# Basic App
# =====================================================
APP_VERSION = "0.6.0-scrs-scoring"
APP_SECRET = os.getenv("APP_SECRET", "change-this-in-render-env")

app = FastAPI(title="SCRS API", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# In-memory token store
# NOTE:
# Acceptable for MVP testing only.
# On redeploy/restart, tokens will reset.
# =====================================================
ISSUED_TOKENS: Dict[str, Dict[str, Any]] = {}


# =====================================================
# Helpers
# =====================================================
def create_token(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    sig = hmac.new(APP_SECRET.encode("utf-8"), raw, hashlib.sha256).hexdigest()
    token = f"{secrets.token_urlsafe(24)}.{sig[:16]}"
    ISSUED_TOKENS[token] = {
        "payload": payload,
        "created_at": int(time.time())
    }
    return token


def verify_bearer_token(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")

    token = authorization.replace("Bearer ", "", 1).strip()

    if token not in ISSUED_TOKENS:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return token


def score_to_level(score: float) -> str:
    if score < 20:
        return "Low"
    elif score < 40:
        return "Moderate"
    elif score < 60:
        return "Elevated"
    elif score < 80:
        return "High"
    return "Critical"


def clamp_score(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 1)


# =====================================================
# Request Model
# =====================================================
class SCRSRequest(BaseModel):
    origin_country: str = Field(..., description="Country where shipment originates")
    destination_country: str = Field(..., description="Destination country")

    supplier_years: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="Years supplier has been operating"
    )

    supplier_financial_level: Optional[str] = Field(
        default="unknown",
        description="Supplier financial stability: strong / adequate / weak / distressed / unknown"
    )

    supplier_dependency_pct: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="Buyer dependency percentage on supplier"
    )

    transport_mode: str = Field(
        ...,
        description="Transport mode: sea / air / rail / road / multimodal"
    )

    route_region: str = Field(
        ...,
        description="Shipping route region"
    )

    product_type: str = Field(
        ...,
        description="Product category"
    )

    payment_terms: str = Field(
        ...,
        description="Payment terms"
    )


# =====================================================
# Scoring Functions
# =====================================================
def map_supplier_years_to_score(years: Optional[int]) -> float:
    if years is None:
        return 50.0
    if years >= 10:
        return 10.0
    if years >= 5:
        return 30.0
    if years >= 2:
        return 60.0
    return 85.0


def map_supplier_financial_to_score(level: Optional[str]) -> float:
    mapping = {
        "strong": 15.0,
        "adequate": 40.0,
        "weak": 70.0,
        "distressed": 90.0,
        "unknown": 55.0
    }
    if not level:
        return 55.0
    return mapping.get(level.lower(), 55.0)


def map_supplier_dependency_to_score(pct: Optional[int]) -> float:
    if pct is None:
        return 50.0
    if pct < 20:
        return 10.0
    if pct < 40:
        return 30.0
    if pct < 60:
        return 55.0
    if pct < 80:
        return 75.0
    return 90.0


def score_supplier(data: SCRSRequest) -> Dict[str, Any]:
    years_score = map_supplier_years_to_score(data.supplier_years)
    financial_score = map_supplier_financial_to_score(data.supplier_financial_level)
    dependency_score = map_supplier_dependency_to_score(data.supplier_dependency_pct)

    # MVP: delivery / quality unavailable, use neutral placeholders
    delivery_score = 50.0
    quality_score = 50.0

    total = (
        years_score * 0.15 +
        financial_score * 0.25 +
        delivery_score * 0.25 +
        quality_score * 0.20 +
        dependency_score * 0.15
    )

    drivers: List[str] = []
    if dependency_score >= 75:
        drivers.append("High supplier dependency")
    if financial_score >= 70:
        drivers.append("Weak supplier financial stability")
    if years_score >= 60:
        drivers.append("Limited supplier operating history")

    return {
        "score": clamp_score(total),
        "drivers": drivers
    }


def score_country(data: SCRSRequest) -> Dict[str, Any]:
    stable_low = {
        "singapore", "japan", "south korea", "germany", "netherlands",
        "australia", "new zealand", "canada", "switzerland", "united states",
        "usa", "uk", "united kingdom"
    }
    stable_mid = {
        "china", "malaysia", "thailand", "vietnam", "uae", "saudi arabia",
        "mexico", "turkey", "poland", "indonesia"
    }
    higher_risk = {
        "india", "philippines", "south africa", "brazil", "argentina", "egypt"
    }
    very_high_risk = {
        "russia", "ukraine", "iran", "iraq", "syria", "yemen", "sudan"
    }

    countries = {
        data.origin_country.strip().lower(),
        data.destination_country.strip().lower()
    }

    score = 40.0
    drivers: List[str] = []

    if countries & very_high_risk:
        score = 85.0
        drivers.append("Exposure to very high-risk country environment")
    elif countries & higher_risk:
        score = 60.0
        drivers.append("Exposure to higher-volatility country environment")
    elif countries & stable_mid:
        score = 35.0
    elif countries & stable_low:
        score = 20.0
    else:
        score = 45.0
        drivers.append("Country risk profile not fully benchmarked")

    return {
        "score": clamp_score(score),
        "drivers": drivers
    }


def score_route(data: SCRSRequest) -> Dict[str, Any]:
    route = data.route_region.strip().lower()
    mode = data.transport_mode.strip().lower()

    route_scores = {
        "red_sea": 85.0,
        "black_sea": 90.0,
        "middle_east_corridor": 65.0,
        "indian_ocean": 50.0,
        "asia_europe": 45.0,
        "transpacific": 35.0,
        "south_asia": 45.0,
        "intra_asia": 25.0,
        "domestic_cross_border": 30.0,
        "other": 50.0
    }

    score = route_scores.get(route, 50.0)

    if mode == "air":
        score = max(10.0, score - 10.0)
    elif mode == "multimodal":
        score = min(100.0, score + 5.0)

    drivers: List[str] = []
    if score >= 75:
        drivers.append("Unstable or conflict-sensitive route")
    elif score >= 55:
        drivers.append("Moderately exposed shipping route")

    return {
        "score": clamp_score(score),
        "drivers": drivers
    }


def score_port(data: SCRSRequest) -> Dict[str, Any]:
    mode = data.transport_mode.strip().lower()
    route = data.route_region.strip().lower()

    if mode in {"air", "road", "rail"}:
        score = 20.0
    else:
        if route in {"red_sea", "black_sea"}:
            score = 70.0
        elif route in {"indian_ocean", "middle_east_corridor", "south_asia"}:
            score = 45.0
        elif route in {"asia_europe", "transpacific"}:
            score = 35.0
        else:
            score = 30.0

    drivers: List[str] = []
    if score >= 60:
        drivers.append("Potential port congestion or handling disruption")

    return {
        "score": clamp_score(score),
        "drivers": drivers
    }


def score_regulatory(data: SCRSRequest) -> Dict[str, Any]:
    product = data.product_type.strip().lower()
    destination = data.destination_country.strip().lower()

    score = 30.0
    drivers: List[str] = []

    if product in {"dangerous_goods", "chemicals"}:
        score = 70.0
        drivers.append("Sensitive product class with elevated compliance burden")
    elif product in {"battery_materials", "metals_minerals"}:
        score = 60.0
        drivers.append("Strategic material class with trade/compliance sensitivity")
    elif product in {"machinery", "industrial_parts"}:
        score = 35.0
    else:
        score = 25.0

    if destination in {"india", "eu", "european union"} and product in {"battery_materials", "chemicals", "dangerous_goods"}:
        score = min(100.0, score + 10.0)
        drivers.append("Destination-side import/compliance requirements may be stricter")

    return {
        "score": clamp_score(score),
        "drivers": drivers
    }


def score_financial(data: SCRSRequest) -> Dict[str, Any]:
    payment = data.payment_terms.strip().lower()

    mapping = {
        "confirmed_lc": 10.0,
        "lc": 20.0,
        "tt_advance": 25.0,
        "tt_partial_balance_bl": 40.0,
        "oa_30": 65.0,
        "oa_60": 80.0,
        "consignment": 90.0,
        "other": 50.0
    }

    score = mapping.get(payment, 50.0)

    drivers: List[str] = []
    if payment in {"oa_30", "oa_60", "consignment"}:
        drivers.append("Open-account or delayed-cashflow exposure")
    if payment == "consignment":
        drivers.append("Consignment structure materially increases payment risk")

    return {
        "score": clamp_score(score),
        "drivers": drivers
    }


def score_commodity(data: SCRSRequest) -> Dict[str, Any]:
    product = data.product_type.strip().lower()

    mapping = {
        "general_goods": 20.0,
        "consumer_goods": 25.0,
        "industrial_parts": 35.0,
        "machinery": 40.0,
        "metals_minerals": 60.0,
        "battery_materials": 75.0,
        "chemicals": 65.0,
        "temperature_sensitive": 55.0,
        "dangerous_goods": 70.0,
        "other": 45.0
    }

    score = mapping.get(product, 45.0)

    drivers: List[str] = []
    if score >= 70:
        drivers.append("High-volatility or tightly regulated product category")
    elif score >= 55:
        drivers.append("Moderately sensitive product category")

    return {
        "score": clamp_score(score),
        "drivers": drivers
    }


def score_disruption(data: SCRSRequest) -> Dict[str, Any]:
    route = data.route_region.strip().lower()
    mode = data.transport_mode.strip().lower()

    if route == "black_sea":
        score = 90.0
    elif route == "red_sea":
        score = 85.0
    elif route in {"middle_east_corridor", "indian_ocean"}:
        score = 45.0
    elif mode == "multimodal":
        score = 40.0
    else:
        score = 25.0

    drivers: List[str] = []
    if score >= 75:
        drivers.append("Elevated disruption exposure from route instability")
    elif score >= 45:
        drivers.append("Moderate exposure to disruption events")

    return {
        "score": clamp_score(score),
        "drivers": drivers
    }


def calculate_scrs(data: SCRSRequest) -> Dict[str, Any]:
    supplier = score_supplier(data)
    country = score_country(data)
    route = score_route(data)
    port = score_port(data)
    regulatory = score_regulatory(data)
    financial = score_financial(data)
    commodity = score_commodity(data)
    disruption = score_disruption(data)

    dimension_scores = {
        "supplier_risk": supplier["score"],
        "country_risk": country["score"],
        "route_risk": route["score"],
        "port_risk": port["score"],
        "regulatory_risk": regulatory["score"],
        "financial_risk": financial["score"],
        "commodity_risk": commodity["score"],
        "disruption_risk": disruption["score"],
    }

    total_score = (
        dimension_scores["supplier_risk"] * 0.25 +
        dimension_scores["country_risk"] * 0.15 +
        dimension_scores["route_risk"] * 0.15 +
        dimension_scores["port_risk"] * 0.10 +
        dimension_scores["regulatory_risk"] * 0.10 +
        dimension_scores["financial_risk"] * 0.10 +
        dimension_scores["commodity_risk"] * 0.10 +
        dimension_scores["disruption_risk"] * 0.05
    )

    total_score = clamp_score(total_score)
    risk_level = score_to_level(total_score)

    all_drivers: List[str] = (
        supplier["drivers"] +
        country["drivers"] +
        route["drivers"] +
        port["drivers"] +
        regulatory["drivers"] +
        financial["drivers"] +
        commodity["drivers"] +
        disruption["drivers"]
    )

    ranked_dimensions = sorted(
        dimension_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top_dimension_names = [item[0] for item in ranked_dimensions[:3]]

    top_drivers: List[str] = []
    seen = set()
    for d in all_drivers:
        if d not in seen:
            seen.add(d)
            top_drivers.append(d)
        if len(top_drivers) >= 3:
            break

    if not top_drivers:
        top_drivers = ["No dominant acute driver detected in MVP rule set"]

    summary = (
        f"SCRS indicates {risk_level.lower()} supply chain exposure, mainly driven by "
        f"{', '.join(top_dimension_names)}."
    )

    return {
        "scrs_score": total_score,
        "risk_level": risk_level,
        "dimension_scores": dimension_scores,
        "risk_drivers": top_drivers,
        "summary": summary
    }


# =====================================================
# Endpoints
# =====================================================
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "service": "scrs-api"
    }


@app.get("/success", response_class=HTMLResponse)
async def success(session_id: Optional[str] = None):
    issued_at = int(time.time())

    payload = {
        "session_id": session_id or f"manual-{issued_at}",
        "scope": "analyze",
        "issued_at": issued_at
    }

    token = create_token(payload)

    html = f"""
    <html>
      <head>
        <title>SCRS Payment Success</title>
        <meta charset="utf-8" />
      </head>
      <body style="font-family: Arial, sans-serif; padding: 32px;">
        <h2>Payment Success</h2>
        <p>Your access token has been issued.</p>
        <p><strong>Token:</strong></p>
        <textarea rows="5" cols="100" readonly>{token}</textarea>
        <p style="margin-top: 16px;">Use this token in your API request header:</p>
        <pre>Authorization: Bearer {token}</pre>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.post("/analyze")
async def analyze(data: SCRSRequest, authorization: Optional[str] = Header(default=None)):
    token = verify_bearer_token(authorization)
    result = calculate_scrs(data)

    response = {
        "status": "success",
        "version": APP_VERSION,
        "authorized": True,
        "token_preview": token[:12] + "...",
        "input_received": data.dict(),
        "result": result
    }

    return JSONResponse(content=response)


@app.get("/")
async def root():
    return {
        "name": "SCRS API",
        "version": APP_VERSION,
        "status": "running"
    }
