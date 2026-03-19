import pandas as pd
import io
import math
from fastapi import UploadFile, File
import os
import time
import json
import hmac
import base64
import hashlib
import csv

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from typing import Dict, Any, Optional

import stripe
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfgen import canvas

# -----------------------------
# Load env
# -----------------------------
load_dotenv()

# -----------------------------
# ENV
# -----------------------------
APP_ENV = os.getenv("APP_ENV", "production")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")

TOKEN_SECRET = os.getenv("TOKEN_SECRET", "riskatlas-secret")
SUCCESS_URL = os.getenv("SUCCESS_URL", "")
CANCEL_URL = os.getenv("CANCEL_URL", "")
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")

PAID_TOKEN_TTL_SECONDS = int(os.getenv("PAID_TOKEN_TTL_SECONDS", "3600"))

SCAN_ENABLED = os.getenv("SCAN_ENABLED", "true").lower() == "true"
PAYMENT_ENABLED = os.getenv("PAYMENT_ENABLED", "true").lower() == "true"
REPORT_DOWNLOAD_ENABLED = os.getenv("REPORT_DOWNLOAD_ENABLED", "true").lower() == "true"

# -----------------------------
# Stripe
# -----------------------------
if not STRIPE_SECRET_KEY:
    raise RuntimeError("Missing STRIPE_SECRET_KEY")

stripe.api_key = STRIPE_SECRET_KEY

# -----------------------------
# App
# -----------------------------
app = FastAPI(
    title="RiskAtlas API",
    version="2.0.0",
    debug=DEBUG,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# =========================
# Load Planning Upload API
# =========================

@app.post("/load-planning/upload")
async def load_planning_upload(file: UploadFile = File(...)):
    try:
        filename = (file.filename or "").lower()

        if not filename.endswith(".csv"):
            return {
                "success": False,
                "error": "Only CSV supported"
            }

        content = await file.read()

        if not content:
            return {
                "success": False,
                "error": "Uploaded file is empty"
            }

        try:
            decoded = content.decode("utf-8-sig")
        except Exception:
            decoded = content.decode("utf-8")

        try:
            df_reader = csv.DictReader(io.StringIO(decoded))
            rows = list(df_reader)
        except Exception as e:
            return {
                "success": False,
                "error": f"CSV parse error: {str(e)}"
            }

        if not rows:
            return {
                "success": False,
                "error": "CSV contains no data rows"
            }

        results = []

        for idx, row in enumerate(rows, start=1):
            try:
                product_name = str(row.get("Product Name", "") or "").strip()
                hs_code = str(row.get("Suggested HS Code", "") or "").strip()

                length = float(row.get("Max Outer Length (mm)", 0) or 0)
                width = float(row.get("Max Outer Width (mm)", 0) or 0)
                height = float(row.get("Max Outer Height (mm)", 0) or 0)
                weight = float(row.get("Gross Weight per Unit (kg)", 0) or 0)
                qty = int(float(row.get("Quantity", 0) or 0))
                stack = int(float(row.get("Max Stack Layers", 1) or 1))

                if qty <= 0:
                    continue

                volume_per_unit = (length * width * height) / 1_000_000_000
                total_volume = volume_per_unit * qty
                total_weight = weight * qty

                # V1 简化逻辑：先按一个基础柜型估算
                # 后续可以换成真实 container template engine
                units_per_container = 10
                estimated_container_count = max(1, (qty + units_per_container - 1) // units_per_container)

                results.append({
                    "product_name": product_name or f"Line {idx}",
                    "hs_code": hs_code,
                    "units_per_container": units_per_container,
                    "estimated_container_count": estimated_container_count,
                    "total_volume_m3": round(total_volume, 3),
                    "total_weight_kg": round(total_weight, 2),
                })

            except Exception as row_err:
                results.append({
                    "product_name": f"Line {idx}",
                    "hs_code": "",
                    "units_per_container": 0,
                    "estimated_container_count": 0,
                    "total_volume_m3": 0,
                    "total_weight_kg": 0,
                })

        return {
            "success": True,
            "results": results
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN] if ALLOWED_ORIGIN != "*" else ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# -----------------------------
# In-memory storage
# Production later: move to DB
# -----------------------------
SCANS: Dict[str, Dict[str, Any]] = {}
PAID: Dict[str, Dict[str, Any]] = {}
PROCESSED_WEBHOOK_EVENTS: Dict[str, float] = {}
RATE_LIMIT_BUCKETS: Dict[str, list] = {}

REPORT_DIR = "/tmp/riskatlas_reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# -----------------------------
# Models
# -----------------------------
class RiskScanRequest(BaseModel):
    company_name: str = Field(..., min_length=1, max_length=100)
    country: str = Field(..., min_length=1, max_length=60)
    industry: str = Field(..., min_length=1, max_length=80)
    logistics_complexity: int = Field(..., ge=1, le=5)
    event_disruption: int = Field(..., ge=1, le=5)
    email: str = Field(..., min_length=5, max_length=120)


class CheckoutRequest(BaseModel):
    session_id: str = Field(..., min_length=8, max_length=120)
    amount_cents: int = Field(..., ge=100, le=500000)
    currency: str = Field(default="aud", min_length=3, max_length=3)
    product_name: str = Field(default="RiskAtlas Exposure Report", min_length=3, max_length=120)


# -----------------------------
# Helpers
# -----------------------------
def now_ts() -> int:
    return int(time.time())


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def rate_limit_check(key: str, max_requests: int, window_seconds: int) -> None:
    now = time.time()
    bucket = RATE_LIMIT_BUCKETS.get(key, [])
    bucket = [ts for ts in bucket if now - ts < window_seconds]

    if len(bucket) >= max_requests:
        raise HTTPException(status_code=429, detail="Too many requests, please try again later.")

    bucket.append(now)
    RATE_LIMIT_BUCKETS[key] = bucket


def generate_session_id(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(raw + str(now_ts()).encode("utf-8")).hexdigest()
    return f"scan_{digest[:24]}"


def grade_from_score(score: int) -> str:
    if score <= 20:
        return "A"
    if score <= 40:
        return "B"
    if score <= 60:
        return "C"
    if score <= 80:
        return "D"
    return "E"


def exposure_level(score: int) -> str:
    if score <= 20:
        return "Low"
    if score <= 40:
        return "Guarded"
    if score <= 60:
        return "Moderate"
    if score <= 80:
        return "High"
    return "Critical"


def risk_engine(req: RiskScanRequest) -> Dict[str, Any]:
    country = req.country.lower()
    industry = req.industry.lower()

    country_risk = 18
    if country in {"singapore", "japan", "australia"}:
        country_risk = 10
    elif country in {"india", "indonesia", "vietnam"}:
        country_risk = 18
    elif country in {"nigeria", "pakistan"}:
        country_risk = 24

    industry_risk = 16
    if industry in {"battery", "battery materials", "chemicals", "mining"}:
        industry_risk = 22
    elif industry in {"electronics", "automotive"}:
        industry_risk = 18

    logistics_risk = min(req.logistics_complexity * 6, 30)
    event_risk = min(req.event_disruption * 5 + 1, 20)

    total = min(country_risk + industry_risk + logistics_risk + event_risk, 100)
    grade = grade_from_score(total)
    level = exposure_level(total)

    return {
        "score": total,
        "grade": grade,
        "exposure_level": level,
        "dimensions": {
            "country_risk": country_risk,
            "industry_sensitivity": industry_risk,
            "logistics_complexity": logistics_risk,
            "event_disruption": event_risk,
        },
        "summary": (
            f"RiskAtlas evaluates structural exposure across supply chain environments "
            f"based on country conditions, industry sensitivity, logistics complexity, "
            f"and event-driven disruption factors. The overall exposure score is {total}, "
            f"indicating a {level.lower()} level of supply chain exposure."
        ),
    }


def sign_token(session_id: str, email: str, pdf_path: str) -> str:
    payload = {
        "sid": session_id,
        "email": email,
        "pdf": pdf_path,
        "exp": now_ts() + PAID_TOKEN_TTL_SECONDS
    }
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    sig = hmac.new(TOKEN_SECRET.encode("utf-8"), body, hashlib.sha256).digest()

    return (
        base64.urlsafe_b64encode(body).decode("utf-8") + "." +
        base64.urlsafe_b64encode(sig).decode("utf-8")
    )


def verify_token(token: str) -> Dict[str, Any]:
    try:
        payload_b64, sig_b64 = token.split(".")
        body = base64.urlsafe_b64decode(payload_b64.encode("utf-8"))
        sig = base64.urlsafe_b64decode(sig_b64.encode("utf-8"))
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token format")

    expected = hmac.new(TOKEN_SECRET.encode("utf-8"), body, hashlib.sha256).digest()
    if not hmac.compare_digest(sig, expected):
        raise HTTPException(status_code=401, detail="Invalid token signature")

    payload = json.loads(body.decode("utf-8"))
    if payload["exp"] < now_ts():
        raise HTTPException(status_code=401, detail="Token expired")

    return payload


def report_filename(country: str, industry: str, session_id: str) -> str:
    safe_country = country.replace(" ", "-")
    safe_industry = industry.replace(" ", "-")
    return os.path.join(REPORT_DIR, f"riskatlas-report-{safe_country}-{safe_industry}-{session_id}.pdf")


def draw_footer(c: canvas.Canvas, page_num: int):
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.grey)
    c.drawString(20 * mm, 10 * mm, f"Generated by Eastrion RiskAtlas  Page {page_num}")


def draw_cover(c: canvas.Canvas, scan: Dict[str, Any], analysis: Dict[str, Any]):
    c.setFont("Helvetica-Bold", 24)
    c.drawString(20 * mm, 270 * mm, "Eastrion RiskAtlas")

    c.setFont("Helvetica", 18)
    c.drawString(20 * mm, 258 * mm, "Supply Chain Exposure Assessment")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, 230 * mm, "Country")
    c.drawString(80 * mm, 230 * mm, "Industry")
    c.drawString(140 * mm, 230 * mm, "Exposure Score")

    c.setFont("Helvetica", 14)
    c.drawString(20 * mm, 220 * mm, scan["country"])
    c.drawString(80 * mm, 220 * mm, scan["industry"])
    c.drawString(140 * mm, 220 * mm, str(analysis["score"]))

    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, 200 * mm, "Grade")
    c.drawString(80 * mm, 200 * mm, "Exposure Level")

    c.setFont("Helvetica", 14)
    c.drawString(20 * mm, 190 * mm, analysis["grade"])
    c.drawString(80 * mm, 190 * mm, analysis["exposure_level"])

    c.setFont("Helvetica", 11)
    c.drawString(20 * mm, 150 * mm, "Generated by RiskAtlas")
    c.drawString(20 * mm, 142 * mm, "Eastrion – Global Supply Chain Infrastructure")
    c.drawString(20 * mm, 134 * mm, "Confidential – Strategic Assessment Document")

    draw_footer(c, 1)
    c.showPage()


def draw_exec_summary(c: canvas.Canvas, scan: Dict[str, Any], analysis: Dict[str, Any]):
    c.setFont("Helvetica-Bold", 18)
    c.drawString(20 * mm, 270 * mm, "Executive Summary")

    text = c.beginText(20 * mm, 250 * mm)
    text.setFont("Helvetica", 11)
    for line in [
        "RiskAtlas evaluates structural exposure across supply chain environments based on country",
        "conditions, industry sensitivity, logistics complexity, and event-driven disruption factors.",
        f"For the selected parameters, the overall exposure score is {analysis['score']}, indicating a",
        f"{analysis['exposure_level'].lower()} level of supply chain exposure."
    ]:
        text.textLine(line)
    c.drawText(text)

    c.setFont("Helvetica-Bold", 14)
    c.drawString(20 * mm, 205 * mm, f"Exposure Score: {analysis['score']}")
    c.drawString(20 * mm, 195 * mm, f"Grade: {analysis['grade']}")
    c.drawString(20 * mm, 185 * mm, f"Exposure Level: {analysis['exposure_level']}")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, 160 * mm, "Overall Interpretation")

    text = c.beginText(20 * mm, 150 * mm)
    text.setFont("Helvetica", 11)
    for line in [
        "The current exposure profile suggests that companies operating within the selected market",
        "environment should consider reinforcing supply chain resilience in a targeted manner.",
        "Priority attention should be given to supplier diversification, logistics route redundancy,",
        "and active monitoring of policy or operational developments."
    ]:
        text.textLine(line)
    c.drawText(text)

    draw_footer(c, 2)
    c.showPage()


def draw_breakdown(c: canvas.Canvas, analysis: Dict[str, Any]):
    dims = analysis["dimensions"]

    c.setFont("Helvetica-Bold", 18)
    c.drawString(20 * mm, 270 * mm, "Exposure Score Breakdown")

    c.setFont("Helvetica", 11)
    c.drawString(20 * mm, 258 * mm, "RiskAtlas evaluates exposure across four structural dimensions that influence supply chain resilience.")

    c.setFont("Helvetica-Bold", 11)
    c.drawString(20 * mm, 240 * mm, "Risk Dimension")
    c.drawString(95 * mm, 240 * mm, "Score")
    c.drawString(120 * mm, 240 * mm, "Interpretation")

    rows = [
        ("Country Risk", dims["country_risk"], "Moderate macro and operating exposure"),
        ("Industry Sensitivity", dims["industry_sensitivity"], "Sector-specific material and compliance sensitivity"),
        ("Logistics Complexity", dims["logistics_complexity"], "Transport and corridor concentration exposure"),
        ("Event Disruption", dims["event_disruption"], "Policy and short-term disruption vulnerability"),
    ]

    y = 228
    c.setFont("Helvetica", 10)
    for label, score, interp in rows:
        c.drawString(20 * mm, y * mm, label)
        c.drawString(95 * mm, y * mm, str(score))
        c.drawString(120 * mm, y * mm, interp[:45])
        y -= 10

    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, 170 * mm, "Exposure Level Definition")

    levels = [
        "0–20   Low        Stable supply environment with limited structural exposure.",
        "21–40  Guarded    Limited but manageable exposure requiring routine monitoring.",
        "41–60  Moderate   Visible structural vulnerability across selected dimensions.",
        "61–80  High       Significant disruption probability and concentrated risk signals.",
        "81–100 Critical   Structural instability across key supply chain conditions.",
    ]

    text = c.beginText(20 * mm, 158 * mm)
    text.setFont("Helvetica", 10)
    for line in levels:
        text.textLine(line)
    c.drawText(text)

    # Simple bars
    c.setFont("Helvetica-Bold", 11)
    c.drawString(20 * mm, 105 * mm, "Dimension Visualization")
    y_bar = 95
    for label, score, _ in rows:
        c.setFont("Helvetica", 10)
        c.drawString(20 * mm, y_bar * mm, label)
        c.setFillColor(colors.HexColor("#4F46E5"))
        c.rect(65 * mm, (y_bar - 2) * mm, score * 1.2 * mm, 4 * mm, fill=1, stroke=0)
        c.setFillColor(colors.black)
        c.drawString(135 * mm, y_bar * mm, str(score))
        y_bar -= 10

    draw_footer(c, 3)
    c.showPage()


def draw_signals(c: canvas.Canvas):
    c.setFont("Helvetica-Bold", 18)
    c.drawString(20 * mm, 270 * mm, "Key Structural Risk Signals")

    sections = [
        ("Supplier Concentration", [
            "Supply chains within the selected industry may remain dependent on a limited number of",
            "upstream suppliers, which can increase vulnerability to disruptions in availability,",
            "lead times, or pricing dynamics."
        ]),
        ("Logistics Corridor Exposure", [
            "Transportation flows may rely on a relatively concentrated set of corridors, ports, or",
            "maritime routes, introducing exposure to congestion, chokepoints, and continuity",
            "pressures."
        ]),
        ("Regulatory and Operating Variability", [
            "The operating environment may be influenced by evolving policy direction, compliance",
            "requirements, or intervention risk, which can affect planning assumptions and",
            "execution predictability."
        ]),
    ]

    y = 248
    for title, lines in sections:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(20 * mm, y * mm, title)
        y -= 8
        text = c.beginText(20 * mm, y * mm)
        text.setFont("Helvetica", 10)
        for line in lines:
            text.textLine(line)
        c.drawText(text)
        y -= 24

    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, 140 * mm, "Key Risk Factors Identified")

    bullets = [
        "Potential infrastructure and execution variability in cross-border operations",
        "Higher compliance, hazardous handling, or strategic material sensitivity",
        "Elevated likelihood of shipment delays, supplier instability, or operational friction",
        "Exposure to compounding disruption from geopolitical or regulatory shifts",
    ]

    text = c.beginText(25 * mm, 130 * mm)
    text.setFont("Helvetica", 10)
    for item in bullets:
        text.textLine(f"• {item}")
    c.drawText(text)

    draw_footer(c, 4)
    c.showPage()


def draw_actions(c: canvas.Canvas, analysis: Dict[str, Any]):
    c.setFont("Helvetica-Bold", 18)
    c.drawString(20 * mm, 270 * mm, "Strategic Interpretation and Actions")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, 250 * mm, "Strategic Interpretation")

    text = c.beginText(20 * mm, 240 * mm)
    text.setFont("Helvetica", 10)
    for line in [
        "The current exposure profile suggests that companies operating within the selected market",
        "environment should consider reinforcing supply chain resilience in a targeted manner.",
        "Priority attention should be given to supplier diversification, logistics route redundancy,",
        "and active monitoring of policy or operational developments."
    ]:
        text.textLine(line)
    c.drawText(text)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, 195 * mm, "Suggested Strategic Actions")

    actions = [
        "Priority 1  Evaluate alternative supplier options to reduce dependency on concentrated upstream sources.",
        "Priority 2  Assess alternate transportation corridors to mitigate logistics concentration and chokepoint exposure.",
        "Priority 3  Establish a more active monitoring rhythm for regulatory and operating environment changes.",
        "Priority 4  Review inventory buffering assumptions for critical inputs, lead times, and continuity planning.",
        "Priority 5  Escalate exposure review to leadership level before scaling commercial commitments.",
    ]

    text = c.beginText(20 * mm, 185 * mm)
    text.setFont("Helvetica", 10)
    for line in actions:
        text.textLine(line)
    c.drawText(text)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, 115 * mm, "Methodology")

    methodology = [
        "Country Risk × 30%",
        "Industry Sensitivity × 25%",
        "Logistics Complexity × 25%",
        "Event Disruption Risk × 20%",
        "The model aggregates structured risk indicators to generate an overall exposure score ranging from 0 to 100."
    ]

    text = c.beginText(20 * mm, 105 * mm)
    text.setFont("Helvetica", 10)
    for line in methodology:
        text.textLine(line)
    c.drawText(text)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, 65 * mm, "Disclaimer")
    c.setFont("Helvetica", 9)
    c.drawString(20 * mm, 57 * mm, "RiskAtlas is a supply chain exposure scanning tool for informational use only.")

    draw_footer(c, 5)
    c.showPage()


def generate_pdf_report(scan: Dict[str, Any]) -> str:
    analysis = scan["analysis"]
    path = report_filename(scan["country"], scan["industry"], scan["session_id"])

    c = canvas.Canvas(path, pagesize=A4)
    draw_cover(c, scan, analysis)
    draw_exec_summary(c, scan, analysis)
    draw_breakdown(c, analysis)
    draw_signals(c)
    draw_actions(c, analysis)
    c.save()

    return path


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
async def root():
    return {"ok": True, "service": "RiskAtlas API", "env": APP_ENV}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "stripe_configured": bool(STRIPE_SECRET_KEY),
        "webhook_configured": bool(STRIPE_WEBHOOK_SECRET),
        "time": now_ts(),
    }


@app.post("/scan")
async def scan(req: RiskScanRequest, request: Request):
    if not SCAN_ENABLED:
        raise HTTPException(status_code=503, detail="Scan temporarily disabled")

    rate_limit_check(f"scan:{client_ip(request)}", max_requests=8, window_seconds=300)

    analysis = risk_engine(req)
    session_id = generate_session_id(req.dict())

    SCANS[session_id] = {
        "session_id": session_id,
        "company_name": req.company_name,
        "country": req.country,
        "industry": req.industry,
        "email": req.email,
        "analysis": analysis,
        "created": now_ts(),
        "paid": False,
    }

    return {
        "session_id": session_id,
        "analysis": analysis,
        "payment_status": "created"
    }


@app.post("/create-checkout-session")
async def checkout(req: CheckoutRequest, request: Request):
    if not PAYMENT_ENABLED:
        raise HTTPException(status_code=503, detail="Payment temporarily disabled")

    rate_limit_check(f"checkout:{client_ip(request)}", max_requests=5, window_seconds=600)

    scan = SCANS.get(req.session_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Session not found")

    if scan.get("paid"):
        raise HTTPException(status_code=400, detail="Session already paid")

    # 把内部 session_id 传回前端 success 页面
    separator = "&" if "?" in SUCCESS_URL else "?"
    success_url = f"{SUCCESS_URL}{separator}session_id={req.session_id}"

    session = stripe.checkout.Session.create(
        mode="payment",
        success_url=success_url,
        cancel_url=CANCEL_URL,
        customer_email=scan["email"],
        line_items=[
            {
                "price_data": {
                    "currency": req.currency.lower(),
                    "product_data": {
                        "name": req.product_name,
                    },
                    "unit_amount": req.amount_cents,
                },
                "quantity": 1,
            }
        ],
        metadata={
            "riskatlas_session_id": req.session_id,
            "company_name": scan["company_name"],
            "country": scan["country"],
            "industry": scan["industry"],
        },
    )

    scan["checkout_session_id"] = session.id
    scan["payment_status"] = "checkout_created"

    return {
        "checkout_url": session.url,
        "checkout_session_id": session.id,
        "payment_status": "checkout_created",
        "success_url": success_url
    }


@app.post("/webhook")
async def webhook(request: Request):
    payload = await request.body()
    sig = request.headers.get("stripe-signature")

    if not sig:
        raise HTTPException(status_code=400, detail="Missing Stripe signature")

    try:
        event = stripe.Webhook.construct_event(
            payload,
            sig,
            STRIPE_WEBHOOK_SECRET
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid webhook")

    event_id = event.get("id")
    if event_id in PROCESSED_WEBHOOK_EVENTS:
        return {"status": "already_processed", "event_id": event_id}

    PROCESSED_WEBHOOK_EVENTS[event_id] = time.time()

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]

        sid = session.get("metadata", {}).get("riskatlas_session_id")
        payment_status = session.get("payment_status")

        if not sid:
            return {"status": "missing_session"}

        if payment_status != "paid":
            return {"status": "ignored", "payment_status": payment_status}

        scan = SCANS.get(sid)
        if not scan:
            return {"status": "unknown_session"}

        scan["paid"] = True
        scan["paid_at"] = now_ts()

        pdf_path = generate_pdf_report(scan)
        token = sign_token(sid, scan["email"], pdf_path)

        PAID[sid] = {
            "token": token,
            "pdf_path": pdf_path,
            "time": now_ts(),
        }

        print("Stripe payment verified:", sid)

    return {"status": "ok"}


@app.get("/payment-status/{session_id}")
async def payment_status(session_id: str):
    scan = SCANS.get(session_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Session not found")

    result = {
        "session_id": session_id,
        "paid": scan.get("paid", False),
    }

    if scan.get("paid") and session_id in PAID:
        result["download_token"] = PAID[session_id]["token"]

    return result


@app.get("/report/json")
async def report_json(token: str):
    payload = verify_token(token)
    sid = payload["sid"]

    scan = SCANS.get(sid)
    if not scan:
        raise HTTPException(status_code=404, detail="Report not found")

    return {
        "title": "Supply Chain Risk Report",
        "company_name": scan["company_name"],
        "country": scan["country"],
        "industry": scan["industry"],
        "score": scan["analysis"]["score"],
        "grade": scan["analysis"]["grade"],
        "exposure_level": scan["analysis"]["exposure_level"],
    }


@app.get("/report/download")
async def report_download(token: str):
    if not REPORT_DOWNLOAD_ENABLED:
        raise HTTPException(status_code=503, detail="Report download temporarily disabled")

    payload = verify_token(token)
    pdf_path = payload["pdf"]

    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF not found")

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=os.path.basename(pdf_path)
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    if DEBUG:
        return JSONResponse(status_code=500, content={"detail": str(exc)})
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
