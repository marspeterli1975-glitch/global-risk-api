import pandas as pd
import io
from typing import Any, Dict, List
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
# =========================================
# Load Planning + Operational Risk Engine
# =========================================

CONTAINER_LIBRARY = {
    "20GP": {
        "volume_m3": 33.0,
        "payload_kg": 28200,
        "label": "20GP",
    },
    "40GP": {
        "volume_m3": 67.0,
        "payload_kg": 26700,
        "label": "40GP",
    },
    "40HQ": {
        "volume_m3": 76.0,
        "payload_kg": 26500,
        "label": "40HQ",
    },
}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def normalize_yes_no(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in {"yes", "y", "true", "1"}


def pick_best_container(total_volume_m3: float, total_weight_kg: float) -> Dict[str, Any]:
    candidates = []

    for container_code, container in CONTAINER_LIBRARY.items():
        volume_capacity = container["volume_m3"]
        payload_capacity = container["payload_kg"]

        volume_count = max(1, int(-(-total_volume_m3 // volume_capacity))) if volume_capacity > 0 else 999
        weight_count = max(1, int(-(-total_weight_kg // payload_capacity))) if payload_capacity > 0 else 999
        required_count = max(volume_count, weight_count)

        volume_utilization = 0.0
        weight_utilization = 0.0

        if required_count > 0:
            volume_utilization = total_volume_m3 / (required_count * volume_capacity)
            weight_utilization = total_weight_kg / (required_count * payload_capacity)

        candidates.append({
            "container_type": container_code,
            "container_label": container["label"],
            "container_volume_m3": volume_capacity,
            "container_payload_kg": payload_capacity,
            "estimated_container_count": required_count,
            "volume_utilization": round(volume_utilization, 3),
            "weight_utilization": round(weight_utilization, 3),
        })

    # 优先柜数少，其次利用率更平衡
    candidates.sort(
        key=lambda x: (
            x["estimated_container_count"],
            abs(0.72 - x["volume_utilization"]),
            abs(0.72 - x["weight_utilization"]),
        )
    )
    return candidates[0]


def evaluate_operational_risk(
    total_volume_m3: float,
    total_weight_kg: float,
    estimated_container_count: int,
    volume_utilization: float,
    weight_utilization: float,
    max_stack_layers: int,
    rotatable: bool,
    invertible: bool,
    allow_mixed_load: bool,
    cargo_form: str,
    preferred_container: str = "",
    only_china_vi: bool = False,
    height_restriction_alert: bool = False,
    weight_restriction_alert: bool = False,
    appointment_required: bool = False,
    closed_body_preferred: bool = False,
) -> Dict[str, Any]:
    score = 0
    flags: List[str] = []
    notes: List[str] = []
    actions: List[str] = []

    # 1. 重量风险
    if total_weight_kg >= 26000:
        score += 30
        flags.append("overweight_risk")
        notes.append("Total shipment weight is beyond or very close to standard container payload ceiling.")
        actions.append("Split shipment into more containers or switch to a lighter batch structure.")
    elif total_weight_kg >= 22000:
        score += 18
        flags.append("weight_limit_close")
        notes.append("Shipment weight is close to a common payload red line.")
        actions.append("Recheck container payload by selected type and destination handling conditions.")

    # 2. 体积利用率风险
    if volume_utilization < 0.30:
        score += 12
        flags.append("low_utilization")
        notes.append("Container volume utilization is low, indicating cost inefficiency.")
        actions.append("Consider combining shipments or switching to a smaller container type.")
    elif volume_utilization > 0.92:
        score += 14
        flags.append("high_density_loading")
        notes.append("Container volume utilization is very high and leaves little execution buffer.")
        actions.append("Review placement margin, handling tolerance, and loading feasibility.")

    # 3. 重量利用率风险
    if weight_utilization > 0.90:
        score += 15
        flags.append("payload_stress")
        notes.append("Payload utilization is high and may amplify operational fragility.")
        actions.append("Add execution buffer or split the shipment to reduce payload concentration.")

    # 4. 多柜复杂度
    if estimated_container_count >= 4:
        score += 12
        flags.append("multi_container_complexity")
        notes.append("Multi-container execution increases coordination complexity.")
        actions.append("Confirm booking, loading sequence, and destination receiving arrangement.")
    elif estimated_container_count >= 2:
        score += 6
        flags.append("multi_container_execution")
        notes.append("More than one container is required, increasing execution steps.")

    # 5. 堆叠与形态
    if max_stack_layers <= 1:
        score += 12
        flags.append("stacking_limited")
        notes.append("Non-stackable or low-stack cargo reduces loading flexibility.")
        actions.append("Review cargo support structure or batch split strategy.")
    elif max_stack_layers == 2:
        score += 6
        flags.append("stacking_constrained")
        notes.append("Stacking limit may reduce achievable loading efficiency.")

    if cargo_form == "liquid":
        score += 10
        flags.append("liquid_handling_attention")
        notes.append("Liquid cargo usually requires additional handling control.")
    elif cargo_form == "powder":
        score += 8
        flags.append("powder_handling_attention")
        notes.append("Powder cargo may have handling and cleanliness sensitivity.")

    if not rotatable:
        score += 5
        flags.append("rotation_not_allowed")
        notes.append("Orientation flexibility is limited because rotation is not allowed.")

    if not invertible:
        score += 5
        flags.append("inversion_not_allowed")
        notes.append("Vertical inversion is not allowed, reducing loading flexibility.")

    if not allow_mixed_load:
        score += 4
        flags.append("dedicated_space_preferred")
        notes.append("Mixed loading is not allowed, which can reduce consolidation flexibility.")

    # 6. 目的地执行限制
    if only_china_vi:
        score += 5
        flags.append("china_vi_constraint")
        notes.append("Destination requires China VI-compliant inland vehicle arrangement.")
        actions.append("Confirm local truck compliance before dispatch.")

    if height_restriction_alert:
        score += 8
        flags.append("height_restriction_alert")
        notes.append("Destination or route may contain height restriction concerns.")
        actions.append("Verify total loaded height and route entry conditions.")

    if weight_restriction_alert:
        score += 8
        flags.append("weight_restriction_alert")
        notes.append("Destination or route may contain weight restriction concerns.")
        actions.append("Verify axle/load restrictions and receiving site access conditions.")

    if appointment_required:
        score += 6
        flags.append("appointment_required")
        notes.append("Delivery requires appointment coordination.")
        actions.append("Book receiving window early to avoid demurrage or waiting time.")

    if closed_body_preferred:
        score += 6
        flags.append("closed_body_preferred")
        notes.append("Closed-body transport is preferred for destination handling.")
        actions.append("Avoid open-body truck if cargo sensitivity or site rule applies.")

    # 7. 偏好柜型不匹配
    if preferred_container and preferred_container in CONTAINER_LIBRARY:
        best = pick_best_container(total_volume_m3, total_weight_kg)
        if preferred_container != best["container_type"]:
            score += 8
            flags.append("preferred_container_mismatch")
            notes.append("Preferred container is not aligned with current shipment structure.")
            actions.append(f"Reconsider preferred container. Current best-fit estimate is {best['container_type']}.")

    score = min(score, 100)

    if score >= 65:
        level = "High"
    elif score >= 35:
        level = "Medium"
    else:
        level = "Low"

    if actions:
        dedup_actions = list(dict.fromkeys(actions))
    else:
        dedup_actions = ["Proceed with booking after basic verification."]

    return {
        "operational_risk_score": score,
        "operational_risk_level": level,
        "risk_flags": flags,
        "risk_notes": notes,
        "recommended_actions": dedup_actions,
    }


def calculate_macro_risk(row: Dict[str, Any], op_risk_score: int) -> Dict[str, Any]:
    """
    V1 先做一个简化版 RiskAtlas 合并层：
    - 不引入国家/港口实时数据
    - 基于货物形态、执行限制、复杂度，形成 Macro/Execution 代理分
    后续可以再接你原来的 RiskAtlas 真正评分引擎。
    """
    macro_score = 22
    execution_score = 18

    cargo_form = str(row.get("Cargo Form", "") or "").strip().lower()
    qty = safe_int(row.get("Quantity", 0), 0)
    note_text = str(row.get("Note", "") or "").strip()

    if cargo_form == "liquid":
        macro_score += 8
        execution_score += 8
    elif cargo_form == "powder":
        macro_score += 6
        execution_score += 6

    if qty >= 100:
        execution_score += 10
    elif qty >= 30:
        execution_score += 5

    if normalize_yes_no(row.get("Appointment Required", "no")):
        execution_score += 8

    if normalize_yes_no(row.get("Height Restriction Alert", "no")):
        execution_score += 8

    if normalize_yes_no(row.get("Weight Restriction Alert", "no")):
        execution_score += 8

    if note_text:
        execution_score += 4

    macro_score = min(macro_score, 100)
    execution_score = min(execution_score, 100)

    final_score = round((macro_score * 0.40) + (op_risk_score * 0.30) + (execution_score * 0.30))

    if final_score >= 75:
        final_level = "High"
    elif final_score >= 45:
        final_level = "Medium"
    else:
        final_level = "Low"

    return {
        "macro_risk_score": macro_score,
        "execution_risk_score": execution_score,
        "final_riskatlas_score": final_score,
        "final_riskatlas_level": final_level,
    }


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
            reader = csv.DictReader(io.StringIO(decoded))
            rows = list(reader)
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

                length = safe_float(row.get("Max Outer Length (mm)", 0), 0)
                width = safe_float(row.get("Max Outer Width (mm)", 0), 0)
                height = safe_float(row.get("Max Outer Height (mm)", 0), 0)
                weight = safe_float(row.get("Gross Weight per Unit (kg)", 0), 0)
                qty = safe_int(row.get("Quantity", 0), 0)
                stack = safe_int(row.get("Max Stack Layers", 1), 1)

                cargo_form = str(row.get("Cargo Form", "solid") or "solid").strip().lower()
                rotatable = normalize_yes_no(row.get("Rotatable", "yes"))
                invertible = normalize_yes_no(row.get("Invertible", "no"))
                allow_mixed_load = normalize_yes_no(row.get("Allow Mixed Load", "yes"))

                preferred_container = str(row.get("Preferred Container", "") or "").strip()
                only_china_vi = normalize_yes_no(row.get("Only China VI", "no"))
                height_restriction_alert = normalize_yes_no(row.get("Height Restriction Alert", "no"))
                weight_restriction_alert = normalize_yes_no(row.get("Weight Restriction Alert", "no"))
                appointment_required = normalize_yes_no(row.get("Appointment Required", "no"))
                closed_body_preferred = normalize_yes_no(row.get("Closed Body Preferred", "no"))

                if qty <= 0:
                    continue

                volume_per_unit = (length * width * height) / 1_000_000_000
                total_volume = volume_per_unit * qty
                total_weight = weight * qty

                best_container = pick_best_container(total_volume, total_weight)

                op_risk = evaluate_operational_risk(
                    total_volume_m3=total_volume,
                    total_weight_kg=total_weight,
                    estimated_container_count=best_container["estimated_container_count"],
                    volume_utilization=best_container["volume_utilization"],
                    weight_utilization=best_container["weight_utilization"],
                    max_stack_layers=stack,
                    rotatable=rotatable,
                    invertible=invertible,
                    allow_mixed_load=allow_mixed_load,
                    cargo_form=cargo_form,
                    preferred_container=preferred_container,
                    only_china_vi=only_china_vi,
                    height_restriction_alert=height_restriction_alert,
                    weight_restriction_alert=weight_restriction_alert,
                    appointment_required=appointment_required,
                    closed_body_preferred=closed_body_preferred,
                )

                riskatlas = calculate_macro_risk(row, op_risk["operational_risk_score"])

                planning_insight = {
                    "recommended_container": best_container["container_type"],
                    "estimated_container_count": best_container["estimated_container_count"],
                    "volume_utilization": best_container["volume_utilization"],
                    "weight_utilization": best_container["weight_utilization"],
                    "efficiency_band": (
                        "High" if best_container["volume_utilization"] >= 0.75
                        else "Medium" if best_container["volume_utilization"] >= 0.45
                        else "Low"
                    ),
                }

                results.append({
                    "product_name": product_name or f"Line {idx}",
                    "hs_code": hs_code,
                    "units_per_container": max(
                        1,
                        int(qty / best_container["estimated_container_count"])
                    ),
                    "estimated_container_count": best_container["estimated_container_count"],
                    "total_volume_m3": round(total_volume, 3),
                    "total_weight_kg": round(total_weight, 2),

                    "planning_insight": planning_insight,
                    "operational_risk": op_risk,
                    "riskatlas": riskatlas,
                })

            except Exception as row_err:
                results.append({
                    "product_name": f"Line {idx}",
                    "hs_code": "",
                    "units_per_container": 0,
                    "estimated_container_count": 0,
                    "total_volume_m3": 0,
                    "total_weight_kg": 0,
                    "planning_insight": {
                        "recommended_container": "N/A",
                        "estimated_container_count": 0,
                        "volume_utilization": 0,
                        "weight_utilization": 0,
                        "efficiency_band": "N/A",
                    },
                    "operational_risk": {
                        "operational_risk_score": 0,
                        "operational_risk_level": "N/A",
                        "risk_flags": [f"row_parse_error_{idx}"],
                        "risk_notes": [str(row_err)],
                        "recommended_actions": ["Check CSV row structure and numeric fields."],
                    },
                    "riskatlas": {
                        "macro_risk_score": 0,
                        "execution_risk_score": 0,
                        "final_riskatlas_score": 0,
                        "final_riskatlas_level": "N/A",
                    },
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


@app.post("/load-planning/run-riskatlas")
async def run_load_planning_riskatlas(file: UploadFile = File(...)):
    """
    给前端一个明确的‘Run Full RiskAtlas Analysis’入口。
    当前 V1 先复用 upload 结果并突出 RiskAtlas 合并层输出。
    """
    upload_result = await load_planning_upload(file)

    if not upload_result.get("success"):
        return upload_result

    results = upload_result.get("results", [])

    portfolio_summary = {
        "line_count": len(results),
        "total_operational_risk_score": 0,
        "total_final_riskatlas_score": 0,
        "highest_risk_line": "",
        "overall_level": "Low",
    }

    if results:
        op_scores = [r["operational_risk"]["operational_risk_score"] for r in results]
        final_scores = [r["riskatlas"]["final_riskatlas_score"] for r in results]

        portfolio_summary["total_operational_risk_score"] = round(sum(op_scores) / len(op_scores))
        portfolio_summary["total_final_riskatlas_score"] = round(sum(final_scores) / len(final_scores))

        highest_idx = final_scores.index(max(final_scores))
        portfolio_summary["highest_risk_line"] = results[highest_idx]["product_name"]

        avg_final = portfolio_summary["total_final_riskatlas_score"]
        if avg_final >= 75:
            portfolio_summary["overall_level"] = "High"
        elif avg_final >= 45:
            portfolio_summary["overall_level"] = "Medium"
        else:
            portfolio_summary["overall_level"] = "Low"

    return {
        "success": True,
        "summary": portfolio_summary,
        "results": results
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
