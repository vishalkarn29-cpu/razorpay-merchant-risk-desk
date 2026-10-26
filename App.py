
import time
import logging
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from prepare import clean_transactions, add_features
from train import build_final_model
from casenote import run_case, MIN_SCORE
from mock_llm import reset as mock_reset

# Setup logging
logging.basicConfig(filename="service.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("service")

# Build model once at startup
MODEL_BUNDLE = build_final_model()
MODEL = MODEL_BUNDLE["model"]
COLUMNS = MODEL_BUNDLE["columns"]
MEDIAN = MODEL_BUNDLE["median"]
THRESHOLD = MODEL_BUNDLE["threshold"]

KNOWN_CATEGORIES = {"Grocery","Food Delivery","Fashion","Electronics","Utilities","Travel"}

app = FastAPI()

# Metrics
metrics = {
    "requests_total": 0,
    "success_total": 0,
    "error_total": 0,
    "human_review_total": 0,
    "latencies": []
}

class Txn(BaseModel):
    txn_id: str
    merchant_category: str
    city: str
    payment_mode: str
    amount_inr: float
    hour_of_day: int
    txn_count_last_24h: int
    avg_amount_30d_inr: float
    distance_from_home_km: Optional[float] = None
    is_new_device: int
    account_age_days: int

class CaseIn(Txn):
    question: str

def prepare_features(txn_dict):
    # build one-row df
    df = pd.DataFrame([txn_dict])
    # clean-style normalisation for merchant_category and city
    df["merchant_category"] = df["merchant_category"].apply(lambda x: x.strip().title() if isinstance(x,str) else x)
    df["city"] = df["city"].apply(lambda x: x.strip().title() if isinstance(x,str) else x)
    # add_features
    df = add_features(df)
    # one-hot
    # need to create dummies for merchant_category, city, payment_mode - but also keep numeric
    # We'll get_dummies on the whole df except identifiers? Our training excluded txn_id etc.
    # For scoring, we have same columns as training input except we need to handle encoding.
    # Build X_raw same as in train: drop txn_id? Actually txn_dict doesn't have customer_id etc.
    # We'll construct X with all columns needed, then get_dummies and align.
    # Create DataFrame with all required columns
    X_raw = df.drop(columns=["txn_id"], errors="ignore")
    # One-hot
    X = pd.get_dummies(X_raw, dtype=int)
    # Fill missing distance already? distance may be None -> NaN
    # Align to COLUMNS
    for col in COLUMNS:
        if col not in X.columns:
            X[col] = 0
    X = X[COLUMNS]
    # fill median
    X["distance_from_home_km"] = X["distance_from_home_km"].fillna(MEDIAN)
    # Ensure no NaN
    X = X.fillna(0)
    return X

def score_txn(txn_dict):
    X = prepare_features(txn_dict)
    prob = float(MODEL.predict_proba(X)[0,1])
    flagged = prob >= THRESHOLD
    return prob, flagged, X

@app.middleware("http")
async def observe(request: Request, call_next):
    t0 = time.perf_counter()
    request.state.info = {}
    response = await call_next(request)
    if request.method == "POST":
        ms = (time.perf_counter() - t0) * 1000
        status = response.status_code
        info = getattr(request.state, "info", {})
        txn_id = info.get("txn_id")
        prob = info.get("fraud_probability")
        safe_default = info.get("safe_default", False)
        guardrail_override = info.get("guardrail_override", False)

        # metrics
        metrics["requests_total"] += 1
        metrics["latencies"].append(ms)
        if status < 400:
            metrics["success_total"] += 1
        else:
            metrics["error_total"] += 1
        if info.get("human_review"):
            metrics["human_review_total"] += 1

        # logging level logic
        # ERROR if status >=500, WARNING if 4xx or safe_default or guardrail_override, else INFO
        if status >= 500:
            level = logging.ERROR
        elif status >= 400 or safe_default or guardrail_override:
            level = logging.WARNING
        else:
            level = logging.INFO

        # build log message
        endpoint = request.url.path
        msg = f"endpoint={endpoint} status={status} latency_ms={ms:.2f}"
        if txn_id:
            msg += f" txn_id={txn_id}"
        if prob is not None:
            msg += f" fraud_prob={prob:.4f}"
        if safe_default:
            msg += " safe_default=True"
        if guardrail_override:
            msg += " guardrail_override=True"
        logger.log(level, msg)
    return response

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/score")
def score_endpoint(txn: Txn, request: Request):
    # Input guardrails - well-typed but impossible
    # amount_inr
    if not (1 <= txn.amount_inr <= 1_000_000):
        return JSONResponse(status_code=400, content={"error": "amount_inr out of range", "field": "amount_inr"})
    if not (0 <= txn.hour_of_day <= 23):
        return JSONResponse(status_code=400, content={"error": "hour_of_day out of range", "field": "hour_of_day"})
    if txn.txn_count_last_24h < 0:
        return JSONResponse(status_code=400, content={"error": "txn_count_last_24h negative", "field": "txn_count_last_24h"})
    if txn.avg_amount_30d_inr <= 0:
        return JSONResponse(status_code=400, content={"error": "avg_amount_30d_inr must be >0", "field": "avg_amount_30d_inr"})
    # merchant_category after normalisation
    cat_norm = txn.merchant_category.strip().title()
    if cat_norm not in KNOWN_CATEGORIES:
        return JSONResponse(status_code=400, content={"error": "unknown merchant_category", "field": "merchant_category"})

    prob, flagged, _ = score_txn(txn.model_dump())
    decision = "hold_for_review" if flagged else "approve"
    human_review = decision == "hold_for_review"
    # Output guardrail: large payment new device
    guardrail_override = False
    if not flagged and txn.amount_inr >= 100000 and txn.is_new_device == 1:
        decision = "hold_for_review"
        human_review = True
        guardrail_override = True

    request.state.info = {"txn_id": txn.txn_id, "fraud_probability": prob, "human_review": human_review, "safe_default": False, "guardrail_override": guardrail_override}
    return {
        "txn_id": txn.txn_id,
        "fraud_probability": round(prob,4),
        "flagged": bool(flagged),
        "decision": decision,
        "human_review": human_review
    }

@app.post("/case")
def case_endpoint(case: CaseIn, request: Request):
    # Same input guardrails
    if not (1 <= case.amount_inr <= 1_000_000):
        return JSONResponse(status_code=400, content={"error": "amount_inr out of range", "field": "amount_inr"})
    if not (0 <= case.hour_of_day <= 23):
        return JSONResponse(status_code=400, content={"error": "hour_of_day out of range", "field": "hour_of_day"})
    if case.txn_count_last_24h < 0:
        return JSONResponse(status_code=400, content={"error": "txn_count_last_24h negative", "field": "txn_count_last_24h"})
    if case.avg_amount_30d_inr <= 0:
        return JSONResponse(status_code=400, content={"error": "avg_amount_30d_inr must be >0", "field": "avg_amount_30d_inr"})
    cat_norm = case.merchant_category.strip().title()
    if cat_norm not in KNOWN_CATEGORIES:
        return JSONResponse(status_code=400, content={"error": "unknown merchant_category", "field": "merchant_category"})

    prob, flagged, _ = score_txn(case.model_dump())
    # initial decision from scoring path
    decision = "hold_for_review" if flagged else "approve"
    human_review = decision == "hold_for_review"
    # large payment guardrail for unflagged
    guardrail_override_large = False
    if not flagged and case.amount_inr >= 100000 and case.is_new_device == 1:
        decision = "hold_for_review"
        human_review = True
        guardrail_override_large = True

    case_note = None
    attempts = 0
    safe_default = False
    guardrail_override_case = False

    if flagged:
        try:
            result = run_case(case.txn_id, prob, case.amount_inr, case.question)
        except ConnectionError as e:
            request.state.info = {"txn_id": case.txn_id, "fraud_probability": prob, "human_review": False, "safe_default": False, "guardrail_override": False}
            return JSONResponse(status_code=503, content={"error": "case-note service temporarily unavailable, try again later"})
        # no_policy_match
        if result["status"] == "no_policy_match":
            case_note = None
            attempts = 0
            decision = "hold_for_review"
            human_review = True
            safe_default = False
            guardrail_override_case = False
        else:
            case_note = result["case_note"]
            attempts = result["attempts"]
            safe_default = result["safe_default"]
            guardrail_override_case = result["guardrail_override"]
            # Part D extra: if flagged and note says approve, replace with hold
            if case_note["recommended_action"] == "approve":
                case_note["recommended_action"] = "hold_for_review"
                guardrail_override_case = True
            decision = case_note["recommended_action"]
            # block_and_notify also human_review true
            if decision in ("hold_for_review","block_and_notify"):
                human_review = True
            else:
                human_review = False
            if safe_default:
                human_review = True
    else:
        # not flagged -> no case note
        case_note = None
        attempts = 0

    # final human_review for case endpoint includes safe_default etc.
    if case_note is None and flagged and result if flagged and 'result' in locals() and result["status"]=="no_policy_match":
        human_review = True

    # For unflagged with large guardrail, decision already hold
    overall_override = guardrail_override_large or guardrail_override_case or safe_default

    request.state.info = {
        "txn_id": case.txn_id,
        "fraud_probability": prob,
        "human_review": human_review,
        "safe_default": safe_default,
        "guardrail_override": overall_override
    }

    return {
        "txn_id": case.txn_id,
        "fraud_probability": round(prob,4),
        "flagged": bool(flagged),
        "decision": decision,
        "human_review": human_review,
        "case_note": case_note,
        "attempts": attempts,
        "safe_default": safe_default,
        "guardrail_override": guardrail_override_case or guardrail_override_large
    }

@app.get("/metrics")
def get_metrics():
    total = metrics["requests_total"]
    success = metrics["success_total"]
    error = metrics["error_total"]
    avg_lat = sum(metrics["latencies"])/len(metrics["latencies"]) if metrics["latencies"] else 0
    human = metrics["human_review_total"]
    success_rate = round(success/total,4) if total else 0
    error_rate = round(error/total,4) if total else 0
    human_rate = round(human/total,4) if total else 0
    return {
        "requests_total": total,
        "success_total": success,
        "error_total": error,
        "success_rate": success_rate,
        "error_rate": error_rate,
        "avg_latency_ms": round(avg_lat,2),
        "human_review_total": human,
        "human_review_rate": human_rate
    }
