
import json
from retrieval import retrieve, MIN_SCORE, POLICIES
from mock_llm import generate, reset

# JSON Schema dict
SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "priority": {"type": "string", "enum": ["low","medium","high"]},
        "recommended_team": {"type": "string", "enum": ["card_risk","upi_risk","merchant_ops","customer_care"]},
        "recommended_action": {"type": "string", "enum": ["approve","hold_for_review","block_and_notify"]},
        "policy_id": {"type": "string", "enum": ["P1","P2","P3","P4","P5","P6"]}
    },
    "required": ["summary","priority","recommended_team","recommended_action","policy_id"]
}

def build_prompt(txn_id, fraud_probability, amount_inr, policy_id, policy_text):
    prompt = f"""Role: You are a fraud analyst assistant for the Merchant Risk Desk.
Task: Write a short case note for the flagged transaction using only the provided policy.
Constraints: Use only the policy text given. Reply with one JSON object and nothing else. Do not add explanations outside JSON.
Output format: JSON with keys summary (string), priority (low|medium|high), recommended_team (card_risk|upi_risk|merchant_ops|customer_care), recommended_action (approve|hold_for_review|block_and_notify), policy_id (P1|P2|P3|P4|P5|P6).
TXN_ID: {txn_id}
FRAUD_PROBABILITY: {fraud_probability}
AMOUNT_INR: {amount_inr}
POLICY_ID: {policy_id}
Policy text: {policy_text}
"""
    return prompt

def validate(text, retrieved_policy_id):
    # Gate 1: valid JSON object
    try:
        note = json.loads(text)
    except:
        return None, 1
    if not isinstance(note, dict):
        return None, 1
    # Gate 2: required keys and all strings
    for k in SCHEMA["required"]:
        if k not in note:
            return None, 2
        if not isinstance(note[k], str):
            return None, 2
    # Gate 3: allowed values and policy_id equals retrieved
    if note["priority"] not in SCHEMA["properties"]["priority"]["enum"]:
        return None, 3
    if note["recommended_team"] not in SCHEMA["properties"]["recommended_team"]["enum"]:
        return None, 3
    if note["recommended_action"] not in SCHEMA["properties"]["recommended_action"]["enum"]:
        return None, 3
    if note["policy_id"] not in SCHEMA["properties"]["policy_id"]["enum"]:
        return None, 3
    if note["policy_id"] != retrieved_policy_id:
        return None, 3
    return note, 0

SAFE_DEFAULT_TEMPLATE = {
    "summary": "Automated case note unavailable; manual review required.",
    "priority": "high",
    "recommended_team": "customer_care",
    "recommended_action": "hold_for_review",
}

def run_case(txn_id, fraud_probability, amount_inr, question, max_attempts=3):
    pid, ptext, score = retrieve(question)
    if score < MIN_SCORE:
        return {
            "status": "no_policy_match",
            "attempts": 0,
            "failures": [],
            "case_note": None,
            "safe_default": False,
            "guardrail_override": False
        }
    prompt = build_prompt(txn_id, fraud_probability, amount_inr, pid, ptext)
    failures = []
    attempts = 0
    note = None
    safe_default = False
    guardrail_override = False
    for i in range(max_attempts):
        attempts += 1
        # may raise ConnectionError - propagate
        text = generate(prompt)
        parsed, gate = validate(text, pid)
        if gate==0:
            note = parsed
            break
        else:
            failures.append(gate)
    if note is None:
        # safe default
        safe_default = True
        note = {
            **SAFE_DEFAULT_TEMPLATE,
            "policy_id": pid
        }
    # Output guardrail: if approve and prob>=0.5 -> hold
    if note["recommended_action"] == "approve" and fraud_probability >= 0.5:
        note["recommended_action"] = "hold_for_review"
        guardrail_override = True

    return {
        "status": "ok",
        "attempts": attempts,
        "failures": failures,
        "case_note": note,
        "safe_default": safe_default,
        "guardrail_override": guardrail_override
    }

if __name__ == "__main__":
    reset()
    with open("case_scenarios.json") as f:
        scenarios = json.load(f)
    print("Scenario | Retrieved | Attempts | Failures | Status | SafeDefault | FinalAction | Override")
    total_calls = 0
    first_call_ok = 0
    recovered = 0
    safe_defaults = 0
    overrides = 0
    for idx, sc in enumerate(scenarios, start=1):
        res = run_case(sc["txn_id"], sc["fraud_probability"], sc["amount_inr"], sc["question"])
        total_calls += res["attempts"]
        if res["attempts"]==1 and not res["failures"]:
            first_call_ok+=1
        elif res["attempts"]>1 and res["status"]=="ok" and not res["safe_default"]:
            recovered+=1
        if res["safe_default"]:
            safe_defaults+=1
        if res["guardrail_override"]:
            overrides+=1
        print(f"S{idx} ({sc['txn_id']}) | {res['case_note']['policy_id'] if res['case_note'] else None} | {res['attempts']} | {res['failures']} | {res['status']} | {res['safe_default']} | {res['case_note']['recommended_action'] if res['case_note'] else None} | {res['guardrail_override']}")
    print(f"Totals: {total_calls} calls, {first_call_ok} first-call ok, {recovered} recovered by retry, {safe_defaults} safe default, {overrides} override")
