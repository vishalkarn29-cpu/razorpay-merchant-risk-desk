
import time
import requests

BASE = "http://127.0.0.1:8000"

BENIGN = {"merchant_category": "Grocery", "city": "Pune", "payment_mode": "UPI", "amount_inr": 1500,
          "hour_of_day": 13, "txn_count_last_24h": 0, "avg_amount_30d_inr": 1600,
          "distance_from_home_km": 3.0, "is_new_device": 0, "account_age_days": 900}
RISKY  = {"merchant_category": "Electronics", "city": "Mumbai", "payment_mode": "Card", "amount_inr": 45000,
          "hour_of_day": 2, "txn_count_last_24h": 9, "avg_amount_30d_inr": 3000,
          "distance_from_home_km": 60.0, "is_new_device": 1, "account_age_days": 40}
BIG    = {"merchant_category": "Utilities", "city": "Pune", "payment_mode": "NetBanking", "amount_inr": 150000,
          "avg_amount_30d_inr": 250000, "distance_from_home_km": 0.0, "is_new_device": 1, "account_age_days": 2200,
          "hour_of_day": 13, "txn_count_last_24h": 0}

def wait_health():
    for _ in range(120):
        try:
            r = requests.get(f"{BASE}/health", timeout=1)
            if r.status_code==200:
                return True
        except:
            pass
        time.sleep(1)
    return False

def run():
    print("Waiting for service...")
    if not wait_health():
        print("Service not up")
        return
    print("Service up")
    results = []
    # Q1
    r = requests.get(f"{BASE}/health")
    results.append(("Q1","happy",200,r.status_code, r.status_code==200))

    # Q2
    payload = {**BENIGN, "txn_id":"T800001"}
    r = requests.post(f"{BASE}/score", json=payload)
    j = r.json() if r.status_code==200 else {}
    cond = r.status_code==200 and j.get("flagged")==False and j.get("decision")=="approve" and j.get("human_review")==False
    results.append(("Q2","happy",200,r.status_code, cond, j))

    # Q3
    payload = {**RISKY, "txn_id":"T800002"}
    r = requests.post(f"{BASE}/score", json=payload)
    j = r.json() if r.status_code==200 else {}
    cond = r.status_code==200 and j.get("flagged")==True and j.get("decision")=="hold_for_review" and j.get("human_review")==True
    results.append(("Q3","escalation",200,r.status_code, cond, j))

    # Q4
    payload = {**BIG, "txn_id":"T800003"}
    r = requests.post(f"{BASE}/score", json=payload)
    j = r.json() if r.status_code==200 else {}
    cond = r.status_code==200 and j.get("flagged")==False and j.get("decision")=="hold_for_review" and j.get("human_review")==True
    results.append(("Q4","escalation",200,r.status_code, cond, j))

    # Q5
    payload = {**BENIGN, "txn_id":"T800004", "amount_inr":-5}
    r = requests.post(f"{BASE}/score", json=payload)
    j = r.json() if r.status_code==400 else {}
    cond = r.status_code==400 and j.get("field")=="amount_inr"
    results.append(("Q5","edge",400,r.status_code, cond, j))

    # Q6
    payload = {**BENIGN, "txn_id":"T800005"}
    del payload["hour_of_day"]
    r = requests.post(f"{BASE}/score", json=payload)
    cond = r.status_code==422
    results.append(("Q6","edge",422,r.status_code, cond, {}))

    # Q7
    payload = {**RISKY, "txn_id":"T900002", "question":"more than 6 UPI payments in 24 hours velocity alert"}
    r = requests.post(f"{BASE}/case", json=payload)
    j = r.json() if r.status_code==200 else {}
    cond = r.status_code==200 and j.get("flagged")==True and j.get("attempts")==2 and j.get("safe_default")==False and (j.get("case_note") or {}).get("policy_id")=="P2" and j.get("human_review")==True
    results.append(("Q7","escalation",200,r.status_code, cond, j))

    # Q8
    payload = {**RISKY, "txn_id":"T900005", "question":"electronics merchant payment on an account younger than 30 days"}
    r = requests.post(f"{BASE}/case", json=payload)
    j = r.json() if r.status_code==200 else {}
    cond = r.status_code==200 and j.get("attempts")==3 and j.get("safe_default")==True and j.get("decision")=="hold_for_review" and j.get("human_review")==True
    results.append(("Q8","escalation",200,r.status_code, cond, j))

    # Q9
    payload = {**RISKY, "txn_id":"T900099", "question":"card payment above INR 50000 from a new device"}
    r = requests.post(f"{BASE}/case", json=payload)
    cond = r.status_code==503
    results.append(("Q9","edge",503,r.status_code, cond, r.json() if r.status_code==503 else {}))

    # Q10
    payload = {**BENIGN, "txn_id":"T800001", "question":"card payment above INR 50000 from a new device"}
    r = requests.post(f"{BASE}/case", json=payload)
    j = r.json() if r.status_code==200 else {}
    cond = r.status_code==200 and j.get("case_note") is None and j.get("attempts")==0 and j.get("decision")=="approve" and j.get("human_review")==False
    results.append(("Q10","happy",200,r.status_code, cond, j))

    # Q11
    payload = {**RISKY, "txn_id":"T800006", "question":"what is the cafeteria menu today"}
    r = requests.post(f"{BASE}/case", json=payload)
    j = r.json() if r.status_code==200 else {}
    cond = r.status_code==200 and j.get("case_note") is None and j.get("attempts")==0 and j.get("decision")=="hold_for_review" and j.get("human_review")==True
    results.append(("Q11","edge",200,r.status_code, cond, j))

    # print table
    print("\nId | Category | Expected | Observed | PASS/FAIL")
    for res in results:
        id_, cat, exp, obs, ok, *_ = res
        print(f"{id_} | {cat} | {exp} | {obs} | {'PASS' if ok else 'FAIL'}")

    # per category
    from collections import Counter
    cat_pass = {}
    cat_total = {}
    for res in results:
        cat = res[1]
        ok = res[4]
        cat_total[cat] = cat_total.get(cat,0)+1
        cat_pass[cat] = cat_pass.get(cat,0)+(1 if ok else 0)
    print("\nPer-category:")
    for cat in cat_total:
        print(f"{cat}: {cat_pass[cat]}/{cat_total[cat]}")

    # metrics
    r = requests.get(f"{BASE}/metrics")
    print("\nMetrics:", r.json())

if __name__ == "__main__":
    run()
