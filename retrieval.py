
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open("fraud_policies.json") as f:
    POLICIES = json.load(f)

texts = [p["text"] for p in POLICIES]
ids = [p["policy_id"] for p in POLICIES]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)

def retrieve(query):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, tfidf_matrix)[0]
    best_idx = int(sims.argmax())
    return ids[best_idx], texts[best_idx], float(sims[best_idx])

# Calibrate MIN_SCORE
with open("retrieval_queries.json") as f:
    queries = json.load(f)

scores = []
for q in queries:
    pid, txt, score = retrieve(q["query"])
    scores.append((q["query"], q["expected_policy"], pid, score))

in_scope_scores = [s[3] for s in scores if s[1] is not None]
out_scope_scores = [s[3] for s in scores if s[1] is None]
min_in = min(in_scope_scores) if in_scope_scores else 0
max_out = max(out_scope_scores) if out_scope_scores else 0
# Choose MIN_SCORE above max_out and at most min_in
# pick midpoint if possible else just above max_out
MIN_SCORE = (max_out + min_in) / 2
# Ensure within interval
if MIN_SCORE <= max_out:
    MIN_SCORE = max_out + 0.001
if MIN_SCORE > min_in:
    MIN_SCORE = min_in

# For reproducibility, set to 0.45 as it lies in [0.4251, 0.4862] per reference
# But compute actual: reference says out max 0.4251, min in 0.4862
# Our computed may be similar. We'll set to a value that satisfies.
# Let's compute and adjust to be 0.45 if inside.
if max_out < 0.45 <= min_in:
    MIN_SCORE = 0.45

if __name__ == "__main__":
    print("Retrieval table:")
    print("query | expected | retrieved | score")
    hit = 0
    for query, exp, ret, score in scores:
        if exp is not None and exp==ret:
            hit+=1
        print(f"{query} | {exp} | {ret} | {score:.4f}")
    print(f"In-scope hit {hit}/8")
    print(f"max out-of-scope {max_out:.4f}, min in-scope {min_in:.4f}, MIN_SCORE {MIN_SCORE:.4f}")
