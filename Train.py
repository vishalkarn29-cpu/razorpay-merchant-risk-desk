
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from prepare import clean_transactions, add_features

def get_encoded_df():
    raw = pd.read_csv("transactions_raw.csv")
    cleaned = clean_transactions(raw)
    feat = add_features(cleaned)
    # columns to exclude
    exclude = ["txn_id","customer_id","chargeback_filed","is_fraud"]
    X_raw = feat.drop(columns=exclude)
    # one-hot
    X = pd.get_dummies(X_raw, dtype=int)
    # pinned order: 7 numeric, 3 engineered, then 16 one-hot alphabetical
    # get_dummies already produces alphabetical for each categorical? Let's enforce pinned order
    # numeric order in file order: amount_inr, hour_of_day, txn_count_last_24h, avg_amount_30d_inr, distance_from_home_km, is_new_device, account_age_days
    # engineered: amount_vs_avg_ratio, is_night, high_value
    # one-hot: merchant_category alphabetical, city alphabetical, payment_mode alphabetical
    numeric = ["amount_inr","hour_of_day","txn_count_last_24h","avg_amount_30d_inr","distance_from_home_km","is_new_device","account_age_days"]
    engineered = ["amount_vs_avg_ratio","is_night","high_value"]
    # find one-hot columns
    cat_cols = [c for c in X.columns if c.startswith("merchant_category_")]
    city_cols = [c for c in X.columns if c.startswith("city_")]
    mode_cols = [c for c in X.columns if c.startswith("payment_mode_")]
    cat_cols = sorted(cat_cols)
    city_cols = sorted(city_cols)
    mode_cols = sorted(mode_cols)
    pinned = numeric + engineered + cat_cols + city_cols + mode_cols
    X = X[pinned]
    y = feat["is_fraud"]
    return feat, X, y, pinned

def split_and_fill():
    feat, X, y, pinned = get_encoded_df()
    # stratified 60/20/20
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    # median from training
    median = X_train["distance_from_home_km"].median()
    # fill
    for df in [X_train, X_val, X_test]:
        df["distance_from_home_km"] = df["distance_from_home_km"].fillna(median)
    # Need also fill for X_temp? Already done via references, but we modified copies? Actually X_train etc are copies.
    return X_train, X_val, X_test, y_train, y_val, y_test, median, pinned, feat

def eval_at_threshold(y_true, y_prob, thresh):
    y_pred = (y_prob>=thresh).astype(int)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return prec, rec, f1, tn, fp, fn, tp, (y_pred.sum())

def build_final_model():
    X_train, X_val, X_test, y_train, y_val, y_test, median, pinned, feat = split_and_fill()
    # grid search
    param_grid = {
        "max_depth": [4,6,8,None],
        "min_samples_leaf": [5,10,20]
    }
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(rf, param_grid, scoring="f1", cv=cv)
    grid.fit(X_train, y_train)
    best_params = grid.best_params_
    best_score = grid.best_score_
    model = grid.best_estimator_
    # threshold
    y_val_prob = model.predict_proba(X_val)[:,1]
    best_t = 0.1
    for t in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        _, rec, _, _, _, _, _, _ = eval_at_threshold(y_val, y_val_prob, t)
        if rec>=0.65:
            best_t = t
    # if none, best_t stays 0.1 per spec (but our loop already picks highest)
    # To get highest with rec>=0.65, iterate increasing and keep last that qualifies
    qualified = [t for t in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] if eval_at_threshold(y_val, y_val_prob, t)[1]>=0.65]
    if qualified:
        best_t = max(qualified)
    else:
        best_t = 0.1
    return {"model": model, "columns": pinned, "median": median, "threshold": best_t, "best_params": best_params, "best_score": best_score}

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, median, pinned, feat = split_and_fill()
    print(f"Split sizes: train {len(X_train)} val {len(X_val)} test {len(X_test)}")
    print(f"Fraud counts: train {y_train.sum()} val {y_val.sum()} test {y_test.sum()}")
    print(f"Training median distance: {median}")
    print(f"Columns ({len(pinned)}): {pinned}")

    # Three imbalance models
    def make_pipe(class_weight=None):
        return Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, class_weight=class_weight))])

    models = {}
    # plain
    pipe_plain = make_pipe()
    pipe_plain.fit(X_train, y_train)
    # balanced
    pipe_bal = make_pipe(class_weight="balanced")
    pipe_bal.fit(X_train, y_train)
    # oversampled
    fraud_train = X_train[y_train==1]
    non_fraud_train = X_train[y_train==0]
    y_fraud = y_train[y_train==1]
    y_non = y_train[y_train==0]
    # sample fraud to match non-fraud count
    fraud_over = fraud_train.sample(n=len(non_fraud_train), replace=True, random_state=42)
    y_fraud_over = y_fraud.sample(n=len(non_fraud_train), replace=True, random_state=42)
    X_over = pd.concat([non_fraud_train, fraud_over])
    y_over = pd.concat([y_non, y_fraud_over])
    pipe_over = make_pipe()
    pipe_over.fit(X_over, y_over)
    print(f"Oversampled rows: {len(X_over)} (non-fraud {len(non_fraud_train)} fraud {len(fraud_over)})")

    for name, pipe in [("plain", pipe_plain), ("balanced", pipe_bal), ("oversampled", pipe_over)]:
        prob = pipe.predict_proba(X_val)[:,1]
        prec, rec, f1, tn, fp, fn, tp, flagged = eval_at_threshold(y_val, prob, 0.5)
        print(f"{name} @0.5: prec {prec:.3f} rec {rec:.3f} f1 {f1:.3f} TN {tn} FP {fp} FN {fn} TP {tp}")

    # CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # balanced logistic CV
    from sklearn.model_selection import cross_val_score
    pipe_bal_cv = make_pipe(class_weight="balanced")
    scores = cross_val_score(pipe_bal_cv, X_train, y_train, cv=cv, scoring="f1")
    print(f"Balanced LR CV mean {scores.mean():.3f} std {scores.std():.3f}")

    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    scores_rf = cross_val_score(rf, X_train, y_train, cv=cv, scoring="f1")
    print(f"Untuned RF CV mean {scores_rf.mean():.3f} std {scores_rf.std():.3f}")

    # Grid
    param_grid = {"max_depth":[4,6,8,None], "min_samples_leaf":[5,10,20]}
    grid = GridSearchCV(RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42), param_grid, scoring="f1", cv=cv)
    grid.fit(X_train, y_train)
    print(f"Best params {grid.best_params_} best CV F1 {grid.best_score_:.3f}")

    # Threshold table
    model = grid.best_estimator_
    val_prob = model.predict_proba(X_val)[:,1]
    print("Threshold table validation:")
    for t in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        prec, rec, f1, tn, fp, fn, tp, flagged = eval_at_threshold(y_val, val_prob, t)
        print(f"t={t}: prec {prec:.3f} rec {rec:.3f} f1 {f1:.3f} flagged {flagged}")

    qualified = [t for t in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] if eval_at_threshold(y_val, val_prob, t)[1]>=0.65]
    t_star = max(qualified) if qualified else 0.1
    print(f"t_star {t_star}")

    # Test
    test_prob = model.predict_proba(X_test)[:,1]
    prec, rec, f1, tn, fp, fn, tp, flagged = eval_at_threshold(y_test, test_prob, t_star)
    print(f"Test @t_star: prec {prec:.3f} rec {rec:.3f} f1 {f1:.3f} flagged {flagged} TN {tn} FP {fp} FN {fn} TP {tp}")

    # Leakage
    # add chargeback as 27th column
    feat_full, X_full, y_full, pinned_full = get_encoded_df()
    # need chargeback column from feat
    charge = feat_full["chargeback_filed"]
    # split same way but with extra column
    # reuse earlier split indices: we need to redo with same random_state
    X_train_l, X_val_l, X_test_l, y_train_l, y_val_l, y_test_l, median_l, _, _ = split_and_fill()
    # add chargeback - need to get charge aligned to same splits? Use feat split same as before? Simpler: merge by index? Actually get_encoded_df returns feat aligned with X
    # For leakage we create X with chargeback as extra column
    # Recreate split for leaky: use X_full plus charge
    X_leaky = X_full.copy()
    X_leaky["chargeback_filed"] = feat_full["chargeback_filed"].values
    # reorder to put charge last
    X_leaky = X_leaky[pinned + ["chargeback_filed"]]
    X_train_l2, X_test_tmp, y_train_l2, y_test_tmp = train_test_split(X_leaky, y_full, test_size=0.4, stratify=y_full, random_state=42)
    X_val_l2, X_test_l2, y_val_l2, y_test_l2 = train_test_split(X_test_tmp, y_test_tmp, test_size=0.5, stratify=y_test_tmp, random_state=42)
    for df in [X_train_l2, X_val_l2, X_test_l2]:
        df["distance_from_home_km"] = df["distance_from_home_km"].fillna(median_l)
    rf_clean = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, max_depth=grid.best_params_["max_depth"], min_samples_leaf=grid.best_params_["min_samples_leaf"])
    rf_clean.fit(X_train_l, y_train_l)
    prob_clean = rf_clean.predict_proba(X_val_l)[:,1]
    prec_c, rec_c, f1_c, *_ = eval_at_threshold(y_val_l, prob_clean, 0.5)
    rf_leaky = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, max_depth=grid.best_params_["max_depth"], min_samples_leaf=grid.best_params_["min_samples_leaf"])
    rf_leaky.fit(X_train_l2, y_train_l2)
    prob_leaky = rf_leaky.predict_proba(X_val_l2)[:,1]
    prec_l, rec_l, f1_l, *_ = eval_at_threshold(y_val_l2, prob_leaky, 0.5)
    print(f"Leakage clean F1 {f1_c:.3f} leaky F1 {f1_l:.3f}")

    # importance
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    print("Top 5 features:")
    for i in idx[:5]:
        print(f"{pinned[i]}: {importances[i]:.4f}")
