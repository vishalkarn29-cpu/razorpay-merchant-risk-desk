
import pandas as pd
import sys

def clean_transactions(raw_df):
    df = raw_df.copy()
    # (a) drop exact duplicates
    df = df.drop_duplicates()
    # (b) strip and title case
    for col in ["merchant_category", "city"]:
        df[col] = df[col].astype(str).apply(lambda x: x.strip().title() if isinstance(x, str) else x)
        # But need to handle non-string? Actually after astype str, missing becomes "nan" -> avoid
    # Better: redo correctly handling NaN
    # Let's re-implement cleanly
    return df

def clean_transactions_v2(raw_df):
    df = raw_df.copy()
    df = df.drop_duplicates()
    # strip and title case
    df["merchant_category"] = df["merchant_category"].apply(lambda x: x.strip().title() if isinstance(x, str) else x)
    df["city"] = df["city"].apply(lambda x: x.strip().title() if isinstance(x, str) else x)
    # drop missing amount
    df = df[pd.notna(df["amount_inr"]) & (df["amount_inr"]!="")]
    # convert amount to numeric (it may be string "")
    df["amount_inr"] = pd.to_numeric(df["amount_inr"], errors='coerce')
    df = df.dropna(subset=["amount_inr"])
    # drop out of range
    df = df[(df["amount_inr"]>=1) & (df["amount_inr"]<=1_000_000)]
    df = df.reset_index(drop=True)
    return df

# Overwrite with correct logic
def clean_transactions(raw_df):
    df = raw_df.copy()
    # (a)
    df = df.drop_duplicates()
    # (b)
    df["merchant_category"] = df["merchant_category"].apply(lambda x: x.strip().title() if isinstance(x, str) else x)
    df["city"] = df["city"].apply(lambda x: x.strip().title() if isinstance(x, str) else x)
    # (c) drop missing amount
    # amount may be "" string
    df["amount_inr"] = pd.to_numeric(df["amount_inr"], errors='coerce')
    df = df.dropna(subset=["amount_inr"])
    # (d) drop out of range
    df = df[(df["amount_inr"]>=1) & (df["amount_inr"]<=1_000_000)]
    # (e)
    df = df.reset_index(drop=True)
    return df

def add_features(df):
    out = df.copy()
    out["amount_vs_avg_ratio"] = out["amount_inr"] / out["avg_amount_30d_inr"]
    out["is_night"] = out["hour_of_day"].apply(lambda h: 1 if h<=5 else 0)
    out["high_value"] = out["amount_inr"].apply(lambda a: 1 if a>50000 else 0)
    return out

if __name__ == "__main__":
    raw = pd.read_csv("transactions_raw.csv")
    rows, cols = raw.shape
    dup = raw.duplicated().sum()
    print(f"Raw rows: {rows}, cols: {cols}, duplicates: {dup}")
    # after dedup for audit
    dedup = raw.drop_duplicates()
    # missing after dedup
    # amount missing: pd to numeric coercion
    dedup_amount_numeric = pd.to_numeric(dedup["amount_inr"], errors='coerce')
    missing_amount = dedup_amount_numeric.isna().sum()
    missing_dist = dedup["distance_from_home_km"].apply(lambda x: 1 if (x=="" or pd.isna(x)) else 0).sum() if dedup["distance_from_home_km"].dtype==object else dedup["distance_from_home_km"].isna().sum()
    # For accurate: count where distance is "" or NaN
    missing_dist_count = 0
    for v in dedup["distance_from_home_km"]:
        if v=="" or (isinstance(v,float) and pd.isna(v)):
            missing_dist_count+=1
        elif pd.isna(v):
            missing_dist_count+=1
    # out of range
    amount_numeric = pd.to_numeric(dedup["amount_inr"], errors='coerce')
    out_range = ((amount_numeric<1) | (amount_numeric>1_000_000)).sum()
    print(f"After dedup: rows {len(dedup)}, missing amount {missing_amount}, missing distance {missing_dist_count}, out-of-range {out_range}")

    # distinct spellings raw
    print(f"Distinct merchant_category raw: {raw['merchant_category'].nunique()}, city raw: {raw['city'].nunique()}")

    cleaned = clean_transactions(raw)
    print(f"Cleaned rows: {len(cleaned)}")
    print(f"merchant_category distinct: {cleaned['merchant_category'].nunique()}")
    print(cleaned['merchant_category'].value_counts().to_dict())
    print(f"city distinct: {cleaned['city'].nunique()}")
    print(cleaned['payment_mode'].value_counts().to_dict())
    print(f"amount min {cleaned['amount_inr'].min()}, max {cleaned['amount_inr'].max()}")
    print(f"missing distance after clean: {cleaned['distance_from_home_km'].apply(lambda x: pd.isna(pd.to_numeric(x, errors='coerce')) if x=='' else pd.isna(x)).sum() if False else pd.to_numeric(cleaned['distance_from_home_km'], errors='coerce').isna().sum()}")

    # better distance missing count
    dist_missing = pd.to_numeric(cleaned['distance_from_home_km'], errors='coerce').isna().sum()
    print(f"dist missing cleaned: {dist_missing}")

    feat = add_features(cleaned)
    print(f"is_night sum {feat['is_night'].sum()}, high_value sum {feat['high_value'].sum()}, max ratio {feat['amount_vs_avg_ratio'].max():.2f}")

    # fraud
    fraud_rows = feat['is_fraud'].sum()
    fraud_rate = fraud_rows/len(feat)*100
    print(f"fraud rows {fraud_rows} / {len(feat)} = {fraud_rate:.2f}%")
    acc = (len(feat)-fraud_rows)/len(feat)
    print(f"never-fraud accuracy {acc:.4f} recall 0")

    # leakage table
    charge1 = feat[feat['chargeback_filed']==1]
    charge0 = feat[feat['chargeback_filed']==0]
    print(f"chargeback=1 rows {len(charge1)} fraud {charge1['is_fraud'].sum()}")
    print(f"chargeback=0 rows {len(charge0)} fraud {charge0['is_fraud'].sum()}")

    print("Part A checkpoints OK")
