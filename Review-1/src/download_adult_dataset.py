from pathlib import Path
from sklearn.datasets import fetch_openml

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "AdultIncome.csv"
DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

print("Downloading Adult Income dataset from OpenML/UCI...")
adult = fetch_openml(name="adult", version=2, as_frame=True)
df = adult.frame
df.to_csv(DATA_PATH, index=False)

print(f"Saved {len(df):,} rows and {df.shape[1]} columns to:")
print(DATA_PATH)
