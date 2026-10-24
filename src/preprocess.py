import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/Friday-WorkingHours-Morning.pcap_ISCX (1).csv")
OUTPUT_PATH = Path("data/cleaned_network_data.csv")

print("Loading dataset...")

df = pd.read_csv(DATA_PATH, encoding="latin1", low_memory=False)

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Replace infinite values with missing values
df.replace([float("inf"), float("-inf")], float("nan"), inplace=True)

# Remove duplicate records
before = len(df)
df.drop_duplicates(inplace=True)
print("Duplicate rows removed:", before - len(df))

# Clean numeric columns
numeric_cols = df.select_dtypes(include="number").columns
df[numeric_cols] = df[numeric_cols].fillna(
    df[numeric_cols].median()
)

# Remove rows with missing labels
df.dropna(subset=["Label"], inplace=True)

# Save cleaned data
df.to_csv(OUTPUT_PATH, index=False)

print("\n✅ Dataset cleaned successfully!")
print("Cleaned shape:", df.shape)
print("Remaining missing values:", df.isnull().sum().sum())
print("\nTraffic labels:")
print(df["Label"].value_counts())
print("\nSaved to:", OUTPUT_PATH)