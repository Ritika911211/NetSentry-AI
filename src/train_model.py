import pandas as pd
import numpy as np
from pathlib import Path
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

DATA_PATH = Path("data/cleaned_network_data.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

print("Loading cleaned dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)

# Prepare binary target: 0 = BENIGN, 1 = Bot
y = (df["Label"].str.strip() != "BENIGN").astype(int)

# Remove label columns from input features
X = df.drop(columns=["Label", "Target"], errors="ignore")

# Convert features to numeric and clean invalid values
X = X.apply(pd.to_numeric, errors="coerce")
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X = X.fillna(X.median()).fillna(0)

# Remove constant columns
X = X.loc[:, X.nunique() > 1]

print("Features:", X.shape[1])
print("Preparing train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, predictions), 4))
print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Normal traffic", "Bot traffic"],
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

# Save model and feature names
dump(model, MODEL_DIR / "random_forest.joblib")
dump(list(X.columns), MODEL_DIR / "feature_names.joblib")

print("\n✅ Model and feature names saved in models/!")