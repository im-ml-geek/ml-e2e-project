from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Read the CSV and save the model next to this script.
project_dir = Path(__file__).resolve().parent
data_path = project_dir / "course_lead_scoring_2026.csv"
model_path = project_dir / "lead_conversion_model.joblib"

# Use the settings already selected in the notebook.
selected_c = 1.0
cutoff = 0.40

# Separate the model inputs from the actual conversion outcome.
df = pd.read_csv(data_path)
X = df.drop(columns=["converted", "lead_score"])
y = df["converted"]

# Set aside 1,000 leads for the final test.
X_remaining, X_test, y_remaining, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# Split the remaining 4,000 leads into 3,000 for training and 1,000 for validation.
X_train, X_val, y_train, y_val = train_test_split(
    X_remaining, y_remaining,
    test_size=0.25, stratify=y_remaining, random_state=42
)

numeric_cols = ["annual_income", "number_of_courses_viewed", "interaction_count"]
categorical_cols = ["lead_source", "industry", "employment_status", "location"]

# Fill missing numbers with the training median, then scale the numbers.
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# Label missing categories and turn known categories into 0/1 columns.
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols),
])

# Learn the preprocessing values and model weights from the training leads.
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(C=selected_c, max_iter=1000)),
])
model.fit(X_train, y_train)

# Use probabilities for ROC AUC and apply the cutoff for F1 and the confusion matrix.
test_prob = model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= cutoff).astype(int)

print(f"Test ROC AUC: {roc_auc_score(y_test, test_prob):.3f}")
print(f"Test F1:      {f1_score(y_test, test_pred):.3f}")
print("Confusion matrix:")
print(confusion_matrix(y_test, test_pred, labels=[0, 1]))

# Save the fitted pipeline and the decision cutoff together.
joblib.dump({"model": model, "cutoff": cutoff}, model_path)
print(f"Model saved to {model_path}")