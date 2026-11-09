
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

# 1. Set project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = BASE_DIR / "models" / "best_churn_model.pkl"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)

# 2. Load dataset
df = pd.read_csv(DATA_PATH)

# Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Remove customer ID
df = df.drop(columns=["customerID"])

# Separate features and target
X = df.drop(columns=["Churn"])
y = df["Churn"].map({"Yes": 1, "No": 0})

# 3. Recreate the same test split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Load the trained model
model = joblib.load(MODEL_PATH)

# 5. Generate predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 6. Calculate evaluation metrics
print("\n========== MODEL EVALUATION ==========")

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
print(f"PR-AUC:    {average_precision_score(y_test, y_prob):.4f}")

print("\n========== CLASSIFICATION REPORT ==========")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Not Churned", "Churned"],
        zero_division=0
    )
)

# 7. Confusion Matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["Not Churned", "Churned"],
    cmap="Blues"
)
plt.title("Customer Churn - Confusion Matrix")
plt.tight_layout()
plt.savefig(REPORT_DIR / "confusion_matrix.png")
plt.show()

# 8. ROC Curve
RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("Customer Churn - ROC Curve")
plt.tight_layout()
plt.savefig(REPORT_DIR / "roc_curve.png")
plt.show()

# 9. Precision-Recall Curve
PrecisionRecallDisplay.from_predictions(y_test, y_prob)
plt.title("Customer Churn - Precision-Recall Curve")
plt.tight_layout()
plt.savefig(REPORT_DIR / "precision_recall_curve.png")
plt.show()

print("\nEvaluation completed!")
print(f"Charts saved in: {REPORT_DIR}")
