import os
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ASSETS_DIR = BASE_DIR / "assets"

ARTIFACTS_DIR.mkdir(exist_ok=True)
ASSETS_DIR.mkdir(exist_ok=True)

DATA_URL = "https://azuremlexamples.blob.core.windows.net/datasets/credit_card/default_of_credit_card_clients.csv"

def train():
    print("1/5: Loading UCI Credit Card Default benchmark...")
    df = pd.read_csv(DATA_URL, skiprows=1)
    df.rename(columns={"default payment next month": "default"}, inplace=True)
    df.drop(columns=["ID"], inplace=True)

    features = [
        "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
    ]
    X, y = df[features], df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scale_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    print("2/5: Fitting Cost-Sensitive XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.04,
        scale_pos_weight=scale_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=42
    )
    model.fit(X_train, y_train)

    print("3/5: Evaluating Metrics...")
    test_probs = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, test_probs)
    print(f"--> Test ROC-AUC: {auc_score:.4f}")
    print(classification_report(y_test, (test_probs >= 0.55).astype(int), target_names=["Non-Default", "Default"]))

    print("4/5: Generating & Saving Visual Plots...")
    # Plot 1: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="#2563eb", lw=2, label=f"XGBoost (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], color="#9ca3af", lw=1.5, linestyle="--")
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold")
    plt.xlabel("False Positive Rate", fontsize=10)
    plt.ylabel("True Positive Rate", fontsize=10)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "roc_curve.png", dpi=300)
    plt.close()

    # Plot 2: SHAP Global Summary
    explainer = shap.TreeExplainer(model)
    sample_X = X_test.sample(500, random_state=42)
    shap_values = explainer(sample_X)
    
    plt.figure(figsize=(9, 6))
    shap.plots.beeswarm(shap_values, max_display=12, show=False)
    plt.title("TreeSHAP Feature Attribution Summary", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "shap_summary.png", dpi=300)
    plt.close()

    print("5/5: Exporting Serialized Artifacts...")
    joblib.dump(model, ARTIFACTS_DIR / "model.joblib")
    joblib.dump(explainer, ARTIFACTS_DIR / "explainer.joblib")
    joblib.dump(features, ARTIFACTS_DIR / "features.joblib")
    print("Training pipeline complete! Model and visual plots ready.")

if __name__ == "__main__":
    train()