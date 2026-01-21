# Databricks notebook source
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from xgboost import XGBClassifier

# -----------------------------
# 0) Force MLflow LOCAL tracking (Databricks CE safe)
# -----------------------------
mlflow.set_tracking_uri("/Workspace/Users/rcoundinya@gmail.com/stockout-prediction-mlops/tracking-runs")
print("Tracking URI:", mlflow.get_tracking_uri())

experiment_name = "stockout-mlops-local"
mlflow.set_experiment(experiment_name)
print("Experiment:", experiment_name)

# -----------------------------
# 1) Load feature table
# -----------------------------
feature_table = "workspace.default.retail_stockout_features"
df_feat = spark.table(feature_table)

feature_cols = [
    "price", "promo", "daily_sales", "inventory",
    "sales_lag_1", "sales_lag_3", "sales_lag_7",
    "inv_lag_1", "promo_lag_1",
    "sales_roll7_avg", "sales_roll7_sum", "sales_roll14_avg",
    "promo_roll7_sum",
    "inv_to_sales_ratio", "low_inventory_flag"
]

label_col = "label_stockout_next_7d"

df_ml = df_feat.select(*(feature_cols + [label_col]))

print("Training rows:", df_ml.count())
display(df_ml.limit(5))

# -----------------------------
# 2) Convert to Pandas (safe sampling if needed)
# -----------------------------
# If your cluster struggles, reduce sample to 200k
sample_n = 200000
pdf = df_ml.limit(sample_n).toPandas()

X = pdf[feature_cols]
y = pdf[label_col].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape)
print("Test size :", X_test.shape)

# -----------------------------
# 3) Train model + log to MLflow
# -----------------------------
with mlflow.start_run(run_name="xgboost_stockout_next7d"):

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("AUC:", auc)
    print("F1:", f1)
    print("Precision:", precision)
    print("Recall:", recall)

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    mlflow.log_params({
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.08,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "sample_n": sample_n
    })

    mlflow.log_metrics({
        "auc": float(auc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall)
    })

    # Log confusion matrix as artifact
    cm_df = pd.DataFrame(cm, index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"])
    cm_path = "/tmp/confusion_matrix.csv"
    cm_df.to_csv(cm_path, index=True)
    mlflow.log_artifact(cm_path)

    # Log model artifact
    mlflow.sklearn.log_model(model, artifact_path="model")

    print("\n✅ Model + metrics logged to LOCAL MLflow successfully!")

# COMMAND ----------

