# Databricks notebook source
# MAGIC %pip install xgboost

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from pyspark.sql import functions as F

# -----------------------------
# 0) Use LOCAL MLflow tracking (Databricks CE safe)
# -----------------------------
mlflow.set_tracking_uri("/Workspace/Users/rcoundinya@gmail.com/stockout-prediction-mlops/tracking-runs")
print("Tracking URI:", mlflow.get_tracking_uri())

# -----------------------------
# 1) Load features table (source)
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

# -----------------------------
# 2) Pick only latest 30 days for scoring (realistic batch scoring)
# -----------------------------
max_date = df_feat.select(F.max("date")).collect()[0][0]
print("Max date in features:", max_date)

df_score = df_feat.filter(F.col("date") >= F.date_sub(F.lit(max_date), 30))

print("Scoring rows:", df_score.count())
display(df_score.limit(5))

# -----------------------------
# 3) Load latest model safely (NO experiment name dependency)
# -----------------------------
client = mlflow.tracking.MlflowClient()

experiments = client.search_experiments()
print("Experiments found:", [e.name for e in experiments])

latest_run = None

for exp in experiments:
    runs = client.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=1)
    if runs:
        if (latest_run is None) or (runs[0].info.start_time > latest_run.info.start_time):
            latest_run = runs[0]

if latest_run is None:
    raise Exception("❌ No MLflow runs found in file:/tmp/mlruns. Run training notebook first.")

latest_run_id = latest_run.info.run_id
print("✅ Latest Run ID:", latest_run_id)

model_uri = f"runs:/{latest_run_id}/model"
model = mlflow.sklearn.load_model(model_uri)
print("✅ Loaded model from:", model_uri)

# -----------------------------
# 4) Convert scoring data to Pandas + predict
# -----------------------------
pdf_score = df_score.select(
    "store_id", "product_id", "date",
    *feature_cols
).toPandas()

X_score = pdf_score[feature_cols]

pdf_score["stockout_risk_score"] = model.predict_proba(X_score)[:, 1]
pdf_score["stockout_risk_label"] = (pdf_score["stockout_risk_score"] >= 0.5).astype(int)

# -----------------------------
# 5) Convert back to Spark and save predictions as Delta table
# -----------------------------
pred_sdf = spark.createDataFrame(pdf_score)

pred_table = "workspace.default.retail_stockout_predictions"

pred_sdf.write.format("delta").mode("overwrite").saveAsTable(pred_table)

print("✅ Saved predictions table:", pred_table)
display(spark.table(pred_table).limit(10))

# -----------------------------
# 6) Quick sanity: score distribution
# -----------------------------
spark.table(pred_table) \
    .groupBy("stockout_risk_label") \
    .count() \
    .show()

# COMMAND ----------

