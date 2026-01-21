# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# -----------------------------
# 0) Read base data from Delta table
# -----------------------------
table_name = "workspace.default.retail_stockout_data"
df = spark.table(table_name)

# Ensure correct date type
df = df.withColumn("date", F.to_date("date"))

# Window by store+product ordered by date
w = Window.partitionBy("store_id", "product_id").orderBy("date")

# -----------------------------
# 1) Lag features (sales / inventory / promo)
# -----------------------------
df_feat = df \
    .withColumn("sales_lag_1", F.lag("daily_sales", 1).over(w)) \
    .withColumn("sales_lag_3", F.lag("daily_sales", 3).over(w)) \
    .withColumn("sales_lag_7", F.lag("daily_sales", 7).over(w)) \
    .withColumn("inv_lag_1", F.lag("inventory", 1).over(w)) \
    .withColumn("promo_lag_1", F.lag("promo", 1).over(w))

# -----------------------------
# 2) Rolling features (7-day and 14-day)
# -----------------------------
w7  = w.rowsBetween(-6, 0)     # last 7 days including today
w14 = w.rowsBetween(-13, 0)    # last 14 days including today

df_feat = df_feat \
    .withColumn("sales_roll7_avg", F.avg("daily_sales").over(w7)) \
    .withColumn("sales_roll7_sum", F.sum("daily_sales").over(w7)) \
    .withColumn("sales_roll14_avg", F.avg("daily_sales").over(w14)) \
    .withColumn("promo_roll7_sum", F.sum("promo").over(w7))

# -----------------------------
# 3) Inventory pressure features
# -----------------------------
df_feat = df_feat \
    .withColumn(
        "inv_to_sales_ratio",
        F.col("inventory") / (F.col("sales_roll7_avg") + F.lit(1.0))
    ) \
    .withColumn("low_inventory_flag", F.when(F.col("inventory") <= 10, 1).otherwise(0))

# -----------------------------
# 4) Label: will stockout happen in NEXT 7 days?
# -----------------------------
future_window_7 = w.rowsBetween(1, 7)

df_feat = df_feat.withColumn(
    "label_stockout_next_7d",
    F.max("stockout").over(future_window_7)
)

# -----------------------------
# 5) Drop nulls caused by lag/rolling windows
# -----------------------------
df_feat = df_feat.dropna(subset=[
    "sales_lag_1", "sales_lag_3", "sales_lag_7",
    "sales_roll7_avg", "sales_roll14_avg",
    "label_stockout_next_7d"
])

print("Final feature rows:", df_feat.count())
display(df_feat.limit(10))

# -----------------------------
# 6) Select final columns (safe save)
# -----------------------------
selected_cols = [
    "store_id", "product_id", "date",
    "price", "promo", "daily_sales", "inventory", "stockout",
    "sales_lag_1", "sales_lag_3", "sales_lag_7",
    "inv_lag_1", "promo_lag_1",
    "sales_roll7_avg", "sales_roll7_sum", "sales_roll14_avg",
    "promo_roll7_sum",
    "inv_to_sales_ratio", "low_inventory_flag",
    "label_stockout_next_7d"
]

df_feat_final = df_feat.select(*selected_cols)

# -----------------------------
# 7) Save as Delta table
# -----------------------------
feature_table = "workspace.default.retail_stockout_features"

df_feat_final.write.format("delta").mode("overwrite").saveAsTable(feature_table)

print("Saved feature table:", feature_table)

# Quick check
display(spark.table(feature_table).limit(5))

# Label distribution check (important)
spark.table(feature_table) \
    .groupBy("label_stockout_next_7d") \
    .count() \
    .show()