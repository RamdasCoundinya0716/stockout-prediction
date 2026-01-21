# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql import types as T
import random

random.seed(42)

stores = [f"S{i:03d}" for i in range(1, 21)]        # 20 stores
products = [f"P{i:04d}" for i in range(1, 301)]     # 300 products

days = 180
start_date = "2025-01-01"

schema = T.StructType([
    T.StructField("store_id", T.StringType(), False),
    T.StructField("product_id", T.StringType(), False),
    T.StructField("day_offset", T.IntegerType(), False),
    T.StructField("price", T.DoubleType(), False),
    T.StructField("promo", T.IntegerType(), False),
    T.StructField("daily_sales", T.IntegerType(), False),
    T.StructField("inventory", T.IntegerType(), False),
    T.StructField("stockout", T.IntegerType(), False)
])

rows = []
for store in stores:
    for product in products:
        inventory = random.randint(20, 100)
        base_price = round(random.uniform(50, 500), 2)

        for d in range(days):
            promo = 1 if random.random() < 0.08 else 0
            price = base_price * (0.9 if promo else 1.0)

            base_demand = random.randint(1, 8)
            promo_boost = random.randint(2, 8) if promo else 0
            noise = random.randint(0, 3)

            daily_sales = base_demand + promo_boost + noise

            inventory = inventory - daily_sales

            if inventory <= 0:
                stockout = 1
                inventory = 0
            else:
                stockout = 0

            # restock event
            if random.random() < 0.05:
                inventory += random.randint(20, 120)

            rows.append((store, product, d, float(price), int(promo), int(daily_sales), int(inventory), int(stockout)))

df = spark.createDataFrame(rows, schema=schema)

#Create actual date from day_offset
df = df.withColumn(
    "date",
    F.expr(f"date_add(to_date('{start_date}'), day_offset)")
).drop("day_offset")

print("Row count:", df.count())
display(df.limit(10))

# COMMAND ----------

# DBTITLE 1,Cell 2
table_name = "workspace.default.retail_stockout_data"
df.write.format("delta").mode("overwrite").saveAsTable(table_name)
print("Saved as table:", table_name)

# COMMAND ----------

display(spark.table("workspace.default.retail_stockout_data").limit(5))

# COMMAND ----------

