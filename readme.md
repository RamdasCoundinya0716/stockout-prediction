# 📦 Stockout Risk Prediction — End-to-End MLOps (Databricks + MLflow + FastAPI + Docker)
## 🚀 Overview

This project demonstrates an **end-to-end MLOps workflow** for predicting **stockout risk** in a retail environment.

It includes:

* **Batch feature engineering** using Spark on Databricks
* **Model training & tracking** using **XGBoost + MLflow**
* **Batch scoring pipeline** to generate daily/weekly stockout risk predictions
* **Real-time inference API** using **FastAPI**
* **Containerization** using Docker (production-style deployment)

---

## 🎯 Problem Statement

Retail businesses often face losses due to:

* running out of inventory unexpectedly (**stockouts**)
* poor replenishment timing
* demand spikes during promotions

### ✅ Goal

Predict:

> **Will this product go out of stock in the next 7 days?**

This helps teams trigger:

* automated replenishment
* reorder recommendations
* store-level alerts and dashboards

---

## 🧠 ML Task

### Target Variable (Label)

`label_stockout_next_7d`

**Definition:**

* `1` → if stockout happens at least once in the **next 7 days**
* `0` → otherwise

---

## 🏗️ Architecture (Batch + Real-time)

```
+------------------------------+
|  Retail Data (Daily)         |
|  store, product, sales, inv  |
+--------------+---------------+
               |
               v
+------------------------------+
| Delta Table (Raw)            |
| workspace.default.           |
| retail_stockout_data         |
+--------------+---------------+
               |
               v
+------------------------------+
| Feature Engineering (Spark)  |
| - lag features               |
| - rolling trends             |
| - inventory pressure         |
| - label: next 7d stockout    |
+--------------+---------------+
               |
               v
+------------------------------+
| Delta Table (Features)       |
| workspace.default.           |
| retail_stockout_features     |
+--------------+---------------+
               |
               v
+------------------------------+
| Training (XGBoost + MLflow)  |
| - log params + metrics       |
| - log model artifact         |
+--------------+---------------+
               |
               v
+------------------------------+
| Batch Inference (30 days)    |
| - stockout_risk_score        |
| - stockout_risk_label        |
+--------------+---------------+
               |
               v
+------------------------------+
| Delta Table (Predictions)    |
| workspace.default.           |
| retail_stockout_predictions  |
+------------------------------+

Real-time mode:
FastAPI (/predict) → loads trained model → returns risk score + label
```

---

## 🧰 Tech Stack

### Data + ML

* **Databricks Community Edition**
* **Apache Spark (PySpark)**
* **Delta Tables**
* **XGBoost**
* **MLflow** (experiment tracking + model artifacts)

### Serving

* **FastAPI**
* **Uvicorn**
* **Docker**

---

## ✅ Data Tables Created (Databricks)

| Table Name                                      | Purpose                            |
| ----------------------------------------------- | ---------------------------------- |
| `workspace.default.retail_stockout_data`        | Raw retail daily data              |
| `workspace.default.retail_stockout_features`    | Feature engineered dataset + label |
| `workspace.default.retail_stockout_predictions` | Batch-scored predictions           |

---

## 📊 Model Performance (XGBoost)

Trained on engineered Spark features and evaluated on a test split.

**Metrics achieved:**

* **AUC:** `0.9930`
* **F1 Score:** `0.9630`
* **Precision:** `0.9523`
* **Recall:** `0.9739`

Confusion Matrix:

```
[[15114  1158]
 [  620 23108]]
```

---

# ✅ How to Run (Databricks Batch Pipeline)

## 1️⃣ Notebook 01 — Data Generation

Creates synthetic retail data and writes to Delta:

✅ `workspace.default.retail_stockout_data`

## 2️⃣ Notebook 02 — Feature Engineering

Creates lags, rolling features, inventory pressure features and label:

✅ `workspace.default.retail_stockout_features`

## 3️⃣ Notebook 03 — Train + MLflow Logging

Trains XGBoost model and logs:

* params
* metrics
* confusion matrix
* model artifact

## 4️⃣ Notebook 04 — Batch Inference

Scores the most recent 30 days and writes predictions to:

✅ `workspace.default.retail_stockout_predictions`

---

# ✅ Real-time Serving (FastAPI + Docker)

## 📁 Folder Structure

```
stockout-serving/
├── app.py
├── schema.py
├── requirements.txt
├── stockout_model.pkl
└── Dockerfile
```

---

## 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

## 2️⃣ Start FastAPI server

```bash
uvicorn app:app --reload --port 8000
```

Check:

* Health: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
* Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 3️⃣ Test API with PowerShell

```powershell
$body = @{
  price = 200
  promo = 1
  daily_sales = 10
  inventory = 5
  sales_lag_1 = 12
  sales_lag_3 = 9
  sales_lag_7 = 8
  inv_lag_1 = 6
  promo_lag_1 = 1
  sales_roll7_avg = 11
  sales_roll7_sum = 77
  sales_roll14_avg = 10
  promo_roll7_sum = 2
  inv_to_sales_ratio = 0.4
  low_inventory_flag = 1
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" `
  -Method POST `
  -ContentType "application/json" `
  -Body $body
```

✅ Sample response:

```json
{
  "stockout_risk_score": 0.99, // This is the probability/confidence (≈ 99%) that the product will stock out in the next 7 days
  "stockout_risk_label": 1
}
```

---

# 🐳 Docker Deployment

## 1️⃣ Build Docker image

```bash
docker build -t stockout-api:1.0 .
```

## 2️⃣ Run container

```bash
docker run -p 8000:8000 stockout-api:1.0
```

## 3️⃣ Verify

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
```

Expected:

```json
{"status":"ok"}
```

---

# 📌 Key Highlights (Why this project is Enterprise-grade)

✅ Feature engineering with time-based lags and rolling metrics
✅ Future label generation (next 7 days stockout prediction)
✅ MLflow experiment logging + model artifact tracking
✅ Batch inference pipeline writing predictions back to Delta
✅ Real-time serving API for production-style inference
✅ Fully Dockerized deployment

---

# 🔥 Next Improvements (Planned / Roadmap)

* Deploy FastAPI service to **Kubernetes**
* Add **Prometheus metrics** and monitoring
* Add **Kafka streaming ingestion + real-time scoring**
* Implement model drift monitoring and alerting
* Add Feature Store integration using **Feast**

---

# 👤 Author

**Ramdas Coundinya VK**
MLOps / Data Engineering | Kubernetes | MLflow | Spark | FastAPI | Docker