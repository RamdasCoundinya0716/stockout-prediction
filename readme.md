# 📦 Stockout Risk Prediction System — End-to-End MLOps (Databricks + MLflow + FastAPI + Docker + AKS)

## 🚀 Overview

This project implements a complete **end-to-end MLOps pipeline** to predict **retail stockout risk** for store–product combinations.

It covers the full lifecycle:

* ✅ Data ingestion & feature engineering using **Spark on Databricks**
* ✅ Model training using **XGBoost**
* ✅ Experiment tracking + artifact logging using **MLflow**
* ✅ Batch inference + predictions written back into **Delta tables**
* ✅ Real-time inference API using **FastAPI**
* ✅ Containerization using **Docker**
* ✅ Kubernetes deployment manifests (**Deployment + Service**)
* ✅ Public cloud deployment on **Azure Kubernetes Service (AKS)**
* ✅ CI/CD using **GitHub Actions** (auto deploy to AKS on push)

---

## 🎯 Problem Statement

Retail businesses face operational and revenue loss due to:

* unexpected stockouts
* poor replenishment planning
* demand spikes caused by promotions

### ✅ Goal

Predict:

> **Will this product stock out in the next 7 days?**

This enables:

* reorder automation
* alerts & notifications
* inventory dashboards
* proactive supply planning

---

## 🧠 ML Task Definition

### Target Label

`label_stockout_next_7d`

**Definition**

* `1` → if stockout occurs at least once in the **next 7 days**
* `0` → otherwise

---

## 🏗️ Architecture

```
+------------------------------+
| Retail Daily Data            |
| store, product, sales, inv   |
+--------------+---------------+
               |
               v
+------------------------------+
| Delta Table (Raw)            |
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
| retail_stockout_features     |
+--------------+---------------+
               |
               v
+------------------------------+
| Training (XGBoost + MLflow)  |
| - log params                 |
| - log metrics                |
| - log artifacts              |
| - save model                 |
+--------------+---------------+
               |
               v
+------------------------------+
| Batch Inference              |
| - stockout_risk_score        |
| - stockout_risk_label        |
+--------------+---------------+
               |
               v
+------------------------------+
| Delta Table (Predictions)    |
| retail_stockout_predictions  |
+------------------------------+

Real-time Serving:
FastAPI (/predict) -> loads model.pkl -> returns risk score + label

Deployment:
Docker -> Kubernetes -> AKS (public LoadBalancer service)
```

---

## 🧰 Tech Stack

### Data / ML

* Databricks Community Edition
* Apache Spark (PySpark)
* Delta Tables
* XGBoost
* MLflow (Tracking + Model artifacts)

### Serving / Infra

* FastAPI
* Uvicorn
* Docker
* Kubernetes (manifests)
* Azure Kubernetes Service (AKS)

### DevOps / CI/CD

* GitHub Actions
* Docker Hub container registry

---

## ✅ Databricks Delta Tables

| Table                                           | Description                 |
| ----------------------------------------------- | --------------------------- |
| `workspace.default.retail_stockout_data`        | Raw retail dataset          |
| `workspace.default.retail_stockout_features`    | Engineered features + label |
| `workspace.default.retail_stockout_predictions` | Batch inference results     |

---

## 📊 Model Results (XGBoost)

Model trained on Spark-engineered features and evaluated on a held-out test split.

**Performance:**

* **AUC:** `0.9930`
* **F1:** `0.9630`
* **Precision:** `0.9523`
* **Recall:** `0.9739`

Confusion Matrix:

```
[[15114  1158]
 [  620 23108]]
```

---

# ✅ Batch Pipeline (Databricks)

## Notebook Flow

1. **01_data_generation**

   * Creates raw retail data
   * Saves to Delta: `retail_stockout_data`

2. **02_feature_engineering**

   * Builds lag + rolling + inventory pressure features
   * Generates next-7-day stockout label
   * Saves: `retail_stockout_features`

3. **03_model_training**

   * Trains XGBoost model
   * Logs metrics + artifacts to MLflow

4. **04_batch_inference**

   * Scores last 30 days of data
   * Saves: `retail_stockout_predictions`

---

# ✅ Real-time Serving (FastAPI)

## 📁 Folder Structure

```
stockout-serving/
├── app.py
├── schema.py
├── requirements.txt
├── stockout_model.pkl
├── Dockerfile
└── k8s/
    ├── deployment.yaml
    └── service.yaml
```

---

## Run Locally

```bash
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000
```

Check:

* Health: `http://127.0.0.1:8000/health`
* Docs: `http://127.0.0.1:8000/docs`

---

## Sample Request Payload (Swagger)

### ✅ High stockout risk example

```json
{
  "price": 199.0,
  "promo": 1,
  "daily_sales": 14,
  "inventory": 4,
  "sales_lag_1": 16,
  "sales_lag_3": 13,
  "sales_lag_7": 11,
  "inv_lag_1": 6,
  "promo_lag_1": 1,
  "sales_roll7_avg": 15.2,
  "sales_roll7_sum": 106.0,
  "sales_roll14_avg": 13.1,
  "promo_roll7_sum": 3,
  "inv_to_sales_ratio": 0.25,
  "low_inventory_flag": 1
}
```

Expected output:

* `stockout_risk_score` → high (0.85 to 0.99)
* `stockout_risk_label` → 1

---

# 🐳 Docker

## Build Image

```bash
docker build -t stockout-api:1.0 .
```

## Run Container

```bash
docker run -p 8000:8000 stockout-api:1.0
```

---

# ☸️ Kubernetes Deployment (YAML Manifests)

Located in:

```
stockout-serving/k8s/
```

Apply:

```bash
kubectl apply -f stockout-serving/k8s/deployment.yaml
kubectl apply -f stockout-serving/k8s/service.yaml
```

Check:

```bash
kubectl get pods
kubectl get svc
```

---

# ✅ Azure Kubernetes Service (AKS) Deployment

The application is deployed on AKS using:

* Kubernetes `Deployment` with 2 replicas
* Kubernetes `Service` type `LoadBalancer` (public IP)

Public access:

* `http://<EXTERNAL-IP>/health`
* `http://<EXTERNAL-IP>/docs`

---

# ✅ CI/CD (GitHub Actions)

## Workflows included:

✅ CI checks (dependency install + import test)
✅ Auto-deploy to AKS (on push to `main`)

* Builds Docker image
* Pushes to DockerHub
* Applies Kubernetes YAMLs to AKS

---

## 📌 Key Highlights

* ✅ Full batch ML pipeline on Databricks + Delta tables
* ✅ Time-series feature engineering (lags + rolling windows)
* ✅ MLflow-based experiment tracking
* ✅ Batch inference pipeline writing predictions to Delta
* ✅ Real-time inference API (FastAPI)
* ✅ Docker + Kubernetes ready
* ✅ AKS deployment with public service endpoint
* ✅ CI/CD automation via GitHub Actions

---

# 👤 Author

**Ramdas Coundinya VK**
Data Engineering | MLOps | Kubernetes | Databricks | FastAPI | Docker | Azure AKS
