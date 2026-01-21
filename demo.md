# ✅ Stockout Prediction System — Demo Guide
This demo shows an end-to-end MLOps project:
- Batch feature engineering + training (Databricks)
- Real-time inference service (FastAPI)
- Docker + Kubernetes deployment (AKS)
- CI/CD pipeline using GitHub Actions

---

## 1) Live API Demo (Swagger UI)

### ✅ Health Check
**GET**
```

/health

````

Expected output:
```json
{"status":"ok"}
````

### ✅ Swagger UI

Open:

```
/docs
```

---

## 2) Real-Time Prediction Demo

### ✅ High Stockout Risk Example (expected label = 1)

**POST**

```
/predict
```

Request JSON:

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

Expected response (example):

```json
{
  "stockout_risk_score": 0.90,
  "stockout_risk_label": 1
}
```

---

### ✅ Low Stockout Risk Example (expected label = 0)

Request JSON:

```json
{
  "price": 249.0,
  "promo": 0,
  "daily_sales": 6,
  "inventory": 120,
  "sales_lag_1": 5,
  "sales_lag_3": 6,
  "sales_lag_7": 4,
  "inv_lag_1": 125,
  "promo_lag_1": 0,
  "sales_roll7_avg": 5.4,
  "sales_roll7_sum": 38.0,
  "sales_roll14_avg": 5.1,
  "promo_roll7_sum": 0,
  "inv_to_sales_ratio": 18.5,
  "low_inventory_flag": 0
}
```

Expected response (example):

```json
{
  "stockout_risk_score": 0.05,
  "stockout_risk_label": 0
}
```

---

## 3) Run Locally (FastAPI)

From `stockout-serving/`:

Install dependencies:

```bash
pip install -r requirements.txt
```

Start server:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open:

* [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
* [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 4) Docker Demo

Build image:

```bash
docker build -t stockout-api:1.0 .
```

Run container:

```bash
docker run -p 8000:8000 stockout-api:1.0
```

Test:

```bash
curl http://127.0.0.1:8000/health
```

Expected:

```json
{"status":"ok"}
```

---

## 5) Kubernetes Demo (AKS / Minikube)

Kubernetes manifests:

```
stockout-serving/k8s/
  deployment.yaml
  service.yaml
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

## 6) CI/CD Demo (GitHub Actions)

On every push to `main`, GitHub Actions automatically:

* builds Docker image
* pushes image to Docker Hub
* deploys updates to AKS using kubectl

Check status:
GitHub → Actions tab ✅

---

## 7) Batch Pipeline Demo (Databricks)

### Tables Created

* `workspace.default.retail_stockout_data`
* `workspace.default.retail_stockout_features`
* `workspace.default.retail_stockout_predictions`

### Feature Engineering Logic

Spark computes:

* lag features (1, 3, 7 day lags)
* rolling averages/sums (7/14 day)
* inventory pressure ratios
* label = stockout in next 7 days