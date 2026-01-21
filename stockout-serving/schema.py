from pydantic import BaseModel

class PredictRequest(BaseModel):
    price: float
    promo: int
    daily_sales: int
    inventory: int

    sales_lag_1: float
    sales_lag_3: float
    sales_lag_7: float

    inv_lag_1: float
    promo_lag_1: float

    sales_roll7_avg: float
    sales_roll7_sum: float
    sales_roll14_avg: float

    promo_roll7_sum: float
    inv_to_sales_ratio: float
    low_inventory_flag: int


class PredictResponse(BaseModel):
    stockout_risk_score: float
    stockout_risk_label: int
