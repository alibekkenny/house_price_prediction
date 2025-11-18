from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

def run(X_train, y_train, X_val, y_val):
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.8,
        eval_metric="rmse",
        random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, preds))
    print(f"XGBoost RMSE: {rmse}")

    return model
