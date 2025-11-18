from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

def run(X_train, y_train, X_val, y_val):
    model = DummyRegressor(strategy='mean')
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    print(f"GradientBoost RMSE: {rmse}")
    
    return model
