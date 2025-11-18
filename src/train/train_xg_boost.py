from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np

def run(X_train, y_train, X_val, y_val):
    print("Starting XGBoost hyperparameter tuning with GridSearchCV...")
    
    xgb = XGBRegressor(random_state=42, verbosity=0)
    
    # hyperparameter tuning grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # GridSearchCV with 5-fold cross-validation
    grid_search = GridSearchCV(
        xgb,
        param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1,
        scoring='neg_mean_squared_error'
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score (neg_mse): {grid_search.best_score_:.4f}")
    
    best_model = grid_search.best_estimator_
    preds = best_model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    print(f"XGBoost RMSE (validation): {rmse:.4f}")
    
    return best_model