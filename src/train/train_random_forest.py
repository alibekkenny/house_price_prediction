from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np

def run(X_train, y_train, X_val, y_val):
    print("Starting RandomForest hyperparameter tuning with GridSearchCV...")
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # hyperparameter grid
    grid_search = GridSearchCV(
        rf,
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
    print(f"RandomForest RMSE (validation): {rmse:.4f}")
    
    return best_model
