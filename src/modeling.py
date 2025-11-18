import pickle
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import numpy as np

def baseline_model():
    return DummyRegressor(strategy='median')

def get_models():
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42, verbosity=0),
        "SVR": SVR()
    }

def train_with_grid(pipeline, param_grid, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error'):
    gs = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)
    return gs

def save_model(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print("Model saved to", path)

def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
