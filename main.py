import os
from src.data_loader import load_data, brief_info
from src.feature_engineering import add_features, select_k_best_features, select_rfe_features
from src.train import train_baseline, train_linear_regression, train_random_forest, train_xg_boost, train_gradient_boost
import src.preprocessing as prep
import json
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
    
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

DATA_DIR = "data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")

def menu():
    print("""
    1 - Load raw data
    2 - Show brief info of loaded data
    3 - Prepare data
    4 - Train models
    5 - Generate test predictions
    6 - Evaluate predictions
    0 - Exit
    """)

def run():
    df = None
    state = 0
    saved_models = {}
    scaler = None
    selector = None
    feature_columns = None
    
    # Load preprocessing objects if they exist
    prep_file = os.path.join(MODEL_DIR, "preprocessing_objects.pkl")
    if os.path.exists(prep_file):
        prep_objects = joblib.load(prep_file)
        scaler = prep_objects.get('scaler')
        selector = prep_objects.get('selector')
        feature_columns = prep_objects.get('feature_columns')
        print("Loaded preprocessing objects")
    
    if os.path.exists(MODEL_DIR):
        print("Checking for existing trained models...")
        for model_file in os.listdir(MODEL_DIR):
            if model_file.endswith(".pkl") and model_file != "preprocessing_objects.pkl":
                model_name = model_file.replace(".pkl", "")
                model_path = os.path.join(MODEL_DIR, model_file)
                try:
                    saved_models[model_name] = joblib.load(model_path)
                    print(f"Loaded model: {model_name} from {model_path}")
                except Exception as e:
                    print(f"Failed to load {model_name}: {e}")

        if saved_models:
            print("Existing models loaded.")
            state = 4

    while True:
        menu()
        ch = input("Choose: ").strip()
        try:
            if ch == '1':
                if state != 0:
                    print("Data is already loaded.")
                    continue

                if df is not None:
                    print("Data is loaded.")
                else:
                    df = load_data(os.path.join(RAW_DATA_DIR, 'train.csv'))
                    state = 1
                    print("Data loaded.")

            elif ch == '2':
                if df is None:
                    print("Load data first (option 1).")
                    continue
                brief_info(df)

            elif ch == '3':
                if state >= 3:
                    print("Data is already prepared.")
                    continue
                if state == 0:
                    print("Load data first (option 1).")
                    continue

                df = prep.clean_data(df)
                df = add_features(df)
                df = prep.encoding_categorical(df, df.select_dtypes(include=["object"]).columns)
                
                X, y = prep.split_features_target(df, "SalePrice")
                X_train, X_val, y_train, y_val = prep.train_val_split(X, y)
                X_train_scaled, X_val_scaled, scaler = prep.scale_features(X_train, X_val)

                # saving column names before feature selection
                feature_columns = X_train.columns

                X_train_sel, X_val_sel, selector = select_k_best_features(X_train_scaled, y_train, X_val_scaled, k=20)
                # X_train_rfe, X_val_rfe, selector = select_rfe_features(X_train_scaled, y_train, X_val_scaled, n_features=20)
                
                # Save preprocessing objects for later use
                os.makedirs(MODEL_DIR, exist_ok=True)
                prep_objects = {
                    'scaler': scaler,
                    'selector': selector,
                    'feature_columns': feature_columns
                }
                prep_file = os.path.join(MODEL_DIR, "preprocessing_objects.pkl")
                joblib.dump(prep_objects, prep_file)
                print(f"Preprocessing objects saved to {prep_file}")
                
                state = 3
                print("Data was cleaned, features added, encoded, scaled, and selected.")

            elif ch == '4':
                if state >= 4:
                    print("Models are already trained.")
                    continue
                elif state != 3:
                    print("Prepare data first (option 3).")
                    continue
                
                print("Training Dummy Regressor...")
                baseline_model = train_baseline.run(X_train_sel, y_train, X_val_sel, y_val)

                print("Training Linear Regression...")
                lr_model = train_linear_regression.run(X_train_sel, y_train, X_val_sel, y_val)

                print("Training Random Forest...")
                rf_model = train_random_forest.run(X_train_sel, y_train, X_val_sel, y_val)

                print("Training XGBoost...")
                xg_model = train_xg_boost.run(X_train_sel, y_train, X_val_sel, y_val)

                print("Training GradientBoost...")
                gradient_boost_model = train_gradient_boost.run(X_train_sel, y_train, X_val_sel, y_val)

                saved_models = {
                    "DummyRegressor": baseline_model,
                    "LinearRegression": lr_model,
                    "RandomForest": rf_model,
                    "XGBoost": xg_model,
                    "GradientBoost": gradient_boost_model
                }

                os.makedirs("models", exist_ok=True)
                for name, model in saved_models.items():
                    path = os.path.join("models", f"{name}.pkl")
                    joblib.dump(model, path)
                    print(f"Saved {name}, path: {path}")

                state = 4
                print("All models trained successfully.")

            elif ch == '5':
                if state >= 5:
                    print("Models are already evaluated.")
                    continue
                elif state != 4:
                    print("Train models first (option 4).")
                    continue
                
                print("Evaluating models...")
                state = 5
                print("Loading test data...")
                df_test = load_data(os.path.join(RAW_DATA_DIR, 'test.csv'))

                print("Preparing test data...")
                df_test = prep.clean_data(df_test)
                df_test = add_features(df_test)
                df_test = prep.encoding_categorical(df_test, df_test.select_dtypes(include=["object"]).columns)

                df_test = df_test.reindex(columns=feature_columns, fill_value=0)

                X_test_scaled = scaler.transform(df_test)
                X_test_sel = selector.transform(X_test_scaled)

                print("Generating predictions...")
                predictions = {}
                for name, model in saved_models.items():
                    preds = model.predict(X_test_sel)
                    predictions[name] = preds.tolist()

                os.makedirs("outputs", exist_ok=True)
                with open("outputs/predictions.json", "w") as f:
                    json.dump(predictions, f, indent=4)

                print("Predictions saved to outputs/predictions.json")
                state = 5

            elif ch == '6':
                try:
                    plots_dir = "outputs/plots"
                    os.makedirs(plots_dir, exist_ok=True)

                    # Load predictions
                    preds_file = "outputs/predictions.json"
                    if not os.path.exists(preds_file):
                        print("No predictions found. Run option 5 first.")
                        continue

                    with open(preds_file, "r") as f:
                        preds = json.load(f)

                    metrics = {}
                    print("Generating regression plots for predictions...")

                    for model_name, pred_list in preds.items():
                        y_pred = np.array(pred_list)

                        # regression plot (y_pred vs index)
                        plt.figure(figsize=(10, 6))
                        plt.plot(y_pred, marker='o', linestyle='', alpha=0.3, label='Predictions')
                        plt.xlabel("Sample Index")
                        plt.ylabel("Predicted SalePrice")
                        plt.title(f"Predictions: {model_name}")
                        plt.grid(True)
                        plt.savefig(os.path.join(plots_dir, f"{model_name}.png"))
                        plt.close()

                        # Since no ground truth, metrics are placeholders
                        metrics[model_name] = {
                            "rmse": "N/A",
                            "mae": "N/A",
                            "r2": "N/A"
                        }

                    with open("outputs/metrics.json", "w") as f:
                        json.dump(metrics, f, indent=4)

                    print("Prediction plots saved to outputs/plots/")
                    print("Metrics saved (placeholders) to outputs/metrics.json")

                    print("\nModel Performance Summary:")
                    for model_name, m in metrics.items():
                        print(f"{model_name}: RMSE={m['rmse']}, MAE={m['mae']}, R2={m['r2']}")

                except Exception as e:
                    print("Error during plotting:", e)



            elif ch == '0':
                break
            else:
                print("Invalid option. Try again.")
        except Exception as e:
            print("Error:", e)
    

if __name__ == "__main__":
    run()
