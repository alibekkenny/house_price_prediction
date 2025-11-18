import json
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# def save_metrics(model_name, y_true, y_pred, path="outputs/metrics.json"):
#     os.makedirs("outputs", exist_ok=True)

#     rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
#     mae = float(mean_absolute_error(y_true, y_pred))
#     r2 = float(r2_score(y_true, y_pred))

#     metrics_entry = {
#         "model": model_name,
#         "rmse": rmse,
#         "mae": mae,
#         "r2": r2
#     }

#     if os.path.exists(path):
#         with open(path, "r") as f:
#             data = json.load(f)
#     else:
#         data = []

#     data.append(metrics_entry)

#     with open(path, "w") as f:
#         json.dump(data, f, indent=4)

#     print(f"[OK] Saved metrics for {model_name}")

def evaluate_predictions(predictions_path, ground_truth_df):
    with open(predictions_path, "r") as f:
        preds = json.load(f)

    results = {}

    # Ground truth must be aligned by Id
    y_true = ground_truth_df["SalePrice"].values

    for model_name, pred_list in preds.items():
        y_pred = np.array(pred_list)

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)

        results[model_name] = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }

    return results