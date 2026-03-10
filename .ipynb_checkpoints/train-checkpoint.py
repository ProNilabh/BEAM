import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

mlflow.set_tracking_uri("file:mlruns") 
mlflow.set_experiment("BEAM")

def run_training():
    df = pd.read_csv(r"C:\Users\nilab\Documents\artin\ENB2012_data.csv").dropna()
    X = df.drop(columns=['Y1', 'Y2']).values.astype('float32')
    y = df[['Y1', 'Y2']].values.astype('float32')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    configs = [
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
        {"n_estimators": 150, "max_depth": 5, "learning_rate": 0.05},
        {"n_estimators": 200, "max_depth": 7, "learning_rate": 0.01}
    ]

    best_r2 = -float('inf')
    best_run_id = None

    for i, params in enumerate(configs):
        with mlflow.start_run(run_name=f"Pipeline_XGB_Variant_{i}") as run:
            model = xgb.XGBRegressor(**params, random_state=42)
            model.fit(X_train, y_train)
            
            predictions = model.predict(X_test)
            r2 = r2_score(y_test, predictions)
            
            mlflow.log_params(params)
            mlflow.log_metric("r2_score", r2)
            mlflow.xgboost.log_model(model, name="model")
            
            print(f"Variant {i} complete. R2 Score: {r2:.4f}")

            if r2 > best_r2:
                best_r2 = r2
                best_run_id = run.info.run_id

    print("\n" + "="*30)
    print(f"best model found")
    print(f"Run ID: {best_run_id}")
    print(f"Best R2 Score: {best_r2:.4f}")
    print("="*30)
    
    return best_run_id

if __name__ == "__main__":
    run_training()