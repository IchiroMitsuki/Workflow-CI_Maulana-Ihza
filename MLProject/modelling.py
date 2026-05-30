import pandas as pd
import os
import mlflow
import dagshub
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Inisialisasi DagsHub (Gunakan variabel environment dari GitHub Secrets)
dagshub.init(repo_owner='IchiroMitsuki', repo_name='Eksperimen_SML_Maulana-Ihza-Ishlahy', mlflow=True)

def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'Japan_ImbalancePrice_preprocessing')
    
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).squeeze()
    return X_train, y_train

def main():
    X_train, y_train = load_data()
    
    mlflow.set_experiment("CI_Pipeline_Experiment")
    
    with mlflow.start_run():
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Logging ke DagsHub
        mlflow.sklearn.log_model(model, "model")
        
        # Menyimpan model secara lokal untuk dibuild oleh Docker di GitHub Actions
        mlflow.sklearn.save_model(model, "local_model")
        
        print("Training selesai dan model disimpan untuk Docker build.")

if __name__ == "__main__":
    main()