import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def train_rf():
    
    df = pd.read_csv('data/processed/train_cleaned.csv')
    
    
    X = df.drop(columns=['RUL', 'unit_nr'])
    y = df['RUL']
    
    print(" Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(rf, 'models/rf_model.pkl')
    print("Step 2 Complete: Random Forest trained and saved in 'models/'")

if __name__ == "__main__":
    train_rf()