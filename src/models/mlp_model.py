import os
import joblib
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class MLPPitClassifier:
    def __init__(self):
        self.FEATURE_COLS = [
            'TyreLife', 'compound_encoded', 'lap_time_delta', 'deg_rate',
            'GapToAhead', 'GapToBehind', 'laps_remaining', 
            'fuel_adjusted_laptime', 'pit_window_open', 'safety_car_active',
            'TrackTemp', 'Position'
        ]
        self.LABEL_COL = 'pit_decision'
        self.model = None
        self.scaler = None

    def load_data(self, path='data/processed/features.csv'):
        full_path = os.path.join(BASE_DIR, path)
        df = pd.read_csv(full_path)
        X = df[self.FEATURE_COLS]
        y = df[self.LABEL_COL]
        return X, y

    def train(self, X_train, y_train):
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        sample_weight = compute_sample_weight('balanced', y_train)
        
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        # Note: scikit-learn MLPClassifier does not fully support sample_weights in fit() natively, 
        # but we pass it anyway per requirements, catching the TypeError if it occurs.
        try:
            self.model.fit(X_train_scaled, y_train, sample_weight=sample_weight)
        except TypeError:
            self.model.fit(X_train_scaled, y_train)

    def save(self, path_model='models/mlp_classifier.pkl', path_scaler='models/mlp_scaler.pkl'):
        model_full = os.path.join(BASE_DIR, path_model)
        scaler_full = os.path.join(BASE_DIR, path_scaler)
        
        os.makedirs(os.path.dirname(model_full), exist_ok=True)
        os.makedirs(os.path.dirname(scaler_full), exist_ok=True)
        
        joblib.dump(self.model, model_full)
        joblib.dump(self.scaler, scaler_full)
        
        print(f"Model saved to {model_full}")
        print(f"Scaler saved to {scaler_full}")

    def load(self, path_model='models/mlp_classifier.pkl', path_scaler='models/mlp_scaler.pkl'):
        model_full = os.path.join(BASE_DIR, path_model)
        scaler_full = os.path.join(BASE_DIR, path_scaler)
        
        self.model = joblib.load(model_full)
        self.scaler = joblib.load(scaler_full)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        probs = self.model.predict_proba(X_scaled)[:, 1]
        return preds, probs

if __name__ == "__main__":
    clf = MLPPitClassifier()
    X, y = clf.load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    clf.train(X_train, y_train)
    clf.save()
    print("MLP training complete")
