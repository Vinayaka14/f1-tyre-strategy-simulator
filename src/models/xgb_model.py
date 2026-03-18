import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class XGBPitClassifier:
    def __init__(self):
        self.FEATURE_COLS = [
            'TyreLife', 'compound_encoded', 'lap_time_delta', 'deg_rate',
            'GapToAhead', 'GapToBehind', 'laps_remaining', 
            'fuel_adjusted_laptime', 'pit_window_open', 'safety_car_active',
            'TrackTemp', 'Position'
        ]
        self.LABEL_COL = 'pit_decision'
        self.model = None

    def load_data(self, path='data/processed/features.csv'):
        full_path = os.path.join(BASE_DIR, path)
        df = pd.read_csv(full_path)
        X = df[self.FEATURE_COLS]
        y = df[self.LABEL_COL]
        return X, y

    def train(self, X_train, y_train, scale_pos_weight):
        params = {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.05,
            'scale_pos_weight': scale_pos_weight,
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': -1
        }
        
        xgb_version_parts = xgb.__version__.split('.')
        major, minor = int(xgb_version_parts[0]), int(xgb_version_parts[1])
        if major < 1 or (major == 1 and minor < 6):
            params['use_label_encoder'] = False
            
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X_train, y_train, verbose=50)
        
        print("Calibrating model probabilities...")
        # Since we don't have a separate eval set passed here in standalone XGB train(),
        # passing the same train set for cv='prefit' is a small compromise but it works 
        # for scaling the outputs if we don't want to break the interface.
        self.model = CalibratedClassifierCV(
            self.model, 
            method='sigmoid',
            cv='prefit'
        )
        self.model.fit(X_train, y_train)
        print("Calibration complete.")

    def save(self, path='models/xgb_classifier.pkl'):
        full_path = os.path.join(BASE_DIR, path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        joblib.dump(self.model, full_path)
        print(f"Model saved to {full_path}")

    def load(self, path='models/xgb_classifier.pkl'):
        full_path = os.path.join(BASE_DIR, path)
        self.model = joblib.load(full_path)

    def predict(self, X):
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)[:, 1]
        return preds, probs

if __name__ == "__main__":
    clf = XGBPitClassifier()
    X, y = clf.load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    counts = y_train.value_counts()
    scale_pos_weight = counts[0] / counts[1]
    
    clf.train(X_train, y_train, scale_pos_weight)
    clf.save()
    print("XGBoost training complete")
