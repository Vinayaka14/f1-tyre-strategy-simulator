import os
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class LGBMPitClassifier:
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
        self.model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        self.model.fit(X_train, y_train)

    def save(self, path='models/lgbm_classifier.pkl'):
        full_path = os.path.join(BASE_DIR, path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        joblib.dump(self.model, full_path)
        print(f"Model saved to {full_path}")

    def load(self, path='models/lgbm_classifier.pkl'):
        full_path = os.path.join(BASE_DIR, path)
        self.model = joblib.load(full_path)

    def predict(self, X):
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)[:, 1]
        return preds, probs

if __name__ == "__main__":
    clf = LGBMPitClassifier()
    X, y = clf.load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    counts = y_train.value_counts()
    scale_pos_weight = counts[0] / counts[1]
    
    clf.train(X_train, y_train, scale_pos_weight)
    clf.save()
    print("LightGBM training complete")
