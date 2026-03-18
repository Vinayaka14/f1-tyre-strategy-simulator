import os
import logging
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class PitStopClassifier:
    def __init__(self):
        self.FEATURE_COLS = [
            'TyreLife', 'compound_encoded', 'lap_time_delta', 'deg_rate',
            'GapToAhead', 'GapToBehind', 'laps_remaining', 
            'fuel_adjusted_laptime', 'pit_window_open', 'safety_car_active',
            'TrackTemp', 'Position'
        ]
        self.LABEL_COL = 'pit_decision'
        self.model = None
        self.feature_importances_ = None

    def load_data(self, path='data/processed/features.csv'):
        full_path = os.path.join(BASE_DIR, path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Features file not found at {full_path}")
            
        df = pd.read_csv(full_path)
        X = df[self.FEATURE_COLS]
        y = df[self.LABEL_COL]
        
        print(f"Data loaded from {full_path}")
        print(f"Features X shape: {X.shape}")
        print("Target y class distribution:")
        print(y.value_counts())
        
        return X, y

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        counts = y_train.value_counts()
        scale_pos_weight = counts[0] / counts[1]
        print(f"\nComputed scale_pos_weight: {scale_pos_weight:.2f}")
        
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
        
        print("\nStarting XGBoost training...")
        self.model.fit(
            X_train, y_train, 
            eval_set=[(X_test, y_test)], 
            verbose=50
        )
        
        self.feature_importances_ = dict(zip(
            self.FEATURE_COLS,
            self.model.feature_importances_
        ))
        
        return X_test, y_test

    def evaluate(self, X_test, y_test):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
            
        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)[:, 1]
        
        print("\n--- MODEL EVALUATION ---")
        print("\nClassification Report:")
        print(classification_report(y_test, preds))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, preds))
        
        auc = roc_auc_score(y_test, probs)
        print(f"\nROC-AUC Score: {auc:.4f}")
        
        sorted_importances = sorted(self.feature_importances_.items(), key=lambda x: x[1], reverse=True)
        
        print("\nFeature Importances:")
        for feat, imp in sorted_importances:
            print(f"  {feat:25s}: {imp:.4f}")
            
        plt.figure(figsize=(10, 6))
        feats = [x[0] for x in sorted_importances][::-1]
        imps = [x[1] for x in sorted_importances][::-1]
        
        plt.barh(feats, imps, color='skyblue')
        plt.title("XGBoost Feature Importances")
        plt.xlabel("Relative Importance")
        plt.tight_layout()
        
        out_dir = os.path.join(BASE_DIR, 'data', 'processed')
        os.makedirs(out_dir, exist_ok=True)
        plot_path = os.path.join(out_dir, 'feature_importance.png')
        plt.savefig(plot_path, facecolor='white', transparent=False)
        print(f"\nSaved feature importance plot to {plot_path}")
        
        return auc

    def find_optimal_threshold(self, X_test, y_test):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
            
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        print(f"Probability range: min={y_proba.min():.4f} "
              f"max={y_proba.max():.4f} mean={y_proba.mean():.4f}")
        
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
        precisions = precisions[:-1]
        recalls = recalls[:-1]
        
        denom = precisions + recalls
        denom[denom == 0] = 1e-9
        f1_scores = 2 * (precisions * recalls) / denom
        
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        
        print(f"Optimal threshold: {best_threshold:.4f}")
        print(f"F1 at optimal threshold: {best_f1:.4f}")
        
        return best_threshold

    def save(self, path='models/pit_classifier.pkl'):
        if self.model is None:
            raise ValueError("No model to save.")
            
        full_path = os.path.join(BASE_DIR, path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        joblib.dump(self.model, full_path)
        print(f"Model saved to {full_path}")

    def load(self, path='models/pit_classifier.pkl'):
        full_path = os.path.join(BASE_DIR, path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Model file not found at {full_path}")
            
        self.model = joblib.load(full_path)
        print(f"Loaded model from {full_path}")

    def predict(self, feature_dict):
        if self.model is None:
            raise ValueError("Model is not loaded or trained.")
            
        row_data = {col: feature_dict.get(col, 0.0) for col in self.FEATURE_COLS}
        df_single = pd.DataFrame([row_data])
        
        pred = int(self.model.predict(df_single)[0])
        prob = float(self.model.predict_proba(df_single)[0, 1])
        
        return pred, prob
