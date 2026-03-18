import os
import traceback
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

from src.models.xgb_model import XGBPitClassifier
from src.models.lgbm_model import LGBMPitClassifier
from src.models.mlp_model import MLPPitClassifier
from src.model_evaluator import evaluate_model, save_comparison_chart

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    try:
        print("Loading features...")
        features_path = os.path.join(BASE_DIR, 'data', 'processed', 'features.csv')
        df = pd.read_csv(features_path)
        
        # Instantiate any model to get cols
        xgb_clf = XGBPitClassifier()
        X = df[xgb_clf.FEATURE_COLS]
        y = df[xgb_clf.LABEL_COL]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        counts = y_train.value_counts()
        scale_pos_weight = counts[0] / counts[1]
        print(f"scale_pos_weight computed: {scale_pos_weight:.2f}")

        results = []

        # XGBoost
        print("\n--- Training XGBoost ---")
        xgb_clf.train(X_train, y_train, scale_pos_weight)
        xgb_clf.save('models/xgb_classifier.pkl')
        y_pred, y_proba = xgb_clf.predict(X_test)
        results.append(evaluate_model('XGBoost', y_test, y_pred, y_proba))

        # LightGBM
        print("\n--- Training LightGBM ---")
        lgbm_clf = LGBMPitClassifier()
        lgbm_clf.train(X_train, y_train, scale_pos_weight)
        lgbm_clf.save('models/lgbm_classifier.pkl')
        y_pred, y_proba = lgbm_clf.predict(X_test)
        results.append(evaluate_model('LightGBM', y_test, y_pred, y_proba))

        # MLP
        print("\n--- Training MLP ---")
        mlp_clf = MLPPitClassifier()
        mlp_clf.train(X_train, y_train)
        mlp_clf.save('models/mlp_classifier.pkl', 'models/mlp_scaler.pkl')
        y_pred, y_proba = mlp_clf.predict(X_test)
        results.append(evaluate_model('MLP', y_test, y_pred, y_proba))

        # Table
        print("\nModel Comparison Table:")
        print(f"{'Model':<15}{'Accuracy':<12}{'Precision':<12}{'Recall':<10}{'F1':<8}{'ROC-AUC':<8}")
        for r in results:
            print(f"{r['model']:<15}{r['accuracy']:<12.3f}{r['precision']:<12.3f}{r['recall']:<10.3f}{r['f1']:<8.3f}{r['roc_auc']:<8.3f}")

        # Chart
        chart_path = os.path.join(BASE_DIR, 'data', 'processed', 'model_comparison.png')
        save_comparison_chart(results, chart_path)

        # Winner
        best_model_name = ""
        best_f1 = -1
        for r in results:
            if r['f1'] > best_f1:
                best_f1 = r['f1']
                best_model_name = r['model']
                
        print(f"\nBest model: {best_model_name} — F1={best_f1:.3f}.")
        
        best_cls_dest = os.path.join(BASE_DIR, 'models', 'best_classifier.pkl')
        if best_model_name == 'XGBoost':
            shutil.copy(os.path.join(BASE_DIR, 'models', 'xgb_classifier.pkl'), best_cls_dest)
        elif best_model_name == 'LightGBM':
            shutil.copy(os.path.join(BASE_DIR, 'models', 'lgbm_classifier.pkl'), best_cls_dest)
        elif best_model_name == 'MLP':
            shutil.copy(os.path.join(BASE_DIR, 'models', 'mlp_classifier.pkl'), best_cls_dest)
            shutil.copy(os.path.join(BASE_DIR, 'models', 'mlp_scaler.pkl'), os.path.join(BASE_DIR, 'models', 'best_scaler.pkl'))
        
        print(f"Saved to {best_cls_dest}")

    except Exception as e:
        print(f"Error during model comparison: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
