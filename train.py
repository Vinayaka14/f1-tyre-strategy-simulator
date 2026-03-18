import traceback
from src.ml_model import PitStopClassifier

def main():
    try:
        classifier = PitStopClassifier()
        
        # Load and train
        X, y = classifier.load_data()
        X_test, y_test = classifier.train(X, y)
        
        # Evaluate
        roc_auc = classifier.evaluate(X_test, y_test)
        
        optimal_threshold = classifier.find_optimal_threshold(X_test, y_test)
        
        import json
        import os
        os.makedirs('models', exist_ok=True)
        with open('models/threshold.json', 'w') as f:
            json.dump({'optimal_threshold': float(optimal_threshold)}, f)
        print(f"Threshold saved: {optimal_threshold:.4f}")
        
        # Save
        classifier.save()
        
        import shutil
        shutil.copy('models/pit_classifier.pkl', 'models/best_classifier.pkl')
        print("Copied pit_classifier.pkl → best_classifier.pkl")
        
        # Summary
        print(f"\n--- FINAL SUMMARY ---")
        print(f"Model trained successfully. ROC-AUC: {roc_auc:.4f}")
        print("Model saved to models/pit_classifier.pkl")
        print("Copied to models/best_classifier.pkl")
        print("Feature importance chart saved to data/processed/feature_importance.png")
        
    except Exception as e:
        print(f"An error occurred during training: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
