import os
import logging
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'features.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# Global dictionaries that will be loaded/saved
DEGRADATION_CURVES = {}
MAX_TYRE_LIFE = {}

# Auto-load curves from pkl on import
def _load_curves():
    pkl_path = os.path.join(MODEL_DIR, 'tyre_curves.pkl')
    try:
        if os.path.exists(pkl_path):
            data = joblib.load(pkl_path)
            DEGRADATION_CURVES.update(
                data.get('DEGRADATION_CURVES', {}))
            MAX_TYRE_LIFE.update(
                data.get('MAX_TYRE_LIFE', {}))
            logger.info(
                f"Tyre curves loaded: "
                f"{list(DEGRADATION_CURVES.keys())}")
        else:
            logger.warning(
                f"tyre_curves.pkl not found at {pkl_path}")
    except Exception as e:
        logger.warning(f"Could not load tyre curves: {e}")

_load_curves()

def predict_lap_delta(compound_encoded, tyre_age):
    if compound_encoded not in DEGRADATION_CURVES:
        return 0.0
    
    coeffs = DEGRADATION_CURVES[compound_encoded]
    delta = np.polyval(coeffs, tyre_age)
    return max(0.0, float(delta))

def tyre_life_remaining_pct(compound_encoded, tyre_age):
    if compound_encoded not in MAX_TYRE_LIFE:
        # Fallback dictionary if compound is totally unknown
        fallback = {2: 35, 1: 40, 0: 55}
        max_life = fallback.get(compound_encoded, 30)
    else:
        max_life = MAX_TYRE_LIFE[compound_encoded]
        
    pct = 100.0 * (1.0 - tyre_age / max_life)
    return max(0.0, min(100.0, float(pct)))

def main():
    try:
        # 1. LOAD DATA
        if not os.path.exists(FEATURES_FILE):
            logger.error(f"Features file not found at {FEATURES_FILE}")
            return
            
        df = pd.read_csv(FEATURES_FILE)
        initial_len = len(df)
        
        # Filter out SC laps, anomalous lap time deltas, and first lap of stint
        df = df[df['safety_car_active'] == 0]
        df = df[df['lap_time_delta'] <= 10]
        df = df[df['TyreLife'] >= 2]
        
        logger.info(f"Filtered data: {initial_len} -> {len(df)} rows")

        # 2. FIT DEGRADATION CURVES
        # 0=HARD, 1=MEDIUM, 2=SOFT
        compounds_seen = df['compound_encoded'].unique()
        
        for c in [0, 1, 2]:
            if c not in compounds_seen:
                continue
                
            c_data = df[df['compound_encoded'] == c]
            
            if len(c_data) < 10:
                logger.warning(f"Not enough data to fit compound {c}")
                continue
                
            x = c_data['TyreLife'].values
            y = c_data['lap_time_delta'].values
            
            # Fit degree-2 polynomial
            coeffs = np.polyfit(x, y, 2)
            DEGRADATION_CURVES[c] = coeffs
            
            # Store max tyre life with sanity bounds
            observed_max = c_data['TyreLife'].max()
            fallback = {2: 35, 1: 40, 0: 55}
            
            if pd.isna(observed_max) or observed_max < 20 or observed_max > 80:
                MAX_TYRE_LIFE[c] = fallback.get(c, 30)
            else:
                MAX_TYRE_LIFE[c] = float(observed_max)
                
            print(f"Compound {c} (0=H,1=M,2=S) - Fitted Coeffs: {coeffs}")
            print(f"Compound {c} - Max Tyre Life: {MAX_TYRE_LIFE[c]}")

        # 5. SAVE MODEL
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, 'tyre_curves.pkl')
        model_data = {
            'DEGRADATION_CURVES': DEGRADATION_CURVES,
            'MAX_TYRE_LIFE': MAX_TYRE_LIFE
        }
        joblib.dump(model_data, model_path)
        print(f"\nTyre degradation model saved to {model_path}")

        # 6. PLOT CURVES
        plt.figure(figsize=(10, 6))
        x_plot = np.arange(1, 51)
        
        colors = {0: 'green', 1: 'yellow', 2: 'red'}
        labels = {0: 'HARD', 1: 'MEDIUM', 2: 'SOFT'}
        
        for c in [0, 1, 2]:
            if c in DEGRADATION_CURVES:
                y_plot = [predict_lap_delta(c, age) for age in x_plot]
                plt.plot(x_plot, y_plot, color=colors[c], label=labels[c], linewidth=2)
                
        plt.title('Tyre Degradation Curves by Compound')
        plt.xlabel('Tyre Life (Laps)')
        plt.ylabel('Lap Time Delta (Seconds)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        plot_path = os.path.join(PROCESSED_DIR, 'tyre_curves.png')
        plt.savefig(plot_path, facecolor='white', transparent=False)
        print(f"Curve plot saved to {plot_path}")

    except Exception as e:
        logger.error(f"Error in tyre modeling: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
