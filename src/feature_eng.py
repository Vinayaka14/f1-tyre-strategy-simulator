import os
import glob
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

def compute_slope(y):
    # Polyfit needs an array without NaNs. We require 3 elements.
    y_clean = y[~np.isnan(y)]
    if len(y_clean) < 3:
        return 0.0
    # x is just 0, 1, 2...
    x = np.arange(len(y_clean))
    return np.polyfit(x, y_clean, 1)[0]

def main():
    try:
        # 1. LOAD & COMBINE
        csv_files = [
            os.path.join(RAW_DIR, 'laps_2022.csv'),
            os.path.join(RAW_DIR, 'laps_2023.csv'),
            os.path.join(RAW_DIR, 'laps_2024.csv')
        ]
        
        df_list = []
        for f in csv_files:
            if os.path.exists(f):
                df_list.append(pd.read_csv(f))
            else:
                logger.warning(f"File missing: {f}")
                
        if not df_list:
            logger.error("No raw lap CSVs found to process.")
            return

        df = pd.concat(df_list, ignore_index=True)
        print(f"1. LOAD & COMBINE: Total rows loaded: {len(df)}")

        # 2. ENCODE COMPOUND
        # SOFT=2, MEDIUM=1, HARD=0, INTERMEDIATE=3, WET=4, UNKNOWN=-1
        compound_map = {
            'SOFT': 2, 'MEDIUM': 1, 'HARD': 0, 
            'INTERMEDIATE': 3, 'WET': 4, 'UNKNOWN': -1
        }
        df['compound_encoded'] = df['Compound'].map(compound_map).fillna(-1).astype(int)

        # 3. LAP TIME DELTA
        # Per driver, per race
        group_cols = ['Year', 'EventName', 'DriverNumber']
        df = df.sort_values(by=group_cols + ['LapNumber']).reset_index(drop=True)
        
        # expanding().min() to get the best lap *so far* for that driver in that race
        min_so_far = df.groupby(group_cols)['LapTime_s'].transform(lambda x: x.expanding().min())
        df['lap_time_delta'] = df['LapTime_s'] - min_so_far

        # 4. DEGRADATION RATE
        # Slope of LapTime_s over last 3 laps
        df['deg_rate'] = df.groupby(group_cols)['LapTime_s'].transform(
            lambda x: x.rolling(window=3, min_periods=3).apply(compute_slope, raw=True)
        )
        df['deg_rate'] = df['deg_rate'].fillna(0.0)

        # 5. FUEL ADJUSTED LAP TIME
        total_laps = df.groupby(['Year', 'EventName'])['LapNumber'].transform('max')
        df['laps_remaining'] = total_laps - df['LapNumber']
        df['fuel_adjusted_laptime'] = df['LapTime_s'] - (df['laps_remaining'] * 0.03)

        # 6. PIT WINDOW OPEN FLAG
        min_life = {
            'SOFT': 10, 'MEDIUM': 15, 'HARD': 20, 
            'INTERMEDIATE': 5, 'WET': 5, 'UNKNOWN': 10
        }
        df['compound_min_life'] = df['Compound'].map(min_life).fillna(10)
        df['pit_window_open'] = (df['TyreLife'] > df['compound_min_life']).astype(int)

        # 7. SAFETY CAR PROXY
        session_fastest = df.groupby(['Year', 'EventName'])['LapTime_s'].transform('min')
        df['safety_car_active'] = (df['LapTime_s'] > session_fastest * 1.10).astype(int)

        # 8. FINAL FEATURE SELECTION
        keep_columns = [
            'TyreLife', 'compound_encoded', 'lap_time_delta', 'deg_rate', 
            'GapToAhead', 'GapToBehind', 'laps_remaining', 'fuel_adjusted_laptime', 
            'pit_window_open', 'safety_car_active', 'TrackTemp', 'Position', 'PitStop'
        ]
        
        df_out = df[keep_columns].copy()
        df_out.rename(columns={'PitStop': 'pit_decision'}, inplace=True)
        
        # Drop NaNs
        df_out = df_out.dropna()
        
        # Ensure pit_decision is int and only 0 or 1
        df_out['pit_decision'] = df_out['pit_decision'].astype(int)
        df_out = df_out[df_out['pit_decision'].isin([0, 1])]

        # 9. SAVE & REPORT
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        out_path = os.path.join(PROCESSED_DIR, 'features.csv')
        df_out.to_csv(out_path, index=False)

        print(f"\n--- SUCCESS ---")
        print(f"Saved to: {out_path}")
        print(f"Total rows saved: {len(df_out)}")
        
        # Class distribution
        class_counts = df_out['pit_decision'].value_counts()
        total_valid = len(df_out)
        print("\nClass Distribution of pit_decision:")
        for cls_val in [0, 1]:
            count = class_counts.get(cls_val, 0)
            pct = (count / total_valid) * 100 if total_valid > 0 else 0
            print(f"  Class {cls_val}: {count} rows ({pct:.2f}%)")

        # Sanity check metrics
        print("\nSanity Checks:")
        for col in ['TyreLife', 'deg_rate']:
            c_min = df_out[col].min()
            c_max = df_out[col].max()
            c_mean = df_out[col].mean()
            print(f"  {col:<10} | Min: {c_min:8.3f} | Max: {c_max:8.3f} | Mean: {c_mean:8.3f}")

    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
