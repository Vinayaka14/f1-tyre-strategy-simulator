import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import joblib
import pandas as pd
import numpy as np
import logging
import json

from src.rule_engine import evaluate_rules, format_rule_output

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_classifier.pkl')
THRESHOLD_PATH = os.path.join(BASE_DIR, 'models', 'threshold.json')

FEATURE_COLS = [
    'TyreLife', 'compound_encoded', 'lap_time_delta', 'deg_rate',
    'GapToAhead', 'GapToBehind', 'laps_remaining',
    'fuel_adjusted_laptime', 'pit_window_open', 'safety_car_active',
    'TrackTemp', 'Position'
]

COMPOUND_ENCODING = {'SOFT': 2, 'MEDIUM': 1, 'HARD': 0, 'INTERMEDIATE': 3, 'WET': 4}

try:
    if os.path.exists(MODEL_PATH):
        MODEL = joblib.load(MODEL_PATH)
    else:
        logger.warning(f"Model not found at {MODEL_PATH}")
        MODEL = None
except Exception as e:
    logger.warning(f"Failed to load model: {e}")
    MODEL = None

try:
    with open(THRESHOLD_PATH) as f:
        THRESHOLD = json.load(f)['optimal_threshold']
    print(f"Loaded ML threshold: {THRESHOLD:.4f}")
except Exception as e:
    THRESHOLD = 0.03
    print(f"Using default threshold: {THRESHOLD}")

def build_feature_row(race_state):
    tyre_age = race_state.get('tyre_age', 0)
    compound_str = race_state.get('compound', 'UNKNOWN').upper().strip()
    compound_enc = COMPOUND_ENCODING.get(compound_str, -1)
    if compound_enc == -1:
        print(f"WARNING: Unknown compound '{compound_str}'")
    
    # Calculate pit window
    min_stint = {'SOFT': 10, 'MEDIUM': 15, 'HARD': 20, 'INTERMEDIATE': 5, 'WET': 5}
    compound_min = min_stint.get(compound_str, 10)
    pit_window_open = 1 if tyre_age >= compound_min else 0
    
    feature_dict = {
        'TyreLife': tyre_age,
        'compound_encoded': compound_enc,
        'lap_time_delta': race_state.get('lap_time_delta', 0.0),
        'deg_rate': race_state.get('deg_rate', 0.0),
        'GapToAhead': race_state.get('gap_ahead', 0.0),
        'GapToBehind': race_state.get('gap_behind', 0.0),
        'laps_remaining': race_state.get('laps_remaining', 0),
        'fuel_adjusted_laptime': race_state.get('fuel_adjusted_laptime', race_state.get('lap_time', 90.0)),
        'pit_window_open': pit_window_open,
        'safety_car_active': race_state.get('safety_car_active', 0),
        'TrackTemp': race_state.get('track_temp', 30.0),
        'Position': race_state.get('position', 1)
    }
    
    row = pd.DataFrame([feature_dict])[FEATURE_COLS]
    return row

def get_ml_signal(feature_row):
    if MODEL is None:
        return {
            'probability': 0.0,
            'prediction': 0,
            'note': 'Model not loaded'
        }
    probability = float(MODEL.predict_proba(feature_row)[0][1])
    prediction = 1 if probability >= THRESHOLD else 0

    if probability >= THRESHOLD * 3:
        note = "ML strongly recommends pit"
    elif probability >= THRESHOLD * 1.5:
        note = "ML recommends pit"
    elif probability >= THRESHOLD:
        note = "ML leans towards pit"
    elif probability >= THRESHOLD * 0.5:
        note = "ML leans towards staying out"
    else:
        note = "ML recommends staying out"

    return {
        'probability': probability,
        'prediction': prediction,
        'note': note
    }

def combine(rule_result, ml_result):
    rule_signal = rule_result['signal']
    rule_conf   = rule_result['confidence']
    ml_prob     = ml_result['probability']
    ml_pred     = ml_result['prediction']

    PIT_THRESHOLD  = THRESHOLD
    STAY_THRESHOLD = THRESHOLD * 0.5

    # CASE 1 — Both agree: pit
    if rule_signal == 1 and ml_pred == 1:
        return {
            'final_decision': 1,
            'confidence': 'HIGH' if rule_conf == 'HIGH' else 'MEDIUM',
            'reasoning': 'Rule engine and ML model both recommend pit',
            'ml_probability': ml_prob,
            'rule_signal': rule_signal,
            'rule_reason': rule_result['reason']
        }

    # CASE 2 — Both agree: stay out
    if rule_signal == 0 and ml_pred == 0:
        return {
            'final_decision': 0,
            'confidence': 'HIGH' if rule_conf == 'HIGH' else 'MEDIUM',
            'reasoning': 'Rule engine and ML model both recommend stay out',
            'ml_probability': ml_prob,
            'rule_signal': rule_signal,
            'rule_reason': rule_result['reason']
        }

    # CASE 3 — Rule says pit, ML says stay out
    if rule_signal == 1 and ml_pred == 0:
        return {
            'final_decision': 1,
            'confidence': 'MEDIUM',
            'reasoning': f'Rule engine recommends pit — ML is uncertain '
                         f'(prob={ml_prob:.4f} vs threshold={THRESHOLD:.4f}). '
                         f'Trusting rules.',
            'ml_probability': ml_prob,
            'rule_signal': rule_signal,
            'rule_reason': rule_result['reason']
        }

    # CASE 4 — Rule says stay out, ML says pit
    if rule_signal == 0 and ml_pred == 1:
        return {
            'final_decision': 0,
            'confidence': 'LOW',
            'reasoning': f'Rule engine says stay out but ML sees pit '
                         f'signal (prob={ml_prob:.4f}). Monitor closely.',
            'ml_probability': ml_prob,
            'rule_signal': rule_signal,
            'rule_reason': rule_result['reason']
        }

    # CASE 5 — Rule uncertain, ML says pit
    if rule_signal == -1 and ml_pred == 1:
        return {
            'final_decision': 1,
            'confidence': 'MEDIUM',
            'reasoning': f'No clear rule signal — ML recommends pit '
                         f'(prob={ml_prob:.4f})',
            'ml_probability': ml_prob,
            'rule_signal': rule_signal,
            'rule_reason': rule_result['reason']
        }

    # CASE 6 — Rule uncertain, ML says stay out (default)
    return {
        'final_decision': 0,
        'confidence': 'LOW',
        'reasoning': f'No clear signal from either system '
                     f'(prob={ml_prob:.4f}). Stay out by default.',
        'ml_probability': ml_prob,
        'rule_signal': rule_signal,
        'rule_reason': rule_result['reason']
    }

def recommend(race_state):
    rule_result = evaluate_rules(race_state)
    feature_row = build_feature_row(race_state)
    ml_result = get_ml_signal(feature_row)
    decision = combine(rule_result, ml_result)
    
    decision['rule_reason'] = rule_result['reason']
    decision['ml_note'] = ml_result['note']
    decision['action_label'] = "PIT THIS LAP" if decision['final_decision'] == 1 else "STAY OUT"
    decision['color_code'] = "red" if decision['final_decision'] == 1 else "green"
    
    return decision

def test_strategy():
    print("--- Running Strategy Tests ---")
    
    # Scenario A — Clear pit
    scen_a = {
        'safety_car_active': 1, 'tyre_age': 25, 'compound': 'HARD',
        'gap_ahead': 3.0, 'gap_behind': 5.0, 'laps_remaining': 20,
        'position': 4, 'deg_rate': 0.05, 'lap_time_delta': 1.2,
        'track_temp': 32, 'fuel_adjusted_laptime': 88.0
    }
    res_a = recommend(scen_a)
    print(f"Scenario A: {res_a['action_label']} | Confidence={res_a['confidence']} | ML prob={res_a['ml_probability']:.2f} | Reason={res_a['reasoning']}")
    if res_a['final_decision'] == 1:
        print("PASS\n")
    else:
        print("FAIL\n")

    # Scenario B — Clear stay out
    scen_b = {
        'safety_car_active': 0, 'tyre_age': 8, 'compound': 'SOFT',
        'gap_ahead': 5.0, 'gap_behind': 8.0, 'laps_remaining': 40,
        'position': 3, 'deg_rate': 0.03, 'lap_time_delta': 0.3,
        'track_temp': 35, 'fuel_adjusted_laptime': 91.0
    }
    res_b = recommend(scen_b)
    print(f"Scenario B: {res_b['action_label']} | Confidence={res_b['confidence']} | ML prob={res_b['ml_probability']:.2f} | Reason={res_b['reasoning']}")
    if res_b['final_decision'] == 0:
        print("PASS\n")
    else:
        print("FAIL\n")

    # Scenario C — Undercut opportunity
    scen_c = {
        'safety_car_active': 0, 'tyre_age': 22, 'compound': 'MEDIUM',
        'gap_ahead': 1.8, 'gap_behind': 6.0, 'laps_remaining': 28,
        'position': 5, 'deg_rate': 0.07, 'lap_time_delta': 1.4,
        'track_temp': 40, 'fuel_adjusted_laptime': 90.0
    }
    res_c = recommend(scen_c)
    print(f"Scenario C: {res_c['action_label']} | Confidence={res_c['confidence']} | ML prob={res_c['ml_probability']:.2f} | Reason={res_c['reasoning']}")
    if res_c['final_decision'] == 1:
        print("PASS\n")
    else:
        print("FAIL\n")

    # Scenario D — End of race, stay out
    scen_d = {
        'safety_car_active': 0, 'tyre_age': 35, 'compound': 'HARD',
        'gap_ahead': 2.0, 'gap_behind': 1.5, 'laps_remaining': 3,
        'position': 2, 'deg_rate': 0.09, 'lap_time_delta': 2.1,
        'track_temp': 38, 'fuel_adjusted_laptime': 87.5
    }
    res_d = recommend(scen_d)
    print(f"Scenario D: {res_d['action_label']} | Confidence={res_d['confidence']} | ML prob={res_d['ml_probability']:.2f} | Reason={res_d['reasoning']}")
    if res_d['final_decision'] == 0:
        print("PASS\n")
    else:
        print("FAIL\n")

def main():
    try:
        test_strategy()
    except Exception as e:
        logger.error(f"Error executing strategy tests: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
