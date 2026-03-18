import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_rules(race_state):
    """
    Evaluates F1 pit stop rules in priority order.
    Returns: dict with signal (int), reason (str), confidence (str).
    signal: 1 (pit), 0 (stay out), -1 (uncertain)
    """
    # Extract keys safely with defaults just in case
    tyre_age = race_state.get('tyre_age', 0)
    compound = race_state.get('compound', 'UNKNOWN').upper()
    gap_ahead = race_state.get('gap_ahead', 0.0)
    gap_behind = race_state.get('gap_behind', 0.0)
    laps_remaining = race_state.get('laps_remaining', 100)
    position = race_state.get('position', 1)
    safety_car_active = race_state.get('safety_car_active', 0)
    deg_rate = race_state.get('deg_rate', 0.0)
    lap_time_delta = race_state.get('lap_time_delta', 0.0)

    # Minimum stint lengths
    min_stint = {
        'SOFT': 10, 'MEDIUM': 15, 'HARD': 20, 
        'INTERMEDIATE': 5, 'WET': 5
    }
    compound_min = min_stint.get(compound, 10)
    pit_window_open = tyre_age >= compound_min

    # RULE 1 — SAFETY CAR FREE STOP
    if safety_car_active == 1 and tyre_age >= 10:
        return {
            "signal": 1, 
            "reason": "Safety car: free pit stop opportunity", 
            "confidence": "HIGH"
        }

    # RULE 2 — TOO LATE TO PIT
    if laps_remaining <= 4:
        return {
            "signal": 0, 
            "reason": "Too late in race to pit", 
            "confidence": "HIGH"
        }

    # RULE 3 — PIT WINDOW NOT OPEN
    if not pit_window_open:
        return {
            "signal": 0, 
            "reason": f"Pit window not open yet (lap {tyre_age} of minimum {compound_min})", 
            "confidence": "HIGH"
        }

    # RULE 4 — CRITICAL DEGRADATION
    if lap_time_delta >= 2.5 and pit_window_open:
        return {
            "signal": 1, 
            "reason": f"Critical tyre degradation: {lap_time_delta:.1f}s off pace", 
            "confidence": "HIGH"
        }

    # RULE 5 — UNDERCUT OPPORTUNITY
    if gap_ahead > 0 and gap_ahead <= 2.5 and tyre_age >= 18:
        return {
            "signal": 1,
            "reason": f"Undercut window: {gap_ahead:.1f}s ahead, tyre age {tyre_age} laps",
            "confidence": "MEDIUM"
        }

    # RULE 6 — OVERCUT DEFENCE
    if gap_behind <= 1.5 and gap_behind > 0 and tyre_age <= 15:
        return {
            "signal": 0,
            "reason": f"Overcut defence: stay out to build gap ({gap_behind:.1f}s behind)",
            "confidence": "MEDIUM"
        }

    # RULE 7 — NORMAL DEGRADATION PIT
    if deg_rate >= 0.06 and tyre_age >= 20:
        return {
            "signal": 1,
            "reason": f"High degradation rate at tyre age {tyre_age} laps",
            "confidence": "MEDIUM"
        }

    # RULE 7b — MODERATE DEG IN OPEN WINDOW
    if deg_rate >= 0.05 and tyre_age >= 25 and pit_window_open:
        return {
            "signal": 1,
            "reason": f"Moderate degradation building — {tyre_age} laps on tyres",
            "confidence": "MEDIUM"
        }

    # RULE 7c — LONG STINT WARNING
    if tyre_age >= 35:
        return {
            "signal": 1,
            "reason": f"Extended stint warning: {tyre_age} laps on current set",
            "confidence": "MEDIUM"
        }

    # RULE 7d — CONSERVATIVE LEADER STAY
    if position == 1 and gap_ahead == 0 and pit_window_open and deg_rate < 0.05:
        return {
            "signal": 0,
            "reason": "Leading the race with manageable degradation — protect position",
            "confidence": "MEDIUM"
        }

    # RULE 8 — DEFAULT UNCERTAIN
    return {
        "signal": -1, 
        "reason": "No clear rule signal", 
        "confidence": "LOW"
    }

def format_rule_output(result, race_state):
    """
    Returns a human-readable string summarising the recommendation.
    """
    signal_map = {1: "PIT THIS LAP", 0: "STAY OUT", -1: "UNCERTAIN"}
    action = signal_map.get(result['signal'], "UNKNOWN")
    
    out = (
        f"--- PIT STOP ENGINE RECOMMENDATION ---\n"
        f"ACTION: {action} (Confidence: {result['confidence']})\n"
        f"REASON: {result['reason']}\n\n"
        f"--- RACE STATE CONTEXT ---\n"
        f"Tyre: {race_state.get('compound', 'Unknown')} (Age: {race_state.get('tyre_age', 0)})\n"
        f"Laps Remaining: {race_state.get('laps_remaining', 0)} | "
        f"Position: P{race_state.get('position', 0)}\n"
        f"Gap Ahead: {race_state.get('gap_ahead', 0.0):.1f}s | "
        f"Gap Behind: {race_state.get('gap_behind', 0.0):.1f}s\n"
        f"SC Active: {'Yes' if race_state.get('safety_car_active', 0) == 1 else 'No'} | "
        f"Pace Drop: {race_state.get('lap_time_delta', 0.0):.2f}s | "
        f"Deg Rate: {race_state.get('deg_rate', 0.0):.3f}"
    )
    return out

def test_rules():
    print("Running Rule Engine Tests...")
    
    # Test 1: Safety car out, tyre age 20 → signal should be 1
    t1_state = {
        'tyre_age': 20, 'compound': 'HARD', 'gap_ahead': 5.0, 
        'gap_behind': 5.0, 'laps_remaining': 20, 'position': 3, 
        'safety_car_active': 1, 'deg_rate': 0.05, 'lap_time_delta': 1.0
    }
    t1_res = evaluate_rules(t1_state)
    assert t1_res['signal'] == 1, f"Test 1 Failed: {t1_res}"
    print("Test 1 PASS: Safety car detected")

    # Test 2: 3 laps remaining → signal should be 0
    t2_state = {
        'tyre_age': 30, 'compound': 'HARD', 'gap_ahead': 5.0, 
        'gap_behind': 5.0, 'laps_remaining': 3, 'position': 3, 
        'safety_car_active': 0, 'deg_rate': 0.5, 'lap_time_delta': 5.0
    }
    t2_res = evaluate_rules(t2_state)
    assert t2_res['signal'] == 0, f"Test 2 Failed: {t2_res}"
    print("Test 2 PASS: Too late to pit detected")

    # Test 3: SOFT tyre, age 8 laps → signal should be 0 (window not open)
    t3_state = {
        'tyre_age': 8, 'compound': 'SOFT', 'gap_ahead': 5.0, 
        'gap_behind': 5.0, 'laps_remaining': 40, 'position': 3, 
        'safety_car_active': 0, 'deg_rate': 0.1, 'lap_time_delta': 1.0
    }
    t3_res = evaluate_rules(t3_state)
    assert t3_res['signal'] == 0, f"Test 3 Failed: {t3_res}"
    print("Test 3 PASS: Pit window not open detected")

    # Test 4: gap_ahead = 2.0s, tyre_age = 22 → signal should be 1 (undercut)
    # Using MEDIUM tyre so window is open (min 15)
    t4_state = {
        'tyre_age': 22, 'compound': 'MEDIUM', 'gap_ahead': 2.0, 
        'gap_behind': 10.0, 'laps_remaining': 30, 'position': 2, 
        'safety_car_active': 0, 'deg_rate': 0.01, 'lap_time_delta': 1.0
    }
    t4_res = evaluate_rules(t4_state)
    assert t4_res['signal'] == 1, f"Test 4 Failed: {t4_res}"
    print("Test 4 PASS: Undercut opportunity detected")

    # Test 5: lap_time_delta = 3.0s, tyre_age = 25 → signal should be 1 (critical deg)
    t5_state = {
        'tyre_age': 25, 'compound': 'HARD', 'gap_ahead': 10.0, 
        'gap_behind': 10.0, 'laps_remaining': 20, 'position': 1, 
        'safety_car_active': 0, 'deg_rate': 0.02, 'lap_time_delta': 3.0
    }
    t5_res = evaluate_rules(t5_state)
    assert t5_res['signal'] == 1, f"Test 5 Failed: {t5_res}"
    print("Test 5 PASS: Critical degradation detected")

    # Test 6: Moderate deg at lap 28, MEDIUM tyre → signal should be 1 (Rule 7b)
    t6_state = {
        'tyre_age': 28, 'compound': 'MEDIUM', 'gap_ahead': 5.0,
        'gap_behind': 5.0, 'laps_remaining': 20, 'position': 3,
        'safety_car_active': 0, 'deg_rate': 0.055, 'lap_time_delta': 1.2
    }
    t6_res = evaluate_rules(t6_state)
    assert t6_res['signal'] == 1, f"Test 6 Failed: {t6_res}"
    print("Test 6 PASS: Moderate degradation detected")
    print("All tests passed!\n")

def main():
    try:
        test_rules()
        
        # Example Scenario
        example_state = {
            'tyre_age': 24,
            'compound': 'MEDIUM',
            'gap_ahead': 1.5,
            'gap_behind': 4.2,
            'laps_remaining': 28,
            'position': 4,
            'safety_car_active': 0,
            'deg_rate': 0.09,
            'lap_time_delta': 1.8
        }
        
        result = evaluate_rules(example_state)
        print(format_rule_output(result, example_state))
        
    except Exception as e:
        logger.error(f"Error in rule engine: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
