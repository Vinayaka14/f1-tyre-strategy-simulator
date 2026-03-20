import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import json, os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.strategy import recommend
from src.tyre_model import tyre_life_remaining_pct, predict_lap_delta

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="F1 Tyre Strategy Simulator",
    page_icon="🏎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Session State Init ───────────────────────────────────
# scenario_log intentionally NOT using setdefault — resets on every new session
if 'scenario_log' not in st.session_state:
    st.session_state['scenario_log'] = []
if 'logged_keys' not in st.session_state:
    st.session_state['logged_keys'] = set()
st.session_state.setdefault('last_decision', None)
st.session_state.setdefault('active_tab', 'real_race')
st.session_state.setdefault('race_data_cache', {})
st.session_state.setdefault('actual_gap_ahead', 3.0)
st.session_state.setdefault('actual_gap_behind', 3.0)
st.session_state.setdefault('actual_position', 1)
st.session_state.setdefault('actual_driver_ahead', None)
st.session_state.setdefault('actual_driver_behind', None)

# ─── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    [data-testid="metric-container"] label {
        font-size: 0.75rem;
        color: #888888;
    }

    section[data-testid="stSidebar"] {
        border-right: 2px solid #E24B4A;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1a1a1a;
        padding: 4px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #0f0f0f;
        color: #888888;
        border-radius: 6px;
        padding: 8px 24px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E24B4A !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── App Header ────────────────────────────────────────────
st.markdown("""
<div style="display:flex; align-items:center; 
            gap:12px; margin-bottom:8px;">
    <span style="font-size:2rem;">🏎</span>
    <div>
        <h1 style="margin:0; font-size:1.6rem; 
                   color:#ffffff;">
            F1 Tyre Strategy Simulator
        </h1>
        <p style="margin:0; color:#888888; font-size:0.85rem;">
            Real-time pit stop decision engine — 
            powered by XGBoost + Rule Engine
        </p>
    </div>
</div>
<hr style="border-color:#E24B4A; margin-bottom:16px;">
""", unsafe_allow_html=True)


# ─── Cached Data Loaders ──────────────────────────────────
@st.cache_data
def get_race_schedule(season):
    import fastf1
    schedule = fastf1.get_event_schedule(season)
    races = schedule[schedule['EventFormat'] == 'conventional']['EventName'].tolist()
    return races


@st.cache_data(show_spinner=False)
def load_race_session(season, grand_prix):
    """
    Loads a FastF1 race session and returns a
    cleaned DataFrame of all laps for all drivers.
    Returns None if loading fails.
    """
    import fastf1
    cache_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    'data', 'raw'
    )
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    try:
        session = fastf1.get_session(season, grand_prix, 'R')
        session.load(telemetry=False, weather=True, messages=False)
        laps = session.laps.copy()

        cols_needed = [
            'Driver', 'LapNumber', 'Compound',
            'TyreLife', 'LapTime', 'Position',
            'PitInTime', 'PitOutTime',
            'Sector1Time', 'Sector2Time', 'Sector3Time'
        ]
        laps = laps[cols_needed].copy()

        # Convert LapTime to seconds
        laps['LapTime_s'] = laps['LapTime'].dt.total_seconds()

        # Flag pit laps
        laps['Pitted'] = laps['PitInTime'].notna()

        # Get weather for track temp
        weather = session.weather_data
        if weather is not None and len(weather) > 0:
            avg_track_temp = weather['TrackTemp'].mean()
        else:
            avg_track_temp = 30.0

        total_laps = int(laps['LapNumber'].max())

        return {
            'laps': laps,
            'total_laps': total_laps,
            'track_temp': avg_track_temp,
            'session_name': f"{grand_prix} {season}",
            'weather_data': weather
        }
    except Exception as e:
        return None


# ─── Helper Functions ──────────────────────────────────────
def get_driver_lap_data(session_data, driver, lap_number):
    """
    Extracts the race state for a specific driver
    at a specific lap number.
    Returns a dict ready for strategy.recommend().
    """
    laps = session_data['laps']
    driver_laps = laps[laps['Driver'] == driver].copy()

    if driver_laps.empty:
        return None

    current = driver_laps[driver_laps['LapNumber'] == lap_number]
    if current.empty:
        available = driver_laps['LapNumber'].values
        nearest = available[np.argmin(np.abs(available - lap_number))]
        current = driver_laps[driver_laps['LapNumber'] == nearest]

    row = current.iloc[0]

    tyre_age = int(row['TyreLife']) if pd.notna(row['TyreLife']) else 1
    stint_start = lap_number - tyre_age + 1
    stint_laps = driver_laps[
        (driver_laps['LapNumber'] >= stint_start) &
        (driver_laps['LapNumber'] <= lap_number) &
        (driver_laps['LapTime_s'].notna())
    ]['LapTime_s'].values

    if len(stint_laps) >= 3:
        x = np.arange(len(stint_laps[-3:]))
        deg_rate = float(np.polyfit(x, stint_laps[-3:], 1)[0])
        deg_rate = max(0.0, deg_rate)
        # Fallback: polyfit can give near-zero on flat circuits
        # use lap_time_delta / tyre_age as a more reliable proxy
        if deg_rate < 0.005 and tyre_age > 3:
            delta_proxy = float(stint_laps[-1] - np.min(stint_laps))
            deg_rate = round(delta_proxy / max(tyre_age, 1), 4)
    else:
        if tyre_age > 1 and len(stint_laps) > 1:
            delta_proxy = float(stint_laps[-1] - np.min(stint_laps))
            deg_rate = round(delta_proxy / max(tyre_age - 1, 1), 4)
        else:
            deg_rate = 0.05
    deg_rate = max(0.0, min(deg_rate, 0.5))

    if len(stint_laps) > 0:
        lap_time_delta = float(stint_laps[-1] - np.min(stint_laps)) if pd.notna(row['LapTime_s']) else 1.0
        current_laptime = float(stint_laps[-1])
    else:
        lap_time_delta = 1.0
        current_laptime = 90.0

    all_lap_data = laps[laps['LapNumber'] == lap_number].sort_values('Position')
    current_pos = int(row['Position']) if pd.notna(row['Position']) else 10

    gap_ahead = 2.0 if current_pos > 1 else 0.0
    gap_behind = 2.0 if current_pos < 20 else 0.0

    compound = str(row['Compound']) if pd.notna(row['Compound']) else 'MEDIUM'
    laps_remaining = session_data['total_laps'] - lap_number
    fuel_adjusted = current_laptime - (laps_remaining * 0.03)

    return {
        'tyre_age': tyre_age,
        'compound': compound,
        'gap_ahead': gap_ahead,
        'gap_behind': gap_behind,
        'laps_remaining': laps_remaining,
        'position': current_pos,
        'safety_car_active': 0,
        'deg_rate': deg_rate,
        'lap_time_delta': lap_time_delta,
        'track_temp': session_data['track_temp'],
        'fuel_adjusted_laptime': fuel_adjusted,
        'current_laptime': current_laptime,
        'all_lap_positions': all_lap_data,
        'driver_laps': driver_laps,
        'stint_laps': stint_laps.tolist()
    }


def build_race_state_from_overrides(
        tyre_age, compound, lap_number,
        total_laps, position, deg_rate,
        lap_time_delta, track_temp,
        fuel_adjusted_laptime):
    """
    Builds a race_state dict applying all sidebar
    overrides (SC, VSC, rain, gap overrides, pace override).
    Called by both tabs.
    """
    sc_active = 1 if st.session_state.safety_car else 0
    effective_deg = deg_rate * 0.5 if st.session_state.vsc else deg_rate
    effective_compound = 'INTERMEDIATE' if st.session_state.rain else compound
    
    # Apply pace delta override from sidebar
    effective_lap_delta = lap_time_delta + st.session_state.get('pace_delta_override', 0.0)

    return {
        'tyre_age': tyre_age,
        'compound': effective_compound,
        'gap_ahead': st.session_state.gap_ahead_override,
        'gap_behind': st.session_state.gap_behind_override,
        'laps_remaining': total_laps - lap_number,
        'position': position,
        'safety_car_active': sc_active,
        'deg_rate': effective_deg,
        'lap_time_delta': effective_lap_delta,
        'track_temp': track_temp,
        'fuel_adjusted_laptime': fuel_adjusted_laptime
    }


def log_scenario(race_state, decision, mode):
    """Appends a scenario to the session state log."""
    entry = {
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'mode': mode,
        'compound': race_state['compound'],
        'tyre_age': race_state['tyre_age'],
        'laps_remaining': race_state['laps_remaining'],
        'safety_car': race_state['safety_car_active'],
        'gap_ahead': race_state['gap_ahead'],
        'gap_behind': race_state['gap_behind'],
        'decision': decision['action_label'],
        'confidence': decision['confidence'],
        'ml_probability': round(decision['ml_probability'], 4),
        'rule_triggered': decision['rule_reason']
    }
    st.session_state.scenario_log.append(entry)


def render_strategy_card(decision):
    """Renders the main recommendation banner and detail metrics."""
    bg = "#3d0a0a" if decision['final_decision'] == 1 else "#0a3d1a"
    border = "#E24B4A" if decision['final_decision'] == 1 else "#00C851"
    icon = "🔴" if decision['final_decision'] == 1 else "🟢"

    st.markdown(f"""
    <div style="background:{bg}; border:2px solid {border};
                border-radius:12px; padding:20px 24px;
                text-align:center; margin-bottom:16px;">
        <div style="font-size:2rem; font-weight:900;
                    color:white; letter-spacing:2px;">
            {icon} {decision['action_label']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        conf_color = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}
        st.metric("Confidence",
                  f"{conf_color.get(decision['confidence'], '')} "
                  f"{decision['confidence']}")
    with c2:
        sig_map = {1: "PIT ✓", 0: "STAY OUT ✓", -1: "UNCERTAIN ⚠"}
        st.metric("Rule Engine",
                  sig_map.get(decision['rule_signal'], '—'))
    with c3:
        st.metric("ML Model",
                  decision['ml_note'].split()[0],
                  help=f"Raw probability: "
                       f"{decision['ml_probability']:.4f} | "
                       f"Threshold: 0.842")

    with st.expander("💡 Why this recommendation?", expanded=True):
        st.markdown(f"**Strategy reasoning:**  \n"
                    f"{decision['reasoning']}")
        st.markdown(f"**Rule triggered:**  \n"
                    f"{decision['rule_reason']}")
        st.markdown(f"**ML assessment:**  \n"
                    f"{decision['ml_note']}")




def get_actual_gap_context(session_laps, driver, lap_number):
    """
    Returns actual gap context for a driver at a specific lap — position, 
    driver ahead, driver behind, and estimated gaps.
    Returns None if data unavailable.
    """
    lap_data = session_laps[session_laps['LapNumber'] == lap_number][['Driver', 'Position']].copy()

    # Fall back to nearest lap if exact not found
    if lap_data.empty:
        available = session_laps['LapNumber'].unique()
        nearest = available[np.argmin(np.abs(available - lap_number))]
        lap_data = session_laps[session_laps['LapNumber'] == nearest][['Driver', 'Position']].copy()

    if lap_data.empty:
        return None

    lap_data = lap_data.dropna(subset=['Position']).drop_duplicates(subset=['Driver']).sort_values('Position')

    driver_row = lap_data[lap_data['Driver'] == driver]
    if driver_row.empty:
        return None

    current_pos = int(driver_row['Position'].iloc[0])

    # Driver ahead
    ahead_row = lap_data[lap_data['Position'] == current_pos - 1]
    driver_ahead = str(ahead_row['Driver'].iloc[0]) if not ahead_row.empty else None

    # Driver behind
    behind_row = lap_data[lap_data['Position'] == current_pos + 1]
    driver_behind = str(behind_row['Driver'].iloc[0]) if not behind_row.empty else None

    # Estimate gaps from cumulative lap time difference
    gap_ahead_est = 2.0 if driver_ahead else 0.0
    gap_behind_est = 2.0 if driver_behind else 0.0

    try:
        cum_times = {}
        for drv in [driver, driver_ahead, driver_behind]:
            if drv is None: continue
            drv_laps = session_laps[
                (session_laps['Driver'] == drv) &
                (session_laps['LapNumber'] <= lap_number) &
                (session_laps['LapTime_s'].notna()) &
                (session_laps['LapTime_s'] < 200)
            ]['LapTime_s'].sum()
            if drv_laps > 0:
                cum_times[drv] = drv_laps

        if driver in cum_times:
            if driver_ahead in cum_times:
                gap_ahead_est = abs(cum_times[driver] - cum_times[driver_ahead])
                gap_ahead_est = min(gap_ahead_est, 60.0)
            if driver_behind in cum_times:
                gap_behind_est = abs(cum_times[driver_behind] - cum_times[driver])
                gap_behind_est = min(gap_behind_est, 60.0)
    except Exception:
        pass

    return {
        'position': current_pos,
        'driver_ahead': driver_ahead,
        'driver_behind': driver_behind,
        'gap_ahead': round(gap_ahead_est, 1),
        'gap_behind': round(gap_behind_est, 1)
    }


def render_position_bar(session_laps, lap_number, selected_driver):
    """Renders race position bar for all drivers at the given lap."""
    TEAM_COLORS = {
        'VER': '#3671C6', 'PER': '#3671C6',
        'LEC': '#E8002D', 'SAI': '#E8002D',
        'HAM': '#27F4D2', 'RUS': '#27F4D2',
        'NOR': '#FF8000', 'PIA': '#FF8000',
        'ALO': '#358C75', 'STR': '#358C75',
        'GAS': '#B6BABD', 'OCO': '#B6BABD',
        'TSU': '#6692FF', 'RIC': '#6692FF',
        'LAW': '#6692FF', 'HAD': '#6692FF',
        'MAG': '#B6BABD', 'HUL': '#B6BABD',
        'BOT': '#C92D4B', 'ZHO': '#C92D4B',
        'ALB': '#64C4FF', 'SAR': '#64C4FF',
        'BEA': '#6692FF', 'ANT': '#27F4D2'
    }
    COMPOUND_COLORS = {
        'SOFT': '#E8002D', 'MEDIUM': '#FFF200',
        'HARD': '#FFFFFF', 'INTERMEDIATE': '#43B02A',
        'WET': '#0067FF', 'UNKNOWN': '#888888'
    }

    lap_data = session_laps[
        session_laps['LapNumber'] == lap_number
    ][['Driver', 'Position', 'Compound', 'TyreLife']].copy()

    if lap_data.empty:
        available_laps = session_laps['LapNumber'].unique()
        nearest = available_laps[np.argmin(np.abs(available_laps - lap_number))]
        lap_data = session_laps[
            session_laps['LapNumber'] == nearest
        ][['Driver', 'Position', 'Compound', 'TyreLife']].copy()

    lap_data = lap_data.dropna(subset=['Position']).sort_values('Position')

    if lap_data.empty:
        st.caption("Position data not available for this lap.")
        return

    lap_data = lap_data.drop_duplicates(subset=['Driver'], keep='first')

    fig = go.Figure()

    fig.add_shape(type="line",
        x0=0.5, x1=len(lap_data) + 0.5,
        y0=0, y1=0,
        line=dict(color="#333333", width=4))

    for _, row in lap_data.iterrows():
        drv = str(row['Driver'])
        pos = int(row['Position'])
        compound = str(row['Compound']) if pd.notna(row['Compound']) else 'UNKNOWN'
        tyre_age = int(row['TyreLife']) if pd.notna(row['TyreLife']) else 0
        is_selected = (drv == selected_driver)
        color = TEAM_COLORS.get(drv, '#888888')
        comp_color = COMPOUND_COLORS.get(compound, '#888888')

        fig.add_trace(go.Scatter(
            x=[pos], y=[0.15],
            mode='markers+text',
            marker=dict(
                size=24 if is_selected else 18,
                color=color,
                line=dict(
                    color='white' if is_selected else 'rgba(0,0,0,0)',
                    width=3 if is_selected else 0
                )
            ),
            text=[drv],
            textposition='top center',
            textfont=dict(size=11 if is_selected else 9, color='white'),
            hovertemplate=(
                f"<b>P{pos} — {drv}</b><br>"
                f"Compound: {compound}<br>"
                f"Tyre age: {tyre_age} laps<extra></extra>"
            ),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=[pos], y=[-0.2],
            mode='markers',
            marker=dict(size=10, color=comp_color, symbol='square'),
            hovertemplate=f"{compound}<br>Age: {tyre_age} laps<extra></extra>",
            showlegend=False
        ))

    if selected_driver in lap_data['Driver'].values:
        sel_pos = int(lap_data[lap_data['Driver'] == selected_driver]['Position'].values[0])
        fig.add_vline(
            x=sel_pos, line_dash="dot",
            line_color="#E24B4A", line_width=1.5,
            annotation_text=f"◀ {selected_driver}",
            annotation_font_color="#E24B4A",
            annotation_font_size=11
        )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=200,
        margin=dict(l=20, r=20, t=10, b=50),
        xaxis=dict(
            title="Race position →",
            tickvals=list(range(1, len(lap_data) + 1)),
            ticktext=[f"P{i}" for i in range(1, len(lap_data) + 1)],
            gridcolor='#1a1a1a', color='#888888',
            tickfont=dict(size=9)
        ),
        yaxis=dict(visible=False, range=[-0.5, 0.7])
    )
    st.plotly_chart(fig, use_container_width=True)

    leg1, leg2, leg3, leg4, leg5 = st.columns(5)
    with leg1:
        st.markdown("<span style='color:#E8002D'>■</span> "
                    "<span style='color:#888;font-size:0.75rem'>Soft</span>",
                    unsafe_allow_html=True)
    with leg2:
        st.markdown("<span style='color:#FFF200'>■</span> "
                    "<span style='color:#888;font-size:0.75rem'>Medium</span>",
                    unsafe_allow_html=True)
    with leg3:
        st.markdown("<span style='color:#FFFFFF'>■</span> "
                    "<span style='color:#888;font-size:0.75rem'>Hard</span>",
                    unsafe_allow_html=True)
    with leg4:
        st.markdown("<span style='color:#43B02A'>■</span> "
                    "<span style='color:#888;font-size:0.75rem'>Inter</span>",
                    unsafe_allow_html=True)
    with leg5:
        st.markdown("<span style='color:#888;font-size:0.75rem'>◉ = selected driver</span>",
                    unsafe_allow_html=True)


def render_race_history(session_data, driver):
    """Shows actual race history: pit stops, SC/VSC periods, fastest lap, result."""
    laps = session_data['laps']
    driver_laps = laps[laps['Driver'] == driver].copy()

    if driver_laps.empty:
        st.caption("No history data available.")
        return

    pit_laps = driver_laps[driver_laps['Pitted'] == True][['LapNumber', 'Compound', 'TyreLife']].copy()

    valid_times = driver_laps[driver_laps['LapTime_s'].notna() & (driver_laps['LapTime_s'] < 200)]
    if not valid_times.empty:
        fastest_idx = valid_times['LapTime_s'].idxmin()
        fastest_lap = int(valid_times.loc[fastest_idx, 'LapNumber'])
        fastest_time = float(valid_times.loc[fastest_idx, 'LapTime_s'])
    else:
        fastest_lap = None
        fastest_time = None

    last_lap = driver_laps[driver_laps['LapNumber'] == driver_laps['LapNumber'].max()]
    final_pos = int(last_lap['Position'].iloc[0]) \
        if not last_lap.empty and pd.notna(last_lap['Position'].iloc[0]) else 'DNF'

    lap_medians = laps.groupby('LapNumber')['LapTime_s'].median()
    session_median = lap_medians.median()
    sc_proxy_laps = lap_medians[lap_medians > session_median * 1.25].index.tolist()

    sc_periods = []
    if sc_proxy_laps:
        sc_proxy_laps = sorted(sc_proxy_laps)
        start = sc_proxy_laps[0]
        prev = sc_proxy_laps[0]
        for lap in sc_proxy_laps[1:]:
            if lap - prev > 2:
                sc_periods.append((start, prev))
                start = lap
            prev = lap
        sc_periods.append((start, prev))

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Final Position", f"P{final_pos}")
    with m2:
        st.metric("Pit Stops", len(pit_laps))
    with m3:
        if fastest_lap:
            st.metric("Fastest Lap", f"Lap {fastest_lap}", help=f"{fastest_time:.3f}s")
        else:
            st.metric("Fastest Lap", "N/A")
    with m4:
        st.metric("SC Periods Detected", len(sc_periods))

    st.markdown("---")

    events = []
    for _, p in pit_laps.iterrows():
        events.append({
            'Lap': int(p['LapNumber']),
            'Event Type': '🔧 Pit Stop',
            'Detail': f"New {p['Compound']} tyres",
            'Tyre Age at Stop': f"{int(p['TyreLife'])} laps",
        })
    for (s, e) in sc_periods:
        events.append({
            'Lap': s,
            'Event Type': '🚗 Safety Car (proxy)',
            'Detail': f"Laps {s}–{e}",
            'Tyre Age at Stop': '—',
        })
    if fastest_lap:
        events.append({
            'Lap': fastest_lap,
            'Event Type': '⚡ Fastest Lap',
            'Detail': f"{fastest_time:.3f}s",
            'Tyre Age at Stop': '—',
        })

    if not events:
        st.caption("No events found for this driver.")
        return

    events_df = pd.DataFrame(events).sort_values('Lap').reset_index(drop=True)
    st.dataframe(
        events_df[['Lap', 'Event Type', 'Detail', 'Tyre Age at Stop']],
        use_container_width=True, hide_index=True
    )

    st.markdown("**Stint breakdown:**")
    stints = []
    stint_start = 1
    prev_compound = None
    for _, row in driver_laps.sort_values('LapNumber').iterrows():
        compound = str(row['Compound']) if pd.notna(row['Compound']) else 'UNKNOWN'
        if prev_compound is None:
            prev_compound = compound
        if row['Pitted'] == True:
            stints.append({
                'Stint': len(stints) + 1,
                'Laps': f"{stint_start}–{int(row['LapNumber'])}",
                'Compound': prev_compound,
                'Length': int(row['TyreLife'])
            })
            stint_start = int(row['LapNumber']) + 1
            prev_compound = compound

    last_lap_num = int(driver_laps['LapNumber'].max())
    if stint_start <= last_lap_num:
        last_compound = str(
            driver_laps[driver_laps['LapNumber'] == last_lap_num]['Compound'].iloc[0])
        stints.append({
            'Stint': len(stints) + 1,
            'Laps': f"{stint_start}–{last_lap_num}",
            'Compound': last_compound,
            'Length': last_lap_num - stint_start + 1
        })

    if stints:
        st.dataframe(pd.DataFrame(stints), use_container_width=True, hide_index=True)


def render_tyre_simulation(compound_selected, tyre_age_current, base_laptime, lap_number, total_laps, manual_deg_rate=None):
    """
    Simulates tyre degradation curves for all 5 compounds from the current lap forward.
    Uses tyre_model.py predict_lap_delta().
    """
    from src.tyre_model import predict_lap_delta
    COMPOUND_CONFIG = {
        'SOFT':         {'enc': 2, 'color': '#E8002D', 'max_age': 35, 'name': 'Soft'},
        'MEDIUM':       {'enc': 1, 'color': '#FFF200', 'max_age': 40, 'name': 'Medium'},
        'HARD':         {'enc': 0, 'color': '#FFFFFF', 'max_age': 55, 'name': 'Hard'},
        'INTERMEDIATE': {'enc': 3, 'color': '#43B02A', 'max_age': 50, 'name': 'Inter'},
        'WET':          {'enc': 4, 'color': '#0067FF', 'max_age': 40, 'name': 'Wet'}
    }

    rain_active = st.session_state.get('rain', False)
    # Handle both real race and manual builder compound logic
    actual_compound = st.session_state.get('compound_override', compound_selected)
    wet_mode = rain_active or actual_compound in ['INTERMEDIATE', 'WET']

    if wet_mode:
        st.markdown(
            "<div style='background:#0a2a0a; border:1px solid #43B02A; border-radius:6px; padding:6px 12px; "
            "font-size:0.82rem; color:#43B02A; margin-bottom:8px;'>🌧 Rain mode active — WET/INTERMEDIATE curves highlighted</div>",
            unsafe_allow_html=True
        )

    fig = go.Figure()
    critical_laps = {}
    pit_recovery_laps = {}
    PIT_STOP_TIME_LOSS = 22.0
    CRITICAL_DELTA = 2.5

    for compound, cfg in COMPOUND_CONFIG.items():
        if wet_mode:
            if compound in ['INTERMEDIATE', 'WET']:
                opacity = 1.0 if compound == compound_selected else 0.6
                visible = True
            else:
                opacity = 0.2
                visible = True
        else:
            if compound in ['INTERMEDIATE', 'WET']:
                opacity = 0.5
                visible = (compound == compound_selected)
            else:
                opacity = 1.0 if compound == compound_selected else 0.35
                visible = True

        trace_visible = True if visible else 'legendonly'
        
        # Start from current tyre age — show future only, not history
        start_age = max(0, tyre_age_current - 1)
        ages = list(range(start_age, cfg['max_age'] + 1))
        
        if manual_deg_rate is not None and compound == compound_selected:
            # Blend: use manual deg_rate as linear component, tyre model for curve shape
            model_deltas = [float(predict_lap_delta(cfg['enc'], a)) for a in ages]
            # Normalise model to start at 0
            model_base = model_deltas[0]
            model_norm = [d - model_base for d in model_deltas]
            # Scale normalised curve by ratio of manual_deg vs model's average rate
            model_avg_rate = (model_deltas[min(20, len(model_deltas)-1)] / 20) if len(model_deltas) > 20 else 0.05
            scale = (manual_deg_rate / model_avg_rate if model_avg_rate > 0 else 1.0)
            deltas = [model_base + (d * scale) for d in model_norm]
        else:
            deltas = [float(predict_lap_delta(cfg['enc'], a)) for a in ages]
        
        line_width = 2.5 if compound == compound_selected else 1.2
        
        crit_lap = next((ages[i] for i, d in enumerate(deltas) if d >= CRITICAL_DELTA), cfg['max_age'])
        critical_laps[compound] = crit_lap
        
        # Use manual_deg_rate for selected compound
        # fall back to model rate for others
        if (manual_deg_rate is not None and 
            compound == compound_selected and
            manual_deg_rate > 0):
            effective_rate = manual_deg_rate
        else:
            avg_deg = (
                deltas[min(10, len(deltas)-1)] / 10
            )
            effective_rate = avg_deg if avg_deg > 0 else 0.05
        
        recovery = round(
            PIT_STOP_TIME_LOSS / effective_rate, 1
        )
        # Cap at sensible max
        recovery = min(recovery, 200.0)
        pit_recovery_laps[compound] = recovery
        
        fig.add_trace(go.Scatter(
            x=ages, y=deltas, mode='lines',
            name=f"{cfg['name']} ({'selected' if compound == compound_selected else 'ref'})",
            line=dict(color=cfg['color'], width=line_width),
            opacity=opacity,
            visible=trace_visible,
            hovertemplate=(
                f"<b>{cfg['name']}</b><br>Tyre age: %{{x}} laps<br>"
                f"Pace loss: +%{{y:.3f}}s<br>Lap time: ~{base_laptime:.1f}s + delta<extra></extra>"
            )
        ))
        
        if visible and crit_lap <= cfg['max_age']:
            fig.add_trace(go.Scatter(
                x=[crit_lap], y=[CRITICAL_DELTA], mode='markers',
                marker=dict(symbol='x', size=10, color=cfg['color'], line=dict(width=2)),
                showlegend=False,
                hovertemplate=f"{cfg['name']} critical at lap {crit_lap}<extra></extra>"
            ))

    fig.add_hline(
        y=CRITICAL_DELTA, line_dash="dash", line_color="#E24B4A", line_width=1, opacity=0.6,
        annotation_text="Critical deg threshold (Rule 4)", annotation_font_color="#E24B4A",
        annotation_font_size=10, annotation_position="top right"
    )

    fig.add_vline(
        x=tyre_age_current, line_dash="dot", line_color="#FF8800", line_width=1.5,
        annotation_text=f"Now (age {tyre_age_current})", annotation_font_color="#FF8800",
        annotation_font_size=10
    )

    sel_critical = critical_laps.get(compound_selected, 0)
    sel_recovery = pit_recovery_laps.get(compound_selected, 0)
    laps_left_on_tyre = max(0, sel_critical - tyre_age_current)

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=320,
        margin=dict(l=50, r=20, t=50, b=50),
        title=dict(
            text=(f"Tyre degradation simulation — {compound_selected} selected  |  "
                  f"Critical in ~{laps_left_on_tyre} laps  |  Pit recovery: ~{sel_recovery} laps"),
            font=dict(color='#888888', size=11)
        ),
        xaxis=dict(title="Tyre age (laps)", gridcolor='#1a1a1a', color='#888888', dtick=5,
                   range=[max(0, tyre_age_current - 1), None]),
        yaxis=dict(title="Pace loss vs fresh tyre (s)", gridcolor='#1a1a1a', color='#888888'),
        legend=dict(bgcolor='rgba(26,26,26,0.8)', font=dict(color='#888888', size=10),
                    bordercolor='#333', borderwidth=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        color = "#E24B4A" if laps_left_on_tyre <= 5 else "#FF8800" if laps_left_on_tyre <= 12 else "#00C851"
        st.markdown(f"<div style='background:#1a1a1a; border-left:3px solid {color}; padding:8px 12px; "
                    f"border-radius:4px; font-size:0.82rem; color:{color};'>"
                    f"⏱ <b>{laps_left_on_tyre} laps</b> until critical deg on {compound_selected}</div>",
                    unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div style='background:#1a1a1a; border-left:3px solid #FF8800; padding:8px 12px; "
                    f"border-radius:4px; font-size:0.82rem; color:#FF8800;'>"
                    f"🔧 Pit loss (~22s) recovered in <b>~{sel_recovery} laps</b> vs staying out</div>",
                    unsafe_allow_html=True)
    with c3:
        laps_to_end = total_laps - lap_number
        viable = "✅ Tyres can finish" if laps_left_on_tyre >= laps_to_end else "⚠ Tyres won't last to finish"
        finish_color = "#00C851" if laps_left_on_tyre >= laps_to_end else "#E24B4A"
        st.markdown(f"<div style='background:#1a1a1a; border-left:3px solid {finish_color}; padding:8px 12px; "
                    f"border-radius:4px; font-size:0.82rem; color:{finish_color};'>"
                    f"{viable} ({laps_to_end} laps remaining)</div>", unsafe_allow_html=True)


def render_actual_race_chart(driver_laps, session_laps, lap_number, session_data):
    """Renders actual historical race lap times with pit markers, SC periods, and track temp."""
    from plotly.subplots import make_subplots
    valid = driver_laps[driver_laps['LapTime_s'].notna() & (driver_laps['LapTime_s'] < 200)].copy()
    if valid.empty:
        st.caption("No actual lap time data available.")
        return

    lap_medians = session_laps.groupby('LapNumber')['LapTime_s'].median()
    session_median = lap_medians.median()
    sc_laps = sorted(lap_medians[lap_medians > session_median * 1.25].index.tolist())
    
    sc_periods = []
    if sc_laps:
        start = sc_laps[0]
        prev = sc_laps[0]
        for l in sc_laps[1:]:
            if l - prev > 2:
                sc_periods.append((start, prev))
                start = l
            prev = l
        sc_periods.append((start, prev))

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for (s, e) in sc_periods:
        fig.add_vrect(x0=s, x1=e, fillcolor="rgba(255, 200, 0, 0.08)", layer="below", line_width=0,
                      annotation_text="SC", annotation_font_color="#FFB800", annotation_font_size=9)

    fig.add_trace(go.Scatter(x=valid['LapNumber'].tolist(), y=valid['LapTime_s'].tolist(),
                             mode='lines+markers', name='Lap time',
                             line=dict(color='#E24B4A', width=2), marker=dict(size=3, color='#E24B4A'),
                             hovertemplate="Lap %{x}<br>Time: %{y:.3f}s<extra></extra>"), secondary_y=False)


    try:
        weather = session_data.get('weather_data')
        if weather is not None and len(weather) > 0:
            total_laps = int(session_laps['LapNumber'].max())
            temp_laps = np.linspace(1, total_laps, len(weather))
            fig.add_trace(go.Scatter(x=temp_laps.tolist(), y=weather['TrackTemp'].tolist(),
                                     mode='lines', name='Track temp (°C)', line=dict(color='#888888', width=1, dash='dash'),
                                     opacity=0.5, hovertemplate="Lap ~%{x:.0f}<br>Track: %{y:.1f}°C<extra></extra>"),
                          secondary_y=True)
    except Exception: pass

    pit_laps = valid[valid['Pitted'] == True]
    for _, pr in pit_laps.iterrows():
        fig.add_vline(x=pr['LapNumber'], line_dash="solid", line_color="#E24B4A", line_width=1.5, opacity=0.7)
        fig.add_annotation(x=pr['LapNumber'], y=valid['LapTime_s'].max() * 0.98, text="PIT", showarrow=False,
                           font=dict(color="#E24B4A", size=9), bgcolor="rgba(61,10,10,0.8)", bordercolor="#E24B4A", borderwidth=1)

    fig.add_vline(x=lap_number, line_dash="dot", line_color="#FF8800", line_width=2, annotation_text="Now", annotation_font_color="#FF8800")

    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=320,
                      margin=dict(l=50, r=60, t=40, b=50),
                      title=dict(text="Actual race lap times — pit stops marked  |  toggle sectors in legend", font=dict(color='#888888', size=11)),
                      xaxis=dict(title="Race lap", gridcolor='#1a1a1a', color='#888888'),
                      legend=dict(bgcolor='rgba(26,26,26,0.8)', font=dict(color='#888888', size=10), bordercolor='#333', borderwidth=1))
    fig.update_yaxes(title_text="Lap time (s)", gridcolor='#1a1a1a', color='#888888', secondary_y=False, autorange='reversed')
    fig.update_yaxes(title_text="Track temp (°C)", color='#666666', showgrid=False, secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)


# ─── Sidebar ──────────────────────────────────────────────

# Section 1 — Mode indicator
st.sidebar.markdown("""
<div style="background:#1a1a1a; border:1px solid #E24B4A; 
            border-radius:8px; padding:10px; 
            text-align:center; margin-bottom:16px;">
    <span style="color:#E24B4A; font-weight:bold; 
                 font-size:0.9rem;">⚡ STRATEGY ENGINE</span><br>
    <span style="color:#888; font-size:0.75rem;">
        XGBoost + 8-Rule Engine
    </span>
</div>
""", unsafe_allow_html=True)

# Section 2 — Race Selection
st.sidebar.markdown("### 🏆 Race Selection")

season = st.sidebar.selectbox(
    "Season", [2024, 2023, 2022], index=0,
    key="season"
)

grand_prix = st.sidebar.selectbox(
    "Grand Prix", get_race_schedule(season),
    key="grand_prix"
)

DRIVERS_2024 = ['VER', 'PER', 'LEC', 'SAI', 'NOR', 'PIA',
                'HAM', 'RUS', 'ALO', 'STR', 'GAS', 'OCO',
                'TSU', 'RIC', 'MAG', 'HUL', 'BOT', 'ZHO',
                'ALB', 'SAR']

driver = st.sidebar.selectbox(
    "Driver", DRIVERS_2024, key="driver"
)

# Section 3 — Lap selector
st.sidebar.markdown("### 📍 Lap")
lap_number = st.sidebar.slider(
    "Current Lap", min_value=1, max_value=70,
    value=30, key="lap_number"
)

# --- ANALYSE BUTTON AND LOADING LOGIC ---
# This MUST be defined before the overrides below to ensure 
# they use the correct loaded session state on the same run.
# --- LOGGING BUTTON ---
# Auto-load session data reactive
with st.spinner(f"Loading {grand_prix} {season}..."):
    session_data = load_race_session(season, grand_prix)

st.session_state['loaded_session_data'] = session_data
    
if session_data is not None:
        try:
            gap_ctx = get_actual_gap_context(session_data['laps'], driver, lap_number)
            if gap_ctx:
                st.session_state['actual_gap_ahead'] = gap_ctx['gap_ahead']
                st.session_state['actual_gap_behind'] = gap_ctx['gap_behind']
                st.session_state['actual_position'] = gap_ctx['position']
                st.session_state['actual_driver_ahead'] = gap_ctx['driver_ahead']
                st.session_state['actual_driver_behind'] = gap_ctx['driver_behind']
                
                # Update override widgets directly
                st.session_state['gap_ahead_override'] = float(gap_ctx['gap_ahead'])
                st.session_state['gap_behind_override'] = float(gap_ctx['gap_behind'])

            # --- NEW: Compute actual lap_time_delta for syncing ---
            drv_laps_preview = session_data['laps'][
                (session_data['laps']['Driver'] == driver) &
                (session_data['laps']['LapTime_s'].notna())
            ].copy()
            
            if not drv_laps_preview.empty:
                cur = drv_laps_preview[drv_laps_preview['LapNumber'] == lap_number]
                if cur.empty:
                    avail = drv_laps_preview['LapNumber'].values
                    nearest_l = avail[np.argmin(np.abs(avail - lap_number))]
                    cur = drv_laps_preview[drv_laps_preview['LapNumber'] == nearest_l]
                
                if not cur.empty:
                    cur_time = float(cur['LapTime_s'].iloc[0])
                    tyre_life = int(cur['TyreLife'].iloc[0]) if pd.notna(cur['TyreLife'].iloc[0]) else 1
                    stint_start_lap = lap_number - tyre_life + 1
                    stint = drv_laps_preview[drv_laps_preview['LapNumber'] >= stint_start_lap]['LapTime_s']
                    best_in_stint = float(stint.min()) if not stint.empty else cur_time
                    actual_delta_val = max(0.0, cur_time - best_in_stint)
                    st.session_state['actual_lap_delta'] = round(actual_delta_val, 3)
                    # Update override widget
                    st.session_state['lap_delta_override'] = round(actual_delta_val, 3)
        except Exception:
            pass

# Default to cached data if Analyse wasn't clicked this run
if session_data is None:
    session_data = st.session_state.get('loaded_session_data')

# Section 4b — Tyre compound override
st.sidebar.markdown("### 🏎 Tyre Compound")

# Determine actual compound from race data
actual_compound = 'MEDIUM'
try:
    if 'race_data_cache' in st.session_state:
        cached = st.session_state.get('loaded_session_data')
        if cached is not None:
            drv_lap = cached['laps'][
                (cached['laps']['Driver'] == st.session_state.get('driver', 'VER')) &
                (cached['laps']['LapNumber'] == st.session_state.get('lap_number', 30))
            ]
            if not drv_lap.empty and pd.notna(drv_lap['Compound'].iloc[0]):
                actual_compound = str(drv_lap['Compound'].iloc[0])
except Exception:
    pass

COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']
default_idx = COMPOUNDS.index(actual_compound) if actual_compound in COMPOUNDS else 1

compound_override = st.sidebar.selectbox(
    "Tyre compound",
    COMPOUNDS,
    index=default_idx,
    key="compound_override",
    help="Pre-filled with actual race compound. Change to simulate a different tyre."
)

# Only show actual compound reference when Real Race tab has loaded data
if st.session_state.get('loaded_session_data') is not None:
    if actual_compound != compound_override:
        st.sidebar.caption(f"📌 Actual compound: **{actual_compound}** → simulating **{compound_override}**")
    else:
        st.sidebar.caption(f"📌 Actual compound: **{actual_compound}**")
else:
    st.sidebar.caption("📌 Load a race in Real Race tab to see actual compound reference")


# Section 4 — Live Event Overrides
st.sidebar.markdown("### ⚡ Live Events")
st.sidebar.caption("Toggle to simulate mid-race scenarios")

safety_car = st.sidebar.toggle(
    "🚗 Safety Car Deployed",
    value=False, key="safety_car"
)
vsc = st.sidebar.toggle(
    "🟡 Virtual Safety Car (VSC)",
    value=False, key="vsc"
)
rain = st.sidebar.toggle(
    "🌧 Sudden Rain",
    value=False, key="rain"
)

if rain:
    st.sidebar.info("Rain active — compound overridden to INTERMEDIATE")

# Section 5 — Competitor gaps override
st.sidebar.markdown("### 🏎 Gaps & Pace Overrides")
st.sidebar.caption("Simulate competitor stops or pace changes")

gap_ahead_override = st.sidebar.number_input(
    "Gap to car ahead (s)",
    min_value=0.0, max_value=60.0,
    step=0.1,
    key="gap_ahead_override"
)
gap_behind_override = st.sidebar.number_input(
    "Gap to car behind (s)",
    min_value=0.0, max_value=60.0,
    step=0.1,
    key="gap_behind_override"
)

st.sidebar.markdown("### 📉 Pace Delta Override")
st.sidebar.caption("Seconds slower than personal best this stint")

# Show actual value from race data if available
actual_delta = st.session_state.get('actual_lap_delta', None)
if actual_delta is not None:
    st.sidebar.markdown(
        f"<div style='background:#1a1a1a;border-left:3px solid #444;"
        f"border-radius:4px;padding:6px 10px;margin-bottom:6px;"
        f"font-size:0.76rem;color:#888;'>"
        f"Actual race value: <span style='color:#E24B4A;'>+{actual_delta:.3f}s</span>"
        f"</div>",
        unsafe_allow_html=True
    )

lap_delta_override = st.sidebar.number_input(
    "Pace loss (s/lap)",
    min_value=0.0, max_value=10.0,
    step=0.1,
    format="%.3f",
    key="lap_delta_override",
    help="Defaulted to your actual delta vs best in this stint."
)

# Actual gap reference note
try:
    pos = st.session_state.get('actual_position', None)
    drv_ahead = st.session_state.get('actual_driver_ahead', None)
    drv_behind = st.session_state.get('actual_driver_behind', None)
    g_ahead = st.session_state.get('actual_gap_ahead', 3.0)
    g_behind = st.session_state.get('actual_gap_behind', 3.0)
    
    if pos is not None:
        ahead_str = f"{drv_ahead} (+{g_ahead:.1f}s)" if drv_ahead else "None (leading)"
        behind_str = f"{drv_behind} (+{g_behind:.1f}s)" if drv_behind else "None (last)"
        current_driver = st.session_state.get('driver', 'VER')
        current_lap = st.session_state.get('lap_number', 30)
        st.sidebar.markdown(f"""
<div style="background:#1a1a1a;
            border-left:3px solid #444;
            border-radius:4px;
            padding:8px 10px;
            margin-top:4px;">
  <div style="color:#666;font-size:0.7rem;
              margin-bottom:4px;">
    ACTUAL RACE — LAP {current_lap}
  </div>
  <div style="color:#aaa;font-size:0.78rem;">
    P{pos} — {current_driver}
  </div>
  <div style="color:#888;font-size:0.75rem;
              margin-top:4px;">
    ▲ Ahead: <span style="color:#E24B4A;">{ahead_str}</span>
  </div>
  <div style="color:#888;font-size:0.75rem;">
    ▼ Behind: <span style="color:#00C851;">{behind_str}</span>
  </div>
</div>""", unsafe_allow_html=True)
except Exception:
    pass


# Section 7 — Export button
if len(st.session_state.scenario_log) > 0:
    log_df = pd.DataFrame(st.session_state.scenario_log)
    csv = log_df.to_csv(index=False)
    st.sidebar.download_button(
        label="⬇ Export Scenario Log (CSV)",
        data=csv,
        file_name=f"f1_strategy_log_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True
    )

# Section 8 — About box
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="color:#555; font-size:0.72rem; 
            text-align:center; padding:8px;">
    Built with FastF1 · XGBoost · Streamlit<br>
    Data: 2022–2024 F1 Seasons<br>
    65,001 training examples
</div>
""", unsafe_allow_html=True)


RACE_TOTAL_LAPS = {
    'Bahrain Grand Prix': 57,
    'Saudi Arabian Grand Prix': 50,
    'Australian Grand Prix': 58,
    'Japanese Grand Prix': 53,
    'Chinese Grand Prix': 56,
    'Miami Grand Prix': 57,
    'Emilia Romagna Grand Prix': 63,
    'Monaco Grand Prix': 78,
    'Canadian Grand Prix': 70,
    'Spanish Grand Prix': 66,
    'Austrian Grand Prix': 71,
    'British Grand Prix': 52,
    'Hungarian Grand Prix': 70,
    'Belgian Grand Prix': 44,
    'Dutch Grand Prix': 72,
    'Italian Grand Prix': 53,
    'Azerbaijan Grand Prix': 51,
    'Singapore Grand Prix': 62,
    'United States Grand Prix': 56,
    'Mexico City Grand Prix': 71,
    'São Paulo Grand Prix': 71,
    'Las Vegas Grand Prix': 50,
    'Qatar Grand Prix': 57,
    'Abu Dhabi Grand Prix': 58
}

# ─── Main Content — Tabs ──────────────────────────────────
tab1, tab2 = st.tabs([
    "🏁  Real Race Replay",
    "🔧  Manual Scenario Builder"
])

with tab1:
    if session_data is None:
        # Default state before any race is loaded
        st.markdown("""
        <div style="text-align:center; padding:60px 20px; color:#555;">
            <div style="font-size:3rem; margin-bottom:16px;">🏁</div>
            <h3 style="color:#888;">Select a race, driver, and lap in the sidebar</h3>
            <p style="color:#555;">
                Load race data to see real-time strategy recommendations.<br>
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Extract driver data at selected lap
        actual_total = session_data['total_laps']
        effective_lap = min(lap_number, actual_total)
        lap_data = get_driver_lap_data(session_data, driver, effective_lap)

        if lap_data is None:
            st.warning(f"Driver **{driver}** not found in {grand_prix} {season}. "
                       f"They may not have participated in this event.")
        else:
            # Build race state with sidebar overrides applied
            race_state = build_race_state_from_overrides(
                tyre_age=lap_data['tyre_age'],
                compound=st.session_state.get('compound_override', lap_data['compound']),
                lap_number=effective_lap,
                total_laps=actual_total,
                position=lap_data['position'],
                deg_rate=lap_data['deg_rate'],
                # Use the override value directly from sidebar
                lap_time_delta=st.session_state.get('lap_delta_override', lap_data['lap_time_delta']),
                track_temp=lap_data['track_temp'],
                fuel_adjusted_laptime=lap_data['fuel_adjusted_laptime']
            )

            # Run strategy engine
            decision = recommend(race_state)
            st.session_state.last_decision = decision
            
            # Auto-log: only log if this exact 
            # lap/driver/race combo hasn't been 
            # logged already this session
            log_key = (
                f"{season}_{grand_prix}_"
                f"{driver}_{lap_number}"
            )
            if log_key not in st.session_state.get(
                'logged_keys', set()
            ):
                log_scenario(
                    race_state, decision, 
                    mode='Real Race'
                )
                if 'logged_keys' not in st.session_state:
                    st.session_state['logged_keys'] = set()
                st.session_state['logged_keys'].add(log_key)

            # Always render panels - Reactivity!
            # ── Panel A: Strategy Recommendation Card ──
            render_strategy_card(decision)

            st.markdown("---")

            # ── Panel B: Race Context Summary ──
            st.markdown("#### 📊 Race Context")
            ctx1, ctx2, ctx3, ctx4 = st.columns(4)
            with ctx1:
                compound_colors = {
                    'SOFT': '🔴', 'MEDIUM': '🟡', 'HARD': '⚪',
                    'INTERMEDIATE': '🟢', 'WET': '🔵'
                }
                comp_icon = compound_colors.get(race_state['compound'], '⚫')
                st.metric("Compound", f"{comp_icon} {race_state['compound']}")
            with ctx2:
                st.metric("Tyre Age", f"{race_state['tyre_age']} laps")
            with ctx3:
                st.metric("Position", f"P{race_state['position']}")
            with ctx4:
                st.metric("Laps Left", f"{race_state['laps_remaining']}")

            ctx5, ctx6, ctx7, ctx8 = st.columns(4)
            with ctx5:
                st.metric("Deg Rate", f"{race_state['deg_rate']:.3f} s/lap")
            with ctx6:
                st.metric("Lap Δ", f"+{race_state['lap_time_delta']:.2f}s")
            with ctx7:
                st.metric("Track Temp", f"{race_state['track_temp']:.1f}°C")
            with ctx8:
                st.metric("Gap Ahead", f"{race_state['gap_ahead']:.1f}s")

            st.markdown("---")

            # ── Panel B2: Position Bar ──
            st.markdown("#### 🏎 Race Positions")
            render_position_bar(session_data['laps'], effective_lap, driver)

            st.markdown("---")

            # ── Panel C: Tyre Simulation ──
            st.markdown("### 📈 Tyre Simulation")
            st.caption("Predicted pace degradation forward from current state. Reactive to all sidebar changes.")
            render_tyre_simulation(
                compound_selected=st.session_state.get('compound_override', lap_data['compound']),
                tyre_age_current=lap_data['tyre_age'],
                base_laptime=lap_data.get('current_laptime', 90.0),
                lap_number=effective_lap,
                total_laps=session_data['total_laps']
            )

            # ── Panel D: Actual Race Lap Times ──
            st.markdown("### 🏁 Actual Race Lap Times")
            render_actual_race_chart(
                driver_laps=lap_data['driver_laps'],
                session_laps=session_data['laps'],
                lap_number=effective_lap,
                session_data=session_data
            )

            # ── Panel E: Scenario log ──
            if len(st.session_state.scenario_log) > 0:
                st.markdown("#### 📋 Scenario Log")
                log_df = pd.DataFrame(st.session_state.scenario_log)
                st.dataframe(log_df, use_container_width=True, height=200)

            # ── Panel F: Actual Race History ──
            st.markdown("### 📜 Actual Race History")
            render_race_history(session_data, driver)

            # ── Timestamp ──
            st.divider()
            st.caption(f"Analysis for **{driver}** at Lap {effective_lap} — "
                       f"Updated at {datetime.now().strftime('%H:%M:%S')}")

with tab2:
    PIT_LOSS = 22.0
    sim_laps = list(range(1, 16))
    
    st.markdown("#### 🏆 Race Reference")
    t2_col1, t2_col2 = st.columns(2)
    with t2_col1:
        t2_season = st.selectbox("Season", [2024, 2023, 2022], index=0, key="t2_season")
    with t2_col2:
        t2_gp = st.selectbox("Grand Prix", list(RACE_TOTAL_LAPS.keys()), index=0, key="t2_gp")

    t2_total_laps = RACE_TOTAL_LAPS.get(t2_gp, 57)
    st.caption(f"📋 {t2_gp} {t2_season} — {t2_total_laps} total laps")
    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### 🏎 Driver & Tyre")
        
        ALL_DRIVERS = ['VER','PER','LEC','SAI','HAM','RUS','NOR','PIA','ALO','STR','GAS','OCO','TSU','RIC','ALB','SAR','MAG','HUL','BOT','ZHO','LAT','MSC','VET','LAW','BEA','ANT','HAD','DOO','BOR','COL']
        manual_driver = st.selectbox("Driver", ALL_DRIVERS, key="manual_driver")
        manual_compound = st.selectbox("Current compound", ['SOFT','MEDIUM','HARD','INTERMEDIATE','WET'], index=1, key="manual_compound")
        manual_tyre_age = st.slider("Tyre age (laps on current set)", min_value=1, max_value=60, value=15, key="manual_tyre_age")
        manual_position = st.slider("Current position", min_value=1, max_value=20, value=5, key="manual_position")

    with col_right:
        st.markdown("#### 📊 Race State")
        manual_current_lap = st.slider("Current race lap", min_value=1, max_value=t2_total_laps, value=min(30, t2_total_laps), key="manual_current_lap")
        manual_laps_remaining = t2_total_laps - manual_current_lap
        st.caption(f"{manual_laps_remaining} laps remaining of {t2_total_laps}")
        
        manual_gap_ahead = st.number_input("Gap to car ahead (s)", min_value=0.0, max_value=60.0, value=3.0, step=0.1, key="manual_gap_ahead")
        manual_gap_behind = st.number_input("Gap to car behind (s)", min_value=0.0, max_value=60.0, value=4.0, step=0.1, key="manual_gap_behind")
        manual_deg_rate = st.number_input(
            "Degradation rate (s/lap)",
            min_value=0.000,
            max_value=0.500,
            value=0.060,
            step=0.001,
            format="%.3f",
            key="manual_deg_rate",
            help="Typical values: SOFT 0.08-0.18, MEDIUM 0.04-0.09, HARD 0.02-0.05"
        )
        manual_lap_time = st.number_input("Current lap time (s)", min_value=60.0, max_value=150.0, value=90.0, step=0.5, key="manual_lap_time")

    st.markdown("#### 🌡 Conditions")
    cond_col1, cond_col2, cond_col3 = st.columns(3)
    with cond_col1:
        manual_track_temp = st.slider("Track temp (°C)", min_value=15, max_value=60, value=35, key="manual_track_temp")
    with cond_col2:
        manual_lap_delta = st.number_input("Lap time delta vs best (s)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, key="manual_lap_delta")
    with cond_col3:
        manual_sc = st.checkbox("Safety Car active", key="manual_sc")
        manual_vsc = st.checkbox("VSC active", key="manual_vsc")
        manual_rain = st.checkbox(
            "🌧 Rain", key="manual_rain",
            help="Forces compound to INTERMEDIATE"
        )

    st.markdown("---")

    # Build manual race_state
    eff_deg = manual_deg_rate * 0.5 if manual_vsc else manual_deg_rate
    eff_compound = (
        'INTERMEDIATE'
        if (manual_rain or st.session_state.get('rain', False))
        else manual_compound
    )

    if manual_rain:
        st.info("🌧 Rain active — compound overridden to INTERMEDIATE")
    laps_remaining_val = manual_laps_remaining

    manual_race_state = {
        'tyre_age': manual_tyre_age,
        'compound': eff_compound,
        'gap_ahead': manual_gap_ahead,
        'gap_behind': manual_gap_behind,
        'laps_remaining': laps_remaining_val,
        'position': manual_position,
        'safety_car_active': 1 if manual_sc else 0,
        'deg_rate': eff_deg,
        'lap_time_delta': manual_lap_delta,
        'track_temp': manual_track_temp,
        'fuel_adjusted_laptime': manual_lap_time - (laps_remaining_val * 0.03)
    }

    manual_decision = recommend(manual_race_state)
    
    # Auto-log Tab 2
    manual_log_key = (
        f"manual_{eff_compound}_"
        f"{manual_tyre_age}_"
        f"{manual_current_lap}_"
        f"{manual_deg_rate:.3f}"
    )
    if manual_log_key not in st.session_state.get(
        'logged_keys', set()
    ):
        log_scenario(
            manual_race_state, manual_decision,
            mode='Manual'
        )
        if 'logged_keys' not in st.session_state:
            st.session_state['logged_keys'] = set()
        st.session_state['logged_keys'].add(
            manual_log_key)

    st.markdown("### 🎯 Strategy Recommendation")
    render_strategy_card(manual_decision)


    st.markdown("### 🔭 Scenario Projections")
    
    # 1. Optimal Pit Window (Status Bar)
    st.markdown("**Optimal pit window**")
    MIN_STINTS = {'SOFT':10,'MEDIUM':15,'HARD':20,'INTERMEDIATE':5,'WET':5}
    CRITICAL_AGES = {'SOFT':22,'MEDIUM':30,'HARD':45,'INTERMEDIATE':35,'WET':30}
    earliest, latest = MIN_STINTS.get(eff_compound, 10), CRITICAL_AGES.get(eff_compound, 30)
    current_race_lap = manual_current_lap
    earliest_race_lap, latest_race_lap = (current_race_lap + max(0, earliest - manual_tyre_age)), (current_race_lap + max(0, latest - manual_tyre_age))
    latest_race_lap = min(latest_race_lap, t2_total_laps - 3)
    window_open, window_critical = manual_tyre_age >= earliest, manual_tyre_age >= latest
    status_color = "#E24B4A" if window_critical else "#FF8800" if window_open else "#00C851"
    status_text = "⚠ PAST OPTIMAL WINDOW" if window_critical else "✅ IN PIT WINDOW" if window_open else "🔒 WINDOW NOT OPEN YET"
    st.markdown(f"<div style='background:#1a1a1a;border:2px solid {status_color};border-radius:8px;padding:12px;text-align:center;font-size:1rem;color:{status_color};font-weight:bold;margin-bottom:12px;'>{status_text}</div>", unsafe_allow_html=True)
    
    # 2. Main Comparison Row (Undercut Chart | Stint Table)
    proj_col_left, proj_col_right = st.columns([3, 2])
    
    with proj_col_left:
        st.markdown("**Undercut Effect (Seconds Gained/Lost)**")
        PIT_LOSS_SIM = 22.0
        FRESH_DEG_STINT = 0.02
        sim_laps_range = list(range(1, 16))

        # Calculations
        stay_times = [manual_lap_time + (l * manual_deg_rate) for l in sim_laps_range]
        stay_cumulative = np.cumsum(stay_times).tolist()

        FRESH_PACE_GAIN = 1.5 
        WARMUP_LAPS = 3
        pit_times = []
        for l in sim_laps_range:
            if l == 1:
                pit_times.append(manual_lap_time + PIT_LOSS_SIM - FRESH_PACE_GAIN)
            else:
                age_in_stint = l - 1
                gain = FRESH_PACE_GAIN * (1 - age_in_stint / WARMUP_LAPS) if age_in_stint <= WARMUP_LAPS else 0.0
                pit_times.append(manual_lap_time - gain + (age_in_stint * FRESH_DEG_STINT))
        pit_cumulative = np.cumsum(pit_times).tolist()
        
        # Delta: Positive = Faster overall by pitting
        deltas = [s - p for s, p in zip(stay_cumulative, pit_cumulative)]
        crossover = next((sim_laps_range[i] for i, d in enumerate(deltas) if d > 0), None)

        fig_under = go.Figure()
        fig_under.add_hline(y=0, line_dash="dash", line_color="#444", line_width=1)
        fig_under.add_trace(go.Scatter(
            x=sim_laps_range, y=deltas, mode='lines+markers', name='Time Gain/Loss',
            line=dict(color='#00C851', width=3), marker=dict(size=6),
            hovertemplate="Lap +%{x}<br>Net Effect: %{y:+.1f}s<extra></extra>"
        ))
        fig_under.add_trace(go.Scatter(
            x=sim_laps_range, y=[max(0, d) for d in deltas],
            mode='lines', line=dict(color='rgba(0,0,0,0)'),
            fill='tozeroy', fillcolor='rgba(0,200,81,0.15)', showlegend=False
        ))
        fig_under.add_trace(go.Scatter(
            x=sim_laps_range, y=[min(0, d) for d in deltas],
            mode='lines', line=dict(color='rgba(0,0,0,0)'),
            fill='tozeroy', fillcolor='rgba(226,75,74,0.15)', showlegend=False
        ))

        if crossover:
            fig_under.add_vline(x=crossover, line_dash="dot", line_color="#00C851", line_width=1.5,
                                annotation_text=f"Break even at lap +{crossover}",
                                annotation_font_color="#00C851", annotation_font_size=10)

        fig_under.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300,
            margin=dict(l=50, r=20, t=20, b=40),
            xaxis=dict(title="Laps from now", gridcolor='#1a1a1a', color='#888888', dtick=2),
            yaxis=dict(title="Seconds Gained (+) / Lost (-)", gridcolor='#1a1a1a', color='#888888'),
            showlegend=False
        )
        st.plotly_chart(fig_under, use_container_width=True)
        
        if crossover:
            st.caption(f"✅ Pit-stop pays back in **{crossover} laps**. Net gain after 15 laps: **{deltas[-1]:.1f}s**")
        else:
            st.caption(f"⚠️ Pit-stop cost (~22s) is never recovered within 15 laps.")

    with proj_col_right:
        st.markdown("**Lap-by-Lap Verification (s)**")
        comparison_data = []
        for i in range(1, 11):
            s_lap = stay_times[i-1] - (manual_lap_time + (i-1)*manual_deg_rate) + manual_lap_time + (i*manual_deg_rate)
            # Use actual lap times from sim logic
            s_lt = stay_times[i-1] if i == 1 else stay_times[i-1] - stay_times[i-2] # This is wrong, stay_times IS lap times
            s_lt = stay_times[i-1]
            p_lt = pit_times[0] - PIT_LOSS_SIM if i == 1 else pit_times[i-1] - pit_times[i-2] if i > 1 else 0 # pit_times is CUMULATIVE in my logic? No, check loop above
            # Wait, in the loop above:
            # pit_times.append(...) # This appends LAP TIMES.
            # Then pit_cumulative = np.cumsum(pit_times).tolist()
            # OK, so pit_times are LAP TIMES.
            s_lt = stay_times[i-1]
            p_lt = pit_times[i-1] if i > 1 else pit_times[i-1] - PIT_LOSS_SIM
            
            comparison_data.append({
                "Lap": f"+{i}",
                "Stay": f"{s_lt:.2f}",
                "Pit": f"{p_lt:.2f}",
                "Diff": f"{s_lt - p_lt:+.2f}"
            })
        
        st.dataframe(comparison_data, use_container_width=True, hide_index=True)
        
        if crossover and crossover <= 3:
            st.success(f"🎯 **PIT NOW (Lap {manual_current_lap})**")
        elif crossover:
            st.info(f"💡 **PIT AT +{crossover-2} (Lap {manual_current_lap + crossover - 2})**")
        else:
            st.warning("🔭 **STAY OUT LONGER**")

    # 3. Final Comparison — Compound Recommendation Donut
    st.markdown("---")
    comp_col_left, comp_col_right = st.columns([3, 2])
    with comp_col_left:
        st.markdown("**Compound recommendation — % advantage over staying out**")
        PIT_LOSS_COMP = 22.0

        # Include INTER + WET if rain is active
        rain_on = manual_rain or st.session_state.get('rain', False)
        COMP_ANALYSIS = {
            'SOFT':   {'enc': 2, 'color': '#E8002D',
                       'text_color': '#ffffff',
                       'warmup_laps': 2, 'peak_gain': 1.8},
            'MEDIUM': {'enc': 1, 'color': '#FFF200',
                       'text_color': '#1a1a1a',
                       'warmup_laps': 3, 'peak_gain': 1.0},
            'HARD':   {'enc': 0, 'color': '#CCCCCC',
                       'text_color': '#1a1a1a',
                       'warmup_laps': 5, 'peak_gain': 0.3},
        }
        if rain_on:
            COMP_ANALYSIS['INTERMEDIATE'] = {
                'enc': 3, 'color': '#43B02A',
                'text_color': '#ffffff',
                'warmup_laps': 2, 'peak_gain': 2.5
            }
            COMP_ANALYSIS['WET'] = {
                'enc': 4, 'color': '#0067FF',
                'text_color': '#ffffff',
                'warmup_laps': 2, 'peak_gain': 3.0
            }

        stint_details = {}
        for cname, cfg in COMP_ANALYSIS.items():
            total = PIT_LOSS_COMP
            for l in range(1, manual_laps_remaining + 1):
                gain = cfg['peak_gain'] * (
                    1 - l / cfg['warmup_laps']
                ) if l <= cfg['warmup_laps'] else 0.0
                deg = float(predict_lap_delta(cfg['enc'], l))
                lt = manual_lap_time - gain + deg
                total += lt
            stint_details[cname] = {
                'total': total,
                'color': cfg['color'],
                'text_color': cfg['text_color']
            }

        stay_total_comp = (
            manual_laps_remaining * manual_lap_time
            + manual_deg_rate * (
                manual_laps_remaining *
                (manual_laps_remaining + 1) / 2
            )
        )
        stint_details['STAY OUT'] = {
            'total': stay_total_comp,
            'color': '#555555',
            'text_color': '#ffffff'
        }

        sorted_options = sorted(
            stint_details.items(),
            key=lambda x: x[1]['total']
        )
        best_option = sorted_options[0][0]
        best_total = sorted_options[0][1]['total']

        # Compute % recommendation for pit options only
        pit_options = {
            k: v for k, v in stint_details.items()
            if k != 'STAY OUT'
        }
        savings_vs_stay = {
            k: stay_total_comp - v['total']
            for k, v in pit_options.items()
        }
        pct_savings = {
            k: round((s / stay_total_comp) * 100, 2)
            for k, s in savings_vs_stay.items()
        }
        min_pct = min(pct_savings.values())
        shifted = {
            k: v - min_pct + 0.5
            for k, v in pct_savings.items()
        }
        total_shifted = sum(shifted.values())
        rec_pct = {
            k: round(v / total_shifted * 100, 1)
            for k, v in shifted.items()
        }

        labels = list(rec_pct.keys())
        values = list(rec_pct.values())
        slice_colors = [
            pit_options[k]['color'] for k in labels]
        text_colors = [
            pit_options[k]['text_color'] for k in labels]

        fig_donut = go.Figure(go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker=dict(
                colors=slice_colors,
                line=dict(color='#1a1a1a', width=2)
            ),
            textinfo='label+percent',
            textfont=dict(size=12),
            insidetextfont=dict(
                color='black', size=12),
            outsidetextfont=dict(
                color='white', size=11),
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Recommendation strength: %{percent}<br>"
                "<extra></extra>"
            )
        ))
        fig_donut.add_annotation(
            text=f"<b>{best_option}</b><br>recommended",
            x=0.5, y=0.5,
            font=dict(size=13, color='white'),
            showarrow=False
        )
        fig_donut.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            height=280,
            margin=dict(l=10, r=10, t=20, b=10),
            showlegend=True,
            legend=dict(
                bgcolor='rgba(0,0,0,0)',
                font=dict(color='#888888', size=11),
                orientation='v',
                x=1.0, y=0.5
            )
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        st.caption(
            "Slice size = relative time advantage of each compound "
            "vs staying out. Larger slice = stronger recommendation."
        )

    with comp_col_right:
        st.markdown("**Stint Verdict**")
        pit_sorted = [
            (k, v) for k, v in sorted_options
            if k != 'STAY OUT'
        ]
        best_pit = pit_sorted[0] if pit_sorted else None
        saving = (
            stay_total_comp - best_pit[1]['total']
            if best_pit else 0
        )

        # Summary text (like before)
        st.write(
            f"The fastest strategy for the remaining "
            f"**{manual_laps_remaining} laps** is: **{best_option}**."
        )
        if best_option != 'STAY OUT' and saving > 0:
            st.info(
                f"Pitting for fresh **{best_option}** tyres saves "
                f"**{saving:.1f}s** over staying out, including "
                f"the 22s pit loss."
            )
        elif best_option != 'STAY OUT' and saving <= 0:
            st.warning(
                f"Pitting for **{best_option}** costs "
                f"**{abs(saving):.1f}s** more than staying out."
            )
        else:
            st.warning(
                "Staying out is faster than pitting for "
                "any compound right now."
            )

        # Per-compound breakdown
        st.markdown("**Time vs stay out:**")
        for k, v in pit_sorted:
            diff = stay_total_comp - v['total']
            arrow = "🟢" if diff > 0 else "🔴"
            st.markdown(
                f"{arrow} **{k}**: "
                f"{'saves' if diff > 0 else 'costs'} "
                f"**{abs(diff):.1f}s**"
            )

        st.caption(
            "⚠️ Mathematical estimate — assumes linear pace "
            "loss and 22s pit stop cost."
        )

    # --- END TAB 2 CONTENT ---

    st.markdown("### 📋 Scenario History Log")
    if len(st.session_state.scenario_log) > 0:
        log_df = pd.DataFrame(st.session_state.scenario_log)
        st.dataframe(log_df, use_container_width=True, height=200)
    else:
        st.caption("No scenarios logged yet.")
