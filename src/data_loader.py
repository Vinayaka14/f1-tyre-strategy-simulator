import os
import logging
import pandas as pd
import fastf1

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base directory relative to this script so it works from anywhere
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, 'data', 'raw')

SEASONS = [2022, 2023, 2024]

def setup_cache():
    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)

def fetch_season_data(year):
    logger.info(f"Fetching data for season {year}")
    
    try:
        schedule = fastf1.get_event_schedule(year)
    except Exception as e:
        logger.error(f"Failed to fetch schedule for {year}: {e}")
        return

    season_laps_list = []

    for _, event in schedule.iterrows():
        # Only process races (EventFormat usually indicates if there's a race, but fastf1 'EventFormat' might just be 'testing' vs 'conventional').
        if event['EventFormat'] == 'testing':
            continue

        try:
            logger.info(f"Fetching session for {year} {event['EventName']}")
            session = fastf1.get_session(year, event['EventName'], 'R')
            session.load(telemetry=False, weather=True)
            
            # Need to get laps dataframe
            if not hasattr(session, 'laps') or session.laps is None or session.laps.empty:
                logger.warning(f"No laps data for {year} {event['EventName']}")
                continue
            
            laps = session.laps.copy()
            
            # Get track temp from weather data mapped to laps
            # get_weather_data() returns a DataFrame of the same length
            try:
                weather_data = laps.get_weather_data().reset_index(drop=True)
                laps = laps.reset_index(drop=True)
                
                if 'TrackTemp' in weather_data.columns:
                    laps['TrackTemp'] = weather_data['TrackTemp'].values
                else:
                    laps['TrackTemp'] = pd.NA
            except Exception as e:
                logger.warning(f"Could not load weather data for {event['EventName']}: {e}")
                laps['TrackTemp'] = pd.NA

            # Extract basic timing and pit stop boolean
            if 'LapTime' in laps.columns:
                laps['LapTime_s'] = laps['LapTime'].dt.total_seconds()
            else:
                laps['LapTime_s'] = pd.NA
                
            if 'PitInTime' in laps.columns and 'PitOutTime' in laps.columns:
                laps['PitStop'] = ~(laps['PitInTime'].isna() & laps['PitOutTime'].isna())
            elif 'PitInTime' in laps.columns:
                laps['PitStop'] = ~laps['PitInTime'].isna()
            elif 'PitOutTime' in laps.columns:
                laps['PitStop'] = ~laps['PitOutTime'].isna()
            else:
                laps['PitStop'] = False
            
            # Calculate positions and gaps by lap
            if 'LapNumber' in laps.columns and 'Time' in laps.columns:
                laps = laps.sort_values(by=['LapNumber', 'Time']).reset_index(drop=True)
                laps['Position'] = laps.groupby('LapNumber').cumcount() + 1
                
                laps['GapToAhead'] = laps.groupby('LapNumber')['Time'].diff().dt.total_seconds()
                laps['GapToBehind'] = laps.groupby('LapNumber')['Time'].diff(-1).dt.total_seconds().abs()
            else:
                laps['Position'] = pd.NA
                laps['GapToAhead'] = pd.NA
                laps['GapToBehind'] = pd.NA
            
            # Extract requested columns, ensure missing columns are handled safely
            cols_to_keep = [
                'DriverNumber', 'Driver', 'LapNumber', 'Compound', 'TyreLife', 
                'LapTime_s', 'Position', 'GapToAhead', 'GapToBehind', 'TrackTemp', 'PitStop'
            ]
            for col in cols_to_keep:
                if col not in laps.columns:
                    laps[col] = pd.NA
            
            event_laps = laps[cols_to_keep].copy()
            event_laps['Year'] = year
            event_laps['EventName'] = event['EventName']
            
            season_laps_list.append(event_laps)
            
        except Exception as e:
            logger.error(f"Failed to process session for {year} {event['EventName']}: {e}")
            continue
            
    if season_laps_list:
        season_df = pd.concat(season_laps_list, ignore_index=True)
        out_path = os.path.join(CACHE_DIR, f"laps_{year}.csv")
        season_df.to_csv(out_path, index=False)
        logger.info(f"Saved {year} season data to {out_path}")
    else:
        logger.warning(f"No data to save for season {year}")

def main():
    setup_cache()
    for year in SEASONS:
        fetch_season_data(year)

if __name__ == "__main__":
    main()
