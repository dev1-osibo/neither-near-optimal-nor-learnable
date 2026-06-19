"""
Fetch historical weather data from Open-Meteo API.
Free, no API key required, unlimited requests.
https://open-meteo.com/

Fetches for 3 US data center locations:
- Ashburn, VA (data center capital of the world)
- Phoenix, AZ (hot climate, major DC expansion)
- The Dalles, OR (Google/Meta DCs, mild climate)
"""
import requests
import pandas as pd
import os
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

LOCATIONS = {
    'ashburn_va': {'lat': 39.04, 'lon': -77.49, 'name': 'Ashburn, VA (DC Capital)'},
    'phoenix_az': {'lat': 33.45, 'lon': -112.07, 'name': 'Phoenix, AZ (Hot Climate)'},
    'the_dalles_or': {'lat': 45.60, 'lon': -121.18, 'name': 'The Dalles, OR (Pacific NW)'},
}

# Weather variables relevant to DC energy consumption
HOURLY_VARIABLES = [
    'temperature_2m',
    'relative_humidity_2m',
    'dewpoint_2m',
    'surface_pressure',
    'cloud_cover',
    'wind_speed_10m',
    'direct_radiation',         # Solar irradiance (for on-site PV)
    'diffuse_radiation',
    'shortwave_radiation',      # Total solar
]


def fetch_historical_weather(location_key, start_date='2024-01-01', end_date='2025-12-31'):
    """Fetch historical hourly weather data from Open-Meteo."""
    loc = LOCATIONS[location_key]
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        'latitude': loc['lat'],
        'longitude': loc['lon'],
        'start_date': start_date,
        'end_date': end_date,
        'hourly': ','.join(HOURLY_VARIABLES),
        'timezone': 'America/New_York' if 'va' in location_key else ('America/Phoenix' if 'az' in location_key else 'America/Los_Angeles')
    }
    
    print(f"Fetching weather for {loc['name']} ({start_date} to {end_date})...")
    response = requests.get(url, params=params, timeout=60)
    
    if response.status_code != 200:
        print(f"  ERROR: {response.status_code} — {response.text[:200]}")
        return None
    
    data = response.json()
    
    if 'hourly' not in data:
        print(f"  ERROR: No hourly data in response")
        return None
    
    # Convert to DataFrame
    hourly = data['hourly']
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(hourly['time']),
        **{var: hourly[var] for var in HOURLY_VARIABLES if var in hourly}
    })
    
    df['location'] = location_key
    
    # Save
    filename = f"weather_{location_key}_{start_date}_{end_date}.csv"
    filepath = os.path.join(DATA_DIR, filename)
    df.to_csv(filepath, index=False)
    
    print(f"  Saved: {filepath}")
    print(f"  Rows: {len(df):,} | Columns: {len(df.columns)}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Temp range: {df['temperature_2m'].min():.1f}°C to {df['temperature_2m'].max():.1f}°C")
    
    return df


def fetch_forecast_weather(location_key, days_ahead=16):
    """Fetch weather forecast from Open-Meteo (up to 16 days ahead)."""
    loc = LOCATIONS[location_key]
    
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        'latitude': loc['lat'],
        'longitude': loc['lon'],
        'hourly': ','.join(HOURLY_VARIABLES),
        'forecast_days': days_ahead,
        'timezone': 'auto'
    }
    
    print(f"Fetching {days_ahead}-day forecast for {loc['name']}...")
    response = requests.get(url, params=params, timeout=30)
    
    if response.status_code != 200:
        print(f"  ERROR: {response.status_code}")
        return None
    
    data = response.json()
    hourly = data['hourly']
    
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(hourly['time']),
        **{var: hourly[var] for var in HOURLY_VARIABLES if var in hourly}
    })
    df['location'] = location_key
    
    filename = f"weather_forecast_{location_key}_{datetime.now().strftime('%Y%m%d')}.csv"
    filepath = os.path.join(DATA_DIR, filename)
    df.to_csv(filepath, index=False)
    
    print(f"  Saved: {filepath} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("WEATHER DATA COLLECTION — Open-Meteo API (Free)")
    print("=" * 60)
    
    # Fetch 2 years of historical data for each location
    all_data = {}
    for loc_key in LOCATIONS:
        df = fetch_historical_weather(loc_key, '2024-01-01', '2025-12-31')
        if df is not None:
            all_data[loc_key] = df
        print()
    
    # Also fetch current forecasts
    print("\n--- FORECASTS ---")
    for loc_key in LOCATIONS:
        fetch_forecast_weather(loc_key, 16)
        print()
    
    # Summary
    print("\n" + "=" * 60)
    print("DATA COLLECTION SUMMARY")
    print("=" * 60)
    total_rows = sum(len(df) for df in all_data.values())
    print(f"Total historical rows: {total_rows:,}")
    print(f"Locations: {len(all_data)}")
    print(f"Variables per location: {len(HOURLY_VARIABLES)}")
    print(f"Files saved to: {DATA_DIR}")
