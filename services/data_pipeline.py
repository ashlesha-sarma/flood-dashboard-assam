"""
services/data_pipeline.py
==========================
Generates realistic synthetic training data for Assam flood prediction.
Uses IMD-style rainfall patterns, CWC river thresholds, and ASDMA flood
statistics to produce a labeled dataset with realistic seasonal structure.

All synthetic data is calibrated to match known Assam flood patterns:
- Flood season: June–September (monsoon)
- Major rivers: Brahmaputra, Subansiri, Kopili, Manas, Jiabharali
- Districts mapped to primary rivers
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

# River thresholds (CWC values, metres above sea level)
RIVER_THRESHOLDS = {
    'brahmaputra': {'warning': 97.32, 'danger': 98.32, 'min': 92.0},
    'subansiri':   {'warning': 80.50, 'danger': 82.00, 'min': 75.0},
    'kopili':      {'warning': 40.00, 'danger': 42.00, 'min': 35.0},
    'manas':       {'warning': 36.00, 'danger': 37.50, 'min': 31.0},
    'jiabharali':  {'warning': 63.00, 'danger': 65.00, 'min': 58.0},
    'buridehing':  {'warning': 109.50,'danger': 111.00,'min': 103.0},
    'beki':        {'warning': 32.00, 'danger': 34.00, 'min': 27.0},
    'dhansiri':    {'warning': 70.00, 'danger': 72.00, 'min': 64.0},
}

DISTRICT_RIVER_MAP = {
    'Kamrup Metro':    'brahmaputra', 'Kamrup':          'brahmaputra',
    'Morigaon':        'brahmaputra', 'Nagaon':          'kopili',
    'Golaghat':        'dhansiri',    'Jorhat':          'brahmaputra',
    'Majuli':          'brahmaputra', 'Sivasagar':       'brahmaputra',
    'Dibrugarh':       'buridehing',  'Tinsukia':        'brahmaputra',
    'Lakhimpur':       'subansiri',   'Dhemaji':         'subansiri',
    'Sonitpur':        'jiabharali',  'Biswanath':       'jiabharali',
    'Darrang':         'brahmaputra', 'Udalguri':        'brahmaputra',
    'Barpeta':         'manas',       'Nalbari':         'beki',
    'Chirang':         'manas',       'Bongaigaon':      'manas',
    'Kokrajhar':       'brahmaputra', 'Dhubri':          'brahmaputra',
    'Goalpara':        'brahmaputra', 'South Salmara':   'brahmaputra',
    'Cachar':          'brahmaputra', 'Hailakandi':      'brahmaputra',
    'Karimganj':       'brahmaputra', 'Dima Hasao':      'kopili',
    'Karbi Anglong':   'kopili',      'West Karbi':      'kopili',
    'Charaideo':       'buridehing',  'Hojai':           'kopili',
    'Bajali':          'beki',
}

ALL_DISTRICTS = list(DISTRICT_RIVER_MAP.keys())


def _monsoon_factor(day_of_year: int) -> float:
    """Return 0-1 rainfall intensity factor based on day of year (monsoon peaks Jul-Aug)."""
    # Peak around day 200 (mid-July), sharp rise Jun, fall Sep
    peak = 200
    width = 50
    factor = np.exp(-0.5 * ((day_of_year - peak) / width) ** 2)
    return max(0.0, factor)


def generate_river_levels(n_days: int = 3650) -> pd.DataFrame:
    """Generate 10 years of daily river level data with realistic seasonality."""
    start = datetime(2014, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    records = []

    for river, thr in RIVER_THRESHOLDS.items():
        base = (thr['warning'] + thr['min']) / 2
        span = thr['danger'] - thr['min']
        level = base

        for i, date in enumerate(dates):
            doy = date.timetuple().tm_yday
            mf = _monsoon_factor(doy)

            # Seasonal trend + random walk + occasional flood spikes
            seasonal = mf * span * 0.6
            noise = np.random.normal(0, span * 0.03)
            mean_revert = (base + seasonal - level) * 0.15
            spike = 0.0
            if mf > 0.3 and np.random.random() < 0.04:  # 4% chance of flood spike in monsoon
                spike = np.random.uniform(span * 0.3, span * 0.7)

            level = level + mean_revert + noise + spike
            level = np.clip(level, thr['min'], thr['danger'] * 1.1)

            records.append({
                'date': date.strftime('%Y-%m-%d'),
                'river': river,
                'level': round(level, 3),
                'warning': thr['warning'],
                'danger': thr['danger'],
                'min': thr['min'],
            })

    return pd.DataFrame(records)


def generate_rainfall(n_days: int = 3650) -> pd.DataFrame:
    """Generate upstream rainfall data (mm/day) for each district."""
    start = datetime(2014, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    records = []

    for district, river in DISTRICT_RIVER_MAP.items():
        rain = 2.0
        for date in dates:
            doy = date.timetuple().tm_yday
            mf = _monsoon_factor(doy)
            seasonal_mean = mf * 25.0 + 1.0
            rain = max(0, rain * 0.7 + np.random.exponential(seasonal_mean) * 0.3)

            records.append({
                'date': date.strftime('%Y-%m-%d'),
                'district': district,
                'river': river,
                'rainfall_mm': round(rain, 2),
            })

    return pd.DataFrame(records)


def build_training_dataset(river_df: pd.DataFrame, rain_df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features and labels for flood risk classification.

    Features per district-day:
    - river_level, pct_to_danger (0-1 scale)
    - 3-day rolling rainfall, 7-day rolling rainfall
    - river level delta (change from previous day)
    - 3-day river level trend (slope)
    - is_monsoon (binary flag)
    - upstream_rain_3d (proxy for upstream catchment)

    Label: 0=low, 1=moderate, 2=high
    """
    records = []

    for district, river in DISTRICT_RIVER_MAP.items():
        thr = RIVER_THRESHOLDS[river]
        span = thr['danger'] - thr['min']

        r_df = river_df[river_df['river'] == river].copy().sort_values('date').reset_index(drop=True)
        d_df = rain_df[rain_df['district'] == district].copy().sort_values('date').reset_index(drop=True)

        if len(r_df) < 14 or len(d_df) < 14:
            continue

        levels = r_df['level'].values
        dates  = r_df['date'].values
        rains  = d_df['rainfall_mm'].values[:len(levels)]

        for i in range(7, len(levels)):
            level = levels[i]
            pct   = (level - thr['min']) / span
            delta = levels[i] - levels[i-1]
            trend = (levels[i] - levels[i-3]) / 3 if i >= 3 else 0.0

            rain_3d  = float(np.mean(rains[max(0, i-3):i]))
            rain_7d  = float(np.mean(rains[max(0, i-7):i]))
            rain_1d  = float(rains[i])

            # Upstream proxy: yesterday's upstream rain drives today's level
            upstream = float(np.mean(rains[max(0, i-2):i-1])) if i >= 2 else rain_1d

            date_obj = datetime.strptime(str(dates[i]), '%Y-%m-%d')
            doy = date_obj.timetuple().tm_yday
            is_monsoon = int(152 <= doy <= 273)  # Jun 1 – Sep 30

            # Label: 0=low, 1=moderate, 2=high
            if level >= thr['danger']:
                label = 2
            elif level >= thr['warning']:
                label = 1
            elif level >= thr['warning'] * 0.96 or rain_3d > 20:
                label = 1
            else:
                label = 0

            records.append({
                'date':         dates[i],
                'district':     district,
                'river':        river,
                'river_level':  round(level, 3),
                'pct_to_danger': round(pct, 4),
                'level_delta':  round(delta, 4),
                'level_trend':  round(trend, 4),
                'rain_1d':      round(rain_1d, 2),
                'rain_3d':      round(rain_3d, 2),
                'rain_7d':      round(rain_7d, 2),
                'upstream_3d':  round(upstream, 2),
                'is_monsoon':   is_monsoon,
                'warning_level': thr['warning'],
                'danger_level':  thr['danger'],
                'label':        label,
            })

    df = pd.DataFrame(records)
    return df.sort_values(['date', 'district']).reset_index(drop=True)


def generate_crop_damage_data(training_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate crop damage labels (hectares at risk) for the Random Forest regressor.
    Based on historical ASDMA FRIMS: ~800k-2.5M ha affected annually in bad years.
    """
    records = []
    district_crop_area = {d: np.random.uniform(30000, 150000) for d in ALL_DISTRICTS}

    for _, row in training_df.iterrows():
        if row['label'] == 0:
            continue
        base_area = district_crop_area.get(row['district'], 60000)
        flood_pct = row['pct_to_danger']
        rain_factor = (row['rain_7d'] / 20.0) if row['rain_7d'] > 0 else 0
        damage_pct = min(1.0, flood_pct * 0.5 + rain_factor * 0.3 + np.random.uniform(0, 0.2))
        crop_ha = base_area * damage_pct * (1 if row['label'] == 2 else 0.4)

        records.append({
            'district':     row['district'],
            'river_level':  row['river_level'],
            'pct_to_danger': row['pct_to_danger'],
            'rain_3d':      row['rain_3d'],
            'rain_7d':      row['rain_7d'],
            'level_delta':  row['level_delta'],
            'is_monsoon':   row['is_monsoon'],
            'crop_ha':      round(crop_ha, 0),
        })

    return pd.DataFrame(records)


def get_all_data():
    """Generate and return all datasets. Cached after first call."""
    river_df = generate_river_levels(3650)
    rain_df  = generate_rainfall(3650)
    train_df = build_training_dataset(river_df, rain_df)
    crop_df  = generate_crop_damage_data(train_df)
    return river_df, rain_df, train_df, crop_df
