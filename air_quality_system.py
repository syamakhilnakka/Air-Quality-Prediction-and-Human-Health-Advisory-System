"""

        AIR QUALITY PREDICTION & HEALTH ADVISORY SYSTEM
        ML-Based AQI Estimation with Real-Time Health Guidance


Pollutants Tracked : PM2.5 · PM10 · CO · NO2 · SO2 · O3
ML Models          : Random Forest + RandomForest (US EPA AQI standard)
Output             : AQI value, category, dominant pollutant, health advisory
"""

import numpy as np
import pandas as pd
import joblib
import json
from dataclasses import dataclass
from typing import Optional

# ─── AQI Breakpoints (US EPA Standard) ────────────────────────────────────────

PM25_BP = [(0,12.0,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
           (55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,350.4,301,400),(350.5,500.4,401,500)]
PM10_BP = [(0,54,0,50),(55,154,51,100),(155,254,101,150),
           (255,354,151,200),(355,424,201,300),(425,504,301,400),(505,604,401,500)]
CO_BP   = [(0,4.4,0,50),(4.5,9.4,51,100),(9.5,12.4,101,150),
           (12.5,15.4,151,200),(15.5,30.4,201,300),(30.5,40.4,301,400),(40.5,50.4,401,500)]
NO2_BP  = [(0,53,0,50),(54,100,51,100),(101,360,101,150),
           (361,649,151,200),(650,1249,201,300),(1250,1649,301,400),(1650,2049,401,500)]
SO2_BP  = [(0,35,0,50),(36,75,51,100),(76,185,101,150),
           (186,304,151,200),(305,604,201,300),(605,804,301,400),(805,1004,401,500)]
O3_BP   = [(0,54,0,50),(55,70,51,100),(71,85,101,150),
           (86,105,151,200),(106,200,201,300),(201,300,301,400),(301,400,401,500)]

AQI_CATEGORIES = {
    (0,   50):  ("Good",             "#00e400", "🟢"),
    (51,  100): ("Moderate",          "#ffff00", "🟡"),
    (101, 150): ("Unhealthy for Sensitive Groups", "#ff7e00", "🟠"),
    (151, 200): ("Unhealthy",         "#ff0000", "🔴"),
    (201, 300): ("Very Unhealthy",    "#8f3f97", "🟣"),
    (301, 500): ("Hazardous",         "#7e0023", "⚫"),
}

HEALTH_ADVISORIES = {
    "Good": {
        "summary": "Air quality is satisfactory with little or no risk.",
        "general_public": "Enjoy outdoor activities freely.",
        "sensitive_groups": "No restrictions needed.",
        "outdoor_activity": "All activities recommended.",
        "mask_advice": "No mask required.",
        "window_advice": "Open windows for fresh air.",
        "tips": [
            "Great day for outdoor exercise.",
            "Ideal conditions for all age groups.",
            "Air quality poses no health concern.",
        ]
    },
    "Moderate": {
        "summary": "Air quality is acceptable; some pollutants may concern very sensitive individuals.",
        "general_public": "Unusually sensitive people should consider reducing prolonged outdoor exertion.",
        "sensitive_groups": "People with respiratory issues should monitor symptoms.",
        "outdoor_activity": "Light to moderate activities are fine.",
        "mask_advice": "Mask optional for sensitive individuals.",
        "window_advice": "Open windows with caution.",
        "tips": [
            "Limit prolonged strenuous outdoor activity if sensitive.",
            "Keep asthma inhalers handy.",
            "Check air quality before exercising outdoors.",
        ]
    },
    "Unhealthy for Sensitive Groups": {
        "summary": "Sensitive groups may experience health effects; general public is less likely to be affected.",
        "general_public": "Consider reducing prolonged outdoor exertion.",
        "sensitive_groups": "Children, elderly, and those with heart/lung disease should reduce outdoor activity.",
        "outdoor_activity": "Short walks are okay; avoid strenuous exercise.",
        "mask_advice": "N95/KN95 mask recommended for sensitive groups outdoors.",
        "window_advice": "Keep windows closed if sensitive.",
        "tips": [
            "Reschedule outdoor sports to early morning or evening.",
            "Use air purifiers indoors.",
            "Keep medications accessible.",
            "Avoid areas with heavy traffic.",
        ]
    },
    "Unhealthy": {
        "summary": "Everyone may begin to experience health effects; sensitive groups more serious effects.",
        "general_public": "Avoid prolonged or heavy outdoor exertion.",
        "sensitive_groups": "Remain indoors and keep activity levels low.",
        "outdoor_activity": "Avoid outdoor activities; exercise indoors.",
        "mask_advice": "Wear N95/KN95 mask outdoors.",
        "window_advice": "Keep windows closed; use air purifier.",
        "tips": [
            "Limit time spent outside.",
            "Run air purifiers on high.",
            "Wear a well-fitting N95 or KN95 mask if going out.",
            "Watch for symptoms like coughing or shortness of breath.",
            "Hydrate well and stay indoors as much as possible.",
        ]
    },
    "Very Unhealthy": {
        "summary": "Health alert: everyone may experience serious health effects.",
        "general_public": "Avoid all outdoor exertion.",
        "sensitive_groups": "Remain indoors; seek medical attention if experiencing symptoms.",
        "outdoor_activity": "Stay indoors entirely if possible.",
        "mask_advice": "Full respirator or N95 mandatory if outdoors.",
        "window_advice": "Seal windows and doors; HEPA purifier running continuously.",
        "tips": [
            "Stay indoors with windows and doors closed.",
            "Use a HEPA air purifier.",
            "Vulnerable groups should seek medical attention.",
            "Avoid exercise even indoors if ventilation is poor.",
            "Emergency contacts should be on standby for at-risk individuals.",
        ]
    },
    "Hazardous": {
        "summary": "EMERGENCY CONDITIONS: Serious risk for the entire population.",
        "general_public": "Avoid all outdoor activity. Stay indoors.",
        "sensitive_groups": "Evacuate to clean air if possible; seek immediate medical care for symptoms.",
        "outdoor_activity": "Do NOT go outside under any circumstances.",
        "mask_advice": "P100 respirator required if evacuation is necessary.",
        "window_advice": "Seal all openings; multiple HEPA purifiers running.",
        "tips": [
            "EMERGENCY — Do not go outside.",
            "Seal all windows, doors, and vents.",
            "Contact emergency services if experiencing difficulty breathing.",
            "Vulnerable individuals may need emergency evacuation.",
            "Follow civil authority instructions immediately.",
        ]
    }
}

# ─── Core Functions ────────────────────────────────────────────────────────────

def calc_aqi_sub(C: float, breakpoints: list) -> int:
    """Calculate sub-index AQI for a single pollutant."""
    for (Clo, Chi, Ilo, Ihi) in breakpoints:
        if Clo <= C <= Chi:
            return round(((Ihi - Ilo) / (Chi - Clo)) * (C - Clo) + Ilo)
    return 500  # Beyond scale → Hazardous


def calculate_aqi(pm25, pm10, co, no2, so2, o3):
    """
    Calculate overall AQI and identify dominant pollutant.
    Returns (aqi: int, dominant_pollutant: str, sub_indices: dict)
    """
    sub = {
        'PM2.5': calc_aqi_sub(pm25, PM25_BP),
        'PM10':  calc_aqi_sub(pm10, PM10_BP),
        'CO':    calc_aqi_sub(co,   CO_BP),
        'NO2':   calc_aqi_sub(no2,  NO2_BP),
        'SO2':   calc_aqi_sub(so2,  SO2_BP),
        'O3':    calc_aqi_sub(o3,   O3_BP),
    }
    aqi = max(sub.values())
    dominant = max(sub, key=sub.get)
    return aqi, dominant, sub


def get_category(aqi: int):
    """Return (category_name, hex_color, emoji) for an AQI value."""
    for (lo, hi), info in AQI_CATEGORIES.items():
        if lo <= aqi <= hi:
            return info
    return ("Hazardous", "#7e0023", "⚫")


def get_health_advisory(category: str) -> dict:
    """Return the full health advisory dictionary for an AQI category."""
    return HEALTH_ADVISORIES.get(category, HEALTH_ADVISORIES["Hazardous"])


# ─── ML Prediction Class ───────────────────────────────────────────────────────

class AQIPredictor:
    """Wraps trained ML model for AQI prediction."""

    def __init__(self, model_path='aqi_model_rf.pkl', scaler_path='aqi_scaler_rf.pkl'):
        self.model  = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict(self, pm25, pm10, co, no2, so2, o3, hour=12, day_of_year=180) -> float:
        X = np.array([[pm25, pm10, co, no2, so2, o3, hour, day_of_year]])
        X_scaled = self.scaler.transform(X)
        return float(self.model.predict(X_scaled)[0])


# ─── Full Analysis Function ────────────────────────────────────────────────────

def analyze_air_quality(pm25, pm10, co, no2, so2, o3,
                        hour=12, day_of_year=180,
                        predictor: Optional[AQIPredictor] = None,
                        location: str = "Unknown Location") -> dict:
    """
    Full air quality analysis pipeline.

    Parameters
    ----------
    pm25       : PM2.5 concentration (μg/m³)
    pm10       : PM10 concentration (μg/m³)
    co         : CO concentration (ppm)
    no2        : NO2 concentration (ppb)
    so2        : SO2 concentration (ppb)
    o3         : O3 concentration (ppb)
    hour       : Hour of day (0-23) for context
    day_of_year: Day of year (1-365) for seasonal context
    predictor  : Optional AQIPredictor for ML-based estimate
    location   : Label for the report

    Returns
    -------
    dict with full analysis and advisory
    """
    aqi_calc, dominant, sub_indices = calculate_aqi(pm25, pm10, co, no2, so2, o3)
    category, color, emoji = get_category(aqi_calc)
    advisory = get_health_advisory(category)

    ml_aqi = None
    if predictor:
        ml_aqi = round(predictor.predict(pm25, pm10, co, no2, so2, o3, hour, day_of_year))

    return {
        'location': location,
        'inputs': {
            'PM2.5 (μg/m³)': pm25,
            'PM10 (μg/m³)':  pm10,
            'CO (ppm)':       co,
            'NO2 (ppb)':      no2,
            'SO2 (ppb)':      so2,
            'O3 (ppb)':       o3,
        },
        'aqi': {
            'calculated': aqi_calc,
            'ml_predicted': ml_aqi,
            'sub_indices': sub_indices,
            'dominant_pollutant': dominant,
        },
        'category': {
            'name': category,
            'color': color,
            'emoji': emoji,
        },
        'advisory': advisory,
    }


def print_report(result: dict):
    """Pretty-print a full air quality report to the console."""
    SEP = "═" * 68

    print(f"\n{SEP}")
    print(f"  AIR QUALITY REPORT  —  {result['location']}")
    print(SEP)

    print("\n  POLLUTANT READINGS")
    print("  " + "─" * 40)
    for k, v in result['inputs'].items():
        bar_len = min(int(v / 5), 30)
        bar = "█" * bar_len
        print(f"  {k:<20} {v:>8.2f}  {bar}")

    print("\n  SUB-INDEX AQI VALUES")
    print("  " + "─" * 40)
    for poll, idx in result['aqi']['sub_indices'].items():
        marker = " ← DOMINANT" if poll == result['aqi']['dominant_pollutant'] else ""
        print(f"  {poll:<8} sub-AQI = {idx:>4}{marker}")

    calc = result['aqi']['calculated']
    ml   = result['aqi']['ml_predicted']
    cat  = result['category']

    print(f"\n  OVERALL AQI")
    print("  " + "─" * 40)
    print(f"  Calculated (EPA formula) : {calc}")
    if ml is not None:
        print(f"  ML Predicted             : {ml}")
    print(f"  Category   : {cat['emoji']}  {cat['name']}")
    print(f"  Dominant Pollutant: {result['aqi']['dominant_pollutant']}")

    adv = result['advisory']
    print(f"\n  HEALTH ADVISORY")
    print("  " + "─" * 40)
    print(f"  {adv['summary']}")
    print(f"\n  General Public   : {adv['general_public']}")
    print(f"  Sensitive Groups : {adv['sensitive_groups']}")
    print(f"  Outdoor Activity : {adv['outdoor_activity']}")
    print(f"  Mask Advice      : {adv['mask_advice']}")
    print(f"  Windows/Indoors  : {adv['window_advice']}")
    print(f"\n  TIPS")
    for tip in adv['tips']:
        print(f"   • {tip}")

    print(f"\n{SEP}\n")


# ─── Demo / CLI Entry Point ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading ML model …")
    try:
        predictor = AQIPredictor('aqi_model_rf.pkl', 'aqi_scaler_rf.pkl')
    except FileNotFoundError:
        predictor = None
        print("  (Model files not found — running formula-only mode)")

    scenarios = [
        {"label": "Clean Mountain Air",  "pm25": 5,   "pm10": 12,  "co": 0.3, "no2": 10,  "so2": 3,   "o3": 30},
        {"label": "Urban Morning Rush",  "pm25": 38,  "pm10": 65,  "co": 2.1, "no2": 85,  "so2": 22,  "o3": 55},
        {"label": "Industrial Area",     "pm25": 75,  "pm10": 140, "co": 6.0, "no2": 180, "so2": 90,  "o3": 80},
        {"label": "Severe Smog Episode", "pm25": 180, "pm10": 290, "co": 12,  "no2": 420, "so2": 200, "o3": 130},
        {"label": "Wildfire Smoke",      "pm25": 280, "pm10": 380, "co": 15,  "no2": 60,  "so2": 20,  "o3": 95},
    ]

    for sc in scenarios:
        result = analyze_air_quality(
            pm25=sc["pm25"], pm10=sc["pm10"], co=sc["co"],
            no2=sc["no2"],  so2=sc["so2"],  o3=sc["o3"],
            predictor=predictor,
            location=sc["label"]
        )
        print_report(result)
