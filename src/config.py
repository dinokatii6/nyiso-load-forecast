"""Central configuration for the project"""
from pathlib import Path

#project path variables
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"

# Creates directories
for _d in (RAW_DIR, PROCESSED_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# EIA API configuration
EIA_BASE_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
BA_CODE = "NYIS"                 # New York ISO balancing authority code
LOCAL_TZ = "America/New_York"
START_DATE = "2019-01-01"
END_DATE = "2026-07-31"


# Weather sites: a population-weighted proxy for statewide NYISO load 
# These weights are a rough approximation of where electricity is actually consumed.
WEATHER_SITES = {
    "nyc":      {"lat": 40.7128, "lon": -74.0060, "weight": 0.55},
    "buffalo":  {"lat": 42.8864, "lon": -78.8784, "weight": 0.15},
    "albany":   {"lat": 42.6526, "lon": -73.7562, "weight": 0.15},
    "syracuse": {"lat": 43.0481, "lon": -76.1474, "weight": 0.15},
}

# Forecasting setup 
HORIZON_HOURS = 24   # how far ahead we predict
TEST_MONTHS = 12     # final 12 months held out as the test set