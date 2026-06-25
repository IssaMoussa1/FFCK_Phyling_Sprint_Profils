"""Shared configuration for the FFCK Sprint Phyling app."""

import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
REGISTRE = os.path.join(DATA_DIR, "registre.csv")

PHYLING_BASE_URL = "https://api.app.phyling.fr"
PHYLING_CLIENT_ID = 3  # FFCK

PHY_CACHE_ROOT = os.path.join(os.path.expanduser("~"), ".phyling_cache")
CACHE_DIR = os.path.join(PHY_CACHE_ROOT, "cache")
API_RECORDS_CACHE = os.path.join(PHY_CACHE_ROOT, "phyling_records.pkl")
API_RECORDS_CACHE_TTL_S = 6 * 3600

os.makedirs(CACHE_DIR, exist_ok=True)

