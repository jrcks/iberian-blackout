"""
Configuration file for Iberian Blackout Analysis
This module provides centralized configuration management for the analysis toolkit.
"""

import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the analysis toolkit"""

    # Date Configuration
    BEFORE_BLACKOUT_DATE = datetime(2025, 4, 28)
    DURING_BLACKOUT_DATE = datetime(2025, 4, 29)    # Directory Configuration
    DATA_DIR = Path(os.getenv('DATA_DIR', './data'))
    FILTERED_DATA_DIR = Path(os.getenv('FILTERED_DATA_DIR', './filtered_data'))
    PLOTS_DIR = Path(os.getenv('PLOTS_DIR', './plots'))
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_SECRET')
    AWS_ENDPOINT_URL = os.getenv('AWS_ENDPOINT_URL')
    AWS_BUCKET_NAME = os.getenv('AWS_BUCKET_NAME')

    # IP2Location Configuration
    IP2LOCATION_TOKEN = os.getenv('IP2LOCATION_LITE_TOKEN')

    # Measurement Types
    MEASUREMENT_TYPES = ["manycast_v4", "manycast_v6", "unicast_v4", "unicast_v6"]

    # Plot Configuration
    PLOT_DPI = 800
    PLOT_STYLE = 'default'

    # Countries of Interest
    TARGET_COUNTRIES = ['ES', 'PT']  # Spain and Portugal    @classmethod
    def validate_credentials(cls):
        """Validate that required credentials are available"""
        missing_vars = []

        if not cls.AWS_ACCESS_KEY_ID:
            missing_vars.append('AWS_ACCESS_KEY_ID')
        if not cls.AWS_SECRET_ACCESS_KEY:
            missing_vars.append('AWS_ACCESS_KEY_SECRET')
        if not cls.AWS_ENDPOINT_URL:
            missing_vars.append('AWS_ENDPOINT_URL')
        if not cls.AWS_BUCKET_NAME:
            missing_vars.append('AWS_BUCKET_NAME')
        if not cls.IP2LOCATION_TOKEN:
            missing_vars.append('IP2LOCATION_LITE_TOKEN')

        if missing_vars:
            raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}. "
                           "Please check your .env file.")

    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.FILTERED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        (cls.PLOTS_DIR / 'asn_plots').mkdir(parents=True, exist_ok=True)
