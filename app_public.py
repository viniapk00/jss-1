"""Job Shop Scheduling (FJSS) Optimization - Public Cloud Demo Edition.

Dedicated dashboard for public deployment on Streamlit Community Cloud.
Optimized for 3 GB container RAM limits, fast execution, lightweight heuristics,
and instant exploration of precomputed verified MIP benchmarks.

Run with:
    streamlit run app_public.py
"""
import os
import sys
from pathlib import Path

# Force environment mode to public cloud
os.environ['APP_EDITION'] = 'public'

# Add current directory to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Run main application in Public Cloud mode
import app
