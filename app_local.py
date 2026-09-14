"""Job Shop Scheduling (FJSS) Optimization - Local Workstation Edition.

Dedicated dashboard for running on localhost with full multi-core CPU power,
unlimited time limits, full Gurobi & CPLEX exact solvers, and comprehensive dataset support.

Run with:
    streamlit run app_local.py
"""
import os
import sys
from pathlib import Path

# Force environment mode to local
os.environ['APP_EDITION'] = 'local'

# Add current directory to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Run main application in Localhost Workstation mode
import app
