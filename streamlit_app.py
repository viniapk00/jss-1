"""Entry point for Streamlit Community Cloud (Public Demo Edition)."""
import os
import sys
from pathlib import Path

os.environ['APP_EDITION'] = 'public'

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app
