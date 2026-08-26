"""
Configuration for the RoleMemo data pipeline.

Fill in your API credentials and model ID before running any pipeline script.
All scripts import this module to obtain a shared OpenAI-compatible client.
"""

import os
from openai import OpenAI

# ── User configuration ────────────────────────────────────────────────────────
API_KEY   = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
BASE_URL  = os.environ.get("OPENAI_BASE_URL", "YOUR_BASE_URL_HERE")  # e.g. "https://api.deepseek.com/v1"
MODEL_ID  = os.environ.get("MODEL_ID",        "YOUR_MODEL_ID_HERE")  # e.g. "deepseek-chat"
# ─────────────────────────────────────────────────────────────────────────────

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
