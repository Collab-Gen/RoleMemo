"""
Shared API configuration for RoleMemo inference and evaluation scripts.

Set your credentials via environment variables before running any script
under inference/:

    export OPENAI_API_KEY="your-key"
    export OPENAI_BASE_URL="your-base-url"   # any OpenAI-compatible endpoint
    export MODEL_ID="your-model-id"

Or edit the placeholder strings below directly.
"""

import os
from openai import OpenAI

# ── User configuration ────────────────────────────────────────────────────────
API_KEY  = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "YOUR_BASE_URL_HERE")
MODEL_ID = os.environ.get("MODEL_ID",        "YOUR_MODEL_ID_HERE")
# ─────────────────────────────────────────────────────────────────────────────

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
