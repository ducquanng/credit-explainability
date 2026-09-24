"""Paths, seed and the thresholds used to judge explanation quality."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_FILE = DATA_DIR / "german.data"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

DATA_URL = "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"
SEED = 42

# An explanation is only worth reading if the local surrogate actually tracks the
# model near the instance, and if it says the same thing when re-run.
MIN_LOCAL_R2 = 0.60        # weighted R^2 of the surrogate in the neighbourhood
MIN_TOP_K_JACCARD = 0.60   # overlap of top-k features across random seeds
