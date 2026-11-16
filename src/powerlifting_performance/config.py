from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
SUBMISSIONS_DIR = ROOT_DIR / "submissions"

ID = "id"
TARGET = "TotalKg"

NUM_COLS = ["Age", "BodyweightKg", "MeetYear"]
CAT_COLS = [
    "Sex", "Equipment", "AgeClass", "BirthYearClass", "WeightClassKg", "Tested", "Country", "State",
    "Federation", "ParentFederation", "MeetCountry", "MeetState", "Sanctioned",
]
FEATURES = NUM_COLS + CAT_COLS
TREE_CAT_COLS = [c for c in CAT_COLS if c != "WeightClassKg"]

MEET_KEY = ["Federation", "MeetYear", "MeetCountry", "MeetState"]
PROFILE_KEY = ["Sex", "birth_year", "Country", "State", "Federation", "WeightClassKg"]
CONTEXT_KEY = ["Sex", "Equipment", "bodyweight_bin"]
LEVEL_KEYS = ["meet_key", "profile_key"]
KEY_COLS = LEVEL_KEYS + ["context_key", "duplicate_group"]
