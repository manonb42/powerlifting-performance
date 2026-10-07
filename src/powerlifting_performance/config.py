from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
SUBMISSIONS_DIR = ROOT_DIR / "submissions"

ID = "id"
TARGET = "TotalKg"

CAT_COLS = ["Sex", "Equipment", "Tested", "Federation", "Country"]

NUM_COLS = ["Age", "BodyweightKg", "MeetYear"]
ALL_CAT_COLS = [
    "Sex", "Equipment", "AgeClass", "BirthYearClass", "WeightClassKg", "Tested", "Country", "State",
    "Federation", "ParentFederation", "MeetCountry", "MeetState", "Sanctioned",
]
