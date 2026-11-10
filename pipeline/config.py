"""Central configuration: paths, column groups, and constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "Student_Depression_Dataset.csv"
ARTIFACTS_DIR = ROOT / "artifacts"

KAGGLE_DATASET = "mkhoirulmustaqim/student-depression-detection-dataset"
KAGGLE_DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_DATASET}"
CSV_NAME_IN_ZIP = "Student_Depression_Dataset.csv"

TARGET = "Depression"
POSITIVE_LABEL = "Yes"

NUMERIC_FEATURES = ["Age", "Current_semester", "CGPA", "Cost_without_study"]
# Categories with a natural order are ordinal-encoded; the rest are one-hot encoded.
ORDINAL_FEATURES = {
    "Family_background": ["Lower Middle class", "Middle class", "Rich class"],
    "Health_condition": ["Critical", "Abnormal", "Normal", "Good"],
}
NOMINAL_FEATURES = [
    "Gender",
    "Relationship_status",
    "Marital_status",
    "Living_style",
    "friend_circle",
    "Home_division",
]

TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
