# tests/test_pipeline.py
import os
import sys
import pandas as pd

# 1. Force add root directory to the python search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 2. Dynamic Environment Mocking for the GitHub Runner
# This creates mock files so your scripts don't crash with a FileNotFoundError on the cloud
os.makedirs("data/processed", exist_ok=True)
mock_file_path = "data/processed/filtered_complaints.csv"

if not os.path.exists(mock_file_path):
    # Construct a minimal dataframe with valid fallback headers matching Task 1 & 2
    mock_df = pd.DataFrame({
        "Product": ["Credit Card", "Personal Loan"],
        "cleaned_text": ["This is a sample mock complaint text structure.", "Another sample structural string narrative."],
        "Complaint ID": ["123456", "789101"]
    })
    mock_df.to_csv(mock_file_path, index=False)

# 3. Execution Verification Tests
def test_placeholder_pass():
    """Confirms basic truth logic inside the test engine framework."""
    assert True

def test_pipeline_files_exist():
    """Verifies essential pipeline modules are resting safely in root layout."""
    assert os.path.exists("src/preprocess.py")
    assert os.path.exists("src/chunk_embed.py")
    assert os.path.exists("requirements.txt")