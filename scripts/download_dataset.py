"""
Download script for retrieving the official UCI Student Performance dataset.
"""

import io
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from config.settings import RAW_DATA_DIR, RAW_DATA_FILE

UCI_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"


def download_and_extract(target_dir: Path = RAW_DATA_DIR) -> bool:
    """Download the zip archive from UCI ML Repository and extract student-mat.csv."""
    target_dir.mkdir(parents=True, exist_ok=True)
    if RAW_DATA_FILE.exists():
        print(f"Dataset already exists at {RAW_DATA_FILE} (size: {RAW_DATA_FILE.stat().st_size} bytes).")
        return True

    print(f"Retrieving archive from {UCI_URL}...")
    req = urllib.request.Request(UCI_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        content = resp.read()

    with zipfile.ZipFile(io.BytesIO(content)) as outer_zip:
        outer_zip.extractall(target_dir)

    inner_zip_path = target_dir / "student.zip"
    if inner_zip_path.exists():
        with zipfile.ZipFile(inner_zip_path) as inner_zip:
            inner_zip.extractall(target_dir)

    if RAW_DATA_FILE.exists():
        print(f"Successfully downloaded and extracted to {RAW_DATA_FILE}")
        return True
    else:
        print("Failed to locate student-mat.csv after extraction.", file=sys.stderr)
        return False


if __name__ == "__main__":
    download_and_extract()
