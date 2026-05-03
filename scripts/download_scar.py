"""Download MU-Glioma-Post dataset from TCIA using tcia_utils.

This script fetches imaging data from The Cancer Imaging Archive (TCIA)
without requiring IBM Aspera or any browser plugins.
"""

import os
import sys
import time
from pathlib import Path

try:
    from tcia_utils import nbia
except ImportError:
    print("Error: tcia_utils not installed")
    print("Run: pip install tcia_utils")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("Error: requests not installed")
    print("Run: pip install requests")
    sys.exit(1)


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = str(PROJECT_ROOT / "Data" / "Raw" / "MU-Glioma-Post" / "images")
COLLECTION_NAME = "MU Glioma Postoperative"
CLINICAL_URL = "https://www.cancerimagingarchive.net/collection/mu-glioma-post/"


def verify_collection_exists() -> None:
    """Verify collection exists and print all available collections."""
    print("\n" + "=" * 80)
    print("Verifying TCIA Collection")
    print("=" * 80)
    
    url = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getCollectionValues"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        collections = response.json()
        
        print(f"\nSearching for 'glioma' in {len(collections)} available collections...")
        matches = [c.get("Collection", "") for c in collections if "glioma" in c.get("Collection", "").lower()]
        
        if matches:
            print(f"\nMatches found ({len(matches)}):")
            for match in matches:
                print(f"  - {match}")
        else:
            print("\nNo collections with 'glioma' found.")
        
        print(f"\nAll available collections:")
        for c in collections:
            coll_name = c.get("Collection", "")
            print(f"  - {coll_name}")
            
        if COLLECTION_NAME not in [c.get("Collection", "") for c in collections]:
            print(f"\nWarning: Exact collection name '{COLLECTION_NAME}' not found in TCIA.")
            print("Check the list above and update COLLECTION_NAME if needed.")
        else:
            print(f"\n✓ Collection '{COLLECTION_NAME}' found in TCIA.")
            
    except Exception as exc:
        print(f"Error verifying collection: {type(exc).__name__}: {exc}")


def create_output_directory() -> None:
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")


def fetch_series_list() -> list:
    """Fetch list of series from TCIA via REST API."""
    print(f"\nFetching series list from TCIA for collection '{COLLECTION_NAME}'...")
    try:
        series = nbia.getSeries(collection=COLLECTION_NAME)
        if not series:
            print(f"No series found for collection '{COLLECTION_NAME}'")
            print("Note: This dataset may require Aspera or manual download from TCIA.")
            sys.exit(1)
        print(f"Total series found: {len(series)}\n")
        return series
    except Exception as exc:
        print(f"Error fetching series from TCIA: {exc}")
        sys.exit(1)


def download_series(series: list) -> tuple[int, int, int]:
    """Download all series from the list.

    Args:
        series: List of series from TCIA

    Returns:
        Tuple of (attempted, succeeded, failed) counts
    """
    attempted = 0
    succeeded = 0
    failed = 0
    failed_series = []

    print("Starting downloads...\n")
    for index, series_item in enumerate(series, start=1):
        attempted += 1

        try:
            series_uid = series_item.get("SeriesInstanceUID", f"series_{index}")
            print(f"Downloading series {index}/{len(series)}: {series_uid}")

            nbia.downloadSeries(
                seriesData=[series_item],
                path=OUTPUT_DIR,
                format="nifti",
            )
            succeeded += 1

            if index % 10 == 0:
                print(f"  Progress: Downloaded {succeeded}/{attempted} series\n")

            time.sleep(1)

        except Exception as exc:
            failed += 1
            series_uid = series_item.get("SeriesInstanceUID", f"series_{index}")
            error_msg = f"{type(exc).__name__}: {exc}"
            failed_series.append((series_uid, error_msg))
            print(f"  Error downloading series {index}: {error_msg}\n")
            time.sleep(1)
            continue

    return attempted, succeeded, failed, failed_series


def check_clinical_data() -> None:
    """Check if clinical data is available and provide download instructions."""
    clinical_path = str(PROJECT_ROOT / "Data" / "Raw" / "MU-Glioma-Post" / "clinical_data.xlsx")

    if os.path.exists(clinical_path):
        print(f"Clinical data already exists at: {clinical_path}")
        return

    print("\n" + "=" * 80)
    print("Clinical Data")
    print("=" * 80)
    print(
        f"\nClinical data file not found at: {clinical_path}\n"
        f"To obtain clinical data, please:\n"
        f"1. Visit: {CLINICAL_URL}\n"
        f"2. Download the clinical data CSV/XLSX file\n"
        f"3. Save it to: {clinical_path}\n"
    )


def print_summary(attempted: int, succeeded: int, failed: int, failed_series: list) -> None:
    """Print download summary statistics."""
    print("\n" + "=" * 80)
    print("Download Summary")
    print("=" * 80)
    print(f"Total series attempted: {attempted}")
    print(f"Total series succeeded: {succeeded}")
    print(f"Total series failed: {failed}")
    print(f"Success rate: {100 * succeeded / attempted:.1f}%" if attempted > 0 else "N/A")
    print(f"\nOutput directory: {OUTPUT_DIR}")

    if failed_series:
        print(f"\nFailed series ({len(failed_series)}):")
        for series_uid, error_msg in failed_series[:10]:
            print(f"  - {series_uid}: {error_msg}")
        if len(failed_series) > 10:
            print(f"  ... and {len(failed_series) - 10} more")


def main() -> None:
    """Main download pipeline."""
    print("=" * 80)
    print("MU-Glioma-Post Dataset Downloader (TCIA)")
    print("=" * 80)

    create_output_directory()
    verify_collection_exists()
    series = fetch_series_list()
    attempted, succeeded, failed, failed_series = download_series(series)
    check_clinical_data()
    print_summary(attempted, succeeded, failed, failed_series)


if __name__ == "__main__":
    main()
