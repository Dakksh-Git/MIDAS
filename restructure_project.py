"""
Project directory restructuring script with dry-run capability.

This script reorganizes the ML project directory structure according to best practices.
Set DRY_RUN = True to preview changes without executing them.
Set DRY_RUN = False to execute all operations.
"""

from pathlib import Path
import shutil
from datetime import datetime
from typing import List, Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================

DRY_RUN = False  # Set to False to execute; True to preview only
BASE_DIR = Path(r"C:\Minor-II project")
LOG_FILE = BASE_DIR / "restructure_log.txt"

# ============================================================================
# LOGGING SETUP
# ============================================================================

class DualLogger:
    """Log to both console and file."""
    
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.messages: List[str] = []
    
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] [{level}] {message}"
        print(formatted)
        self.messages.append(formatted)
    
    def info(self, message: str):
        self.log(message, "INFO")
    
    def warning(self, message: str):
        self.log(message, "WARNING")
    
    def error(self, message: str):
        self.log(message, "ERROR")
    
    def success(self, message: str):
        self.log(message, "SUCCESS")
    
    def write_to_file(self):
        """Write all logged messages to file."""
        with open(self.log_path, 'w', encoding='utf-8') as f:
            for message in self.messages:
                f.write(message + '\n')

logger = DualLogger(LOG_FILE)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def is_empty_directory(path: Path) -> bool:
    """Check if directory exists and is empty."""
    if not path.exists():
        return False
    if not path.is_dir():
        return False
    return len(list(path.iterdir())) == 0

def move_item(source: Path, destination: Path, item_type: str = "item") -> bool:
    """
    Move source to destination. Create parent directories if needed.
    
    Args:
        source: Source path to move from
        destination: Destination path to move to
        item_type: Description of item type (for logging)
    
    Returns:
        True if successful, False otherwise
    """
    # Check if source exists
    if not source.exists():
        logger.warning(f"Source does not exist, skipping: {source}")
        return False
    
    # Create parent directory of destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created parent directories for: {destination.parent}")
    
    # Log the operation
    logger.info(f"Moving {item_type}: {source} → {destination}")
    
    if not DRY_RUN:
        try:
            shutil.move(str(source), str(destination))
            logger.success(f"✓ Moved {item_type}: {source.name}")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to move {item_type}: {e}")
            return False
    else:
        logger.info(f"[DRY RUN] Would move {item_type}")
        return True

def delete_empty_directory(path: Path, item_type: str = "directory") -> bool:
    """
    Delete directory only if it's empty.
    
    Args:
        path: Path to delete
        item_type: Description of item type (for logging)
    
    Returns:
        True if successful, False otherwise
    """
    if not path.exists():
        logger.warning(f"{item_type} does not exist, skipping: {path}")
        return False
    
    if not is_empty_directory(path):
        logger.warning(f"{item_type} is not empty, skipping: {path}")
        return False
    
    logger.info(f"Deleting empty {item_type}: {path}")
    
    if not DRY_RUN:
        try:
            shutil.rmtree(str(path))
            logger.success(f"✓ Deleted empty {item_type}: {path.name}")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to delete {item_type}: {e}")
            return False
    else:
        logger.info(f"[DRY RUN] Would delete empty {item_type}")
        return True

# ============================================================================
# RESTRUCTURING OPERATIONS
# ============================================================================

def restructure_project():
    """Execute all project restructuring operations."""
    
    logger.info("=" * 80)
    logger.info("PROJECT RESTRUCTURING SCRIPT")
    logger.info("=" * 80)
    logger.info(f"Base directory: {BASE_DIR}")
    logger.info(f"DRY_RUN mode: {DRY_RUN}")
    logger.info("=" * 80)
    
    # Track operations
    operations: List[Tuple[str, bool]] = []
    
    # ========================================================================
    # 1. Move BraTS2020 directories
    # ========================================================================
    logger.info("\n[STEP 1] Restructuring BraTS2020 datasets...")
    
    src = BASE_DIR / "BraTS2020_TrainingData"
    dst = BASE_DIR / "Data" / "Raw" / "BraTS2020" / "TrainingData"
    success = move_item(src, dst, "directory: BraTS2020_TrainingData")
    operations.append(("BraTS2020_TrainingData → Data/Raw/BraTS2020/TrainingData", success))
    
    src = BASE_DIR / "BraTS2020_ValidationData"
    dst = BASE_DIR / "Data" / "Raw" / "BraTS2020" / "ValidationData"
    success = move_item(src, dst, "directory: BraTS2020_ValidationData")
    operations.append(("BraTS2020_ValidationData → Data/Raw/BraTS2020/ValidationData", success))
    
    # ========================================================================
    # 2. Move Inflammatory dataset (rename: remove space)
    # ========================================================================
    logger.info("\n[STEP 2] Moving Inflammatory dataset...")
    
    src = BASE_DIR / "Inflammatory dataset"
    dst = BASE_DIR / "Data" / "Raw" / "Inflammatory"
    success = move_item(src, dst, "directory: Inflammatory dataset")
    operations.append(("Inflammatory dataset → Data/Raw/Inflammatory", success))
    
    # ========================================================================
    # 3. Move Scar dataset/Imaging
    # ========================================================================
    logger.info("\n[STEP 3] Moving Scar dataset Imaging...")
    
    src = BASE_DIR / "Scar dataset" / "Imaging"
    dst = BASE_DIR / "Data" / "Raw" / "Scar"
    success = move_item(src, dst, "directory: Scar dataset/Imaging")
    operations.append(("Scar dataset/Imaging → Data/Raw/Scar", success))
    
    # ========================================================================
    # 4. Move metadata directory
    # ========================================================================
    logger.info("\n[STEP 4] Moving metadata directory...")
    
    src = BASE_DIR / "metadata"
    dst = BASE_DIR / "Data" / "metadata"
    success = move_item(src, dst, "directory: metadata")
    operations.append(("metadata → Data/metadata", success))
    
    # ========================================================================
    # 5. Move remind directory
    # ========================================================================
    logger.info("\n[STEP 5] Moving remind directory...")
    
    src = BASE_DIR / "remind"
    dst = BASE_DIR / "scripts" / "remind"
    success = move_item(src, dst, "directory: remind")
    operations.append(("remind → scripts/remind", success))
    
    # ========================================================================
    # 6. Move utility scripts to scripts/
    # ========================================================================
    logger.info("\n[STEP 6] Moving utility scripts to scripts/...")
    
    utility_scripts = [
        "kaggle_setup.py",
        "download_scar.py",
        "reorganize.py",
        "reorganize_ixi.py"
    ]
    
    for script in utility_scripts:
        src = BASE_DIR / script
        dst = BASE_DIR / "scripts" / script
        success = move_item(src, dst, f"script: {script}")
        operations.append((f"{script} → scripts/{script}", success))
    
    # ========================================================================
    # 7. Move plotting scripts to src/plots/
    # ========================================================================
    logger.info("\n[STEP 7] Moving plotting scripts to src/plots/...")
    
    plot_scripts = [
        "plot_preprocessing_flowchart.py",
        "plot_system_overview.py",
        "plot_training_curves.py"
    ]
    
    for script in plot_scripts:
        src = BASE_DIR / "src" / script
        dst = BASE_DIR / "src" / "plots" / script
        success = move_item(src, dst, f"script: {script}")
        operations.append((f"src/{script} → src/plots/{script}", success))
    
    # ========================================================================
    # 8. Delete empty directories
    # ========================================================================
    logger.info("\n[STEP 8] Cleaning up empty directories...")
    
    ct_dir = BASE_DIR / "Data" / "processed" / "CT"
    success = delete_empty_directory(ct_dir, "directory: Data/processed/CT")
    operations.append(("Delete Data/processed/CT (if empty)", success))
    
    pet_dir = BASE_DIR / "Data" / "processed" / "PET"
    success = delete_empty_directory(pet_dir, "directory: Data/processed/PET")
    operations.append(("Delete Data/processed/PET (if empty)", success))
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("RESTRUCTURING SUMMARY")
    logger.info("=" * 80)
    
    total = len(operations)
    successful = sum(1 for _, success in operations if success)
    skipped = total - successful
    
    for operation, success in operations:
        status = "✓ OK" if success else "✗ SKIP"
        logger.info(f"{status}: {operation}")
    
    logger.info("-" * 80)
    logger.info(f"Total operations: {total}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Skipped/Failed: {skipped}")
    
    if DRY_RUN:
        logger.info("\n[DRY RUN MODE] No actual changes were made.")
        logger.info("Set DRY_RUN = False to execute the restructuring.")
    else:
        logger.info("\n[EXECUTED] All operations have been completed.")
    
    logger.info("=" * 80)
    logger.info(f"Log saved to: {LOG_FILE}")
    logger.info("=" * 80)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    restructure_project()
    logger.write_to_file()
    print(f"\n✓ Complete log written to: {LOG_FILE}")
