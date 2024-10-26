from pathlib import Path

PACKAGE_ROOT_DIR_PATH = Path(__file__).resolve().parent
SRC_DIR_PATH = PACKAGE_ROOT_DIR_PATH.parent
DATA_DIR_PATH = SRC_DIR_PATH / "data"
