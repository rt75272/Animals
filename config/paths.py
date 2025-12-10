# config/paths.py
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent
ROOT_DIR = CONFIG_DIR.parent

# Data directories
DATA_DIR = ROOT_DIR / 'data'
TRANSLATION_FILE = DATA_DIR / 'translation.json'
PROCESSED_DATA_DIR = DATA_DIR / 'dataset'

# Model directory
MODELS_DIR = ROOT_DIR / 'models'

# Source directory
SRC_DIR = ROOT_DIR / 'src'

# Webapp directories
WEBAPP_DIR = SRC_DIR / 'webapp'
STATIC_DIR = WEBAPP_DIR / 'static'
UPLOAD_DIR = STATIC_DIR / 'uploads'
TEMPLATES_DIR = WEBAPP_DIR / 'templates'

# Training output
CLASSES_JSON_PATH = MODELS_DIR / 'idx_to_class.json'
CLASS_NAMES_PATH = DATA_DIR / 'translation.json'
