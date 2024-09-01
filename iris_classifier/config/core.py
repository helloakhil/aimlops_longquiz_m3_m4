# Path setup, and access the config.yml file, datasets folder & trained models
import os
import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Dict, List
from pydantic import BaseModel
from strictyaml import YAML, load

import iris_classifier

# Pipeline Definitions
PIPELINE_NAME = "iris_classifier"
PIPELINE_SAVE_FILE = "iris_classifier_v"

# Project Directories
PACKAGE_ROOT = Path(iris_classifier.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
# CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
# print(CONFIG_FILE_PATH)

DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(TRAINED_MODEL_DIR, exist_ok=True)
