import os
import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import typing as t
import re
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from iris_classifier import __version__ as _version
from iris_classifier.config.core import (
    DATASET_DIR,
    PIPELINE_SAVE_FILE,
    TRAINED_MODEL_DIR,
)


def load_data() -> pd.DataFrame:
    print("Loading Data")
    # loading data
    path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    headernames = [
        "sepal-length",
        "sepal-width",
        "petal-length",
        "petal-width",
        "Class",
    ]
    dataset = pd.read_csv(path, names=headernames)

    # save data
    save_data(dataset=dataset)

    return dataset


def save_data(dataset: pd.DataFrame):
    """
    Save base data to DATASET_DIR
    """
    filename = os.path.join(DATASET_DIR, "base_data.csv")

    dataset.to_csv(filename, index=False)


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{PIPELINE_SAVE_FILE}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
