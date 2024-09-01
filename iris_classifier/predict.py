import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from iris_classifier.config.core import PIPELINE_SAVE_FILE, TRAINED_MODEL_DIR
from iris_classifier.processing.data_manager import load_pipeline
from iris_classifier.processing.validation import validate_inputs

from iris_classifier import __version__ as _version

save_file_name = f"{PIPELINE_SAVE_FILE}{_version}.pkl"
pipeline_file_name = TRAINED_MODEL_DIR / save_file_name

iris_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data: Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model"""

    print(input_data)
    validated_data, errors = validate_inputs(input_dict=input_data)

    if not errors:

        predictions = list(iris_pipe.predict(validated_data))
        probability = float(iris_pipe.predict_proba(validated_data).max())

        results = {"predictions": predictions, "probability":probability, "version": _version, "errors": errors}
        print(results)
    else:
        # results = {"input": input_data, "version": _version, "errors": errors}
        # print(results)
        results = {}
        print(errors)

    return results


if __name__ == "__main__":

    data_in = {
        "sepallength": 5.1,
        "sepalwidth": 3.5,
        "petallength": 1.4,
        "petalwidth": 0.2,
    }

    make_prediction(input_data=data_in)
