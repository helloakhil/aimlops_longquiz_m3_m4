import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError


def validate_inputs(*, input_dict: dict) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    # validated_data = pre_processed[config.model_config.features].copy()
    input_array = None
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        input_obj = IrisSpeciesInput(**input_dict)
        input_array = np.array(
            [
                input_obj.sepallength,
                input_obj.sepalwidth,
                input_obj.petallength,
                input_obj.petalwidth,
            ]
        ).reshape(1, -1)

    except ValidationError as error:
        errors = error.json()

    return input_array, errors


class IrisSpeciesInput(BaseModel):
    sepallength: float
    sepalwidth: float
    petallength: float
    petalwidth: float


class MultipleDataInputs(BaseModel):
    inputs: List[IrisSpeciesInput]
