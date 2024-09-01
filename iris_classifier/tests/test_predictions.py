import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
sys.path.append(str(file.parents[2]))

import pytest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from iris_classifier.config.core import PIPELINE_SAVE_FILE, TRAINED_MODEL_DIR
from iris_classifier.processing.data_manager import load_pipeline
from iris_classifier import __version__ as _version


# Load and prepare the Iris dataset
iris = load_iris()
target_map = {0:"Iris-setosa", 1:"Iris-versicolor", 2:"Iris-virginica"}
feature_names = iris.feature_names
target_names = list(target_map.values())

X = iris.data
y = iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Initialize the classifier
save_file_name = f"{PIPELINE_SAVE_FILE}{_version}.pkl"
pipeline_file_name = TRAINED_MODEL_DIR / save_file_name

clf = load_pipeline(file_name=pipeline_file_name)

# Test Cases


def test_model_training():
    """Test that the model trains without errors"""
    assert clf is not None
    assert hasattr(clf, "predict")  # Ensure that the model has a predict method


def test_model_prediction():
    """Test that the model can make predictions"""
    predictions = clf.predict(X_test)
    assert predictions is not None
    assert len(predictions) == len(X_test)  # Ensure predictions match test size


def test_model_accuracy():
    """Test that the model achieves an acceptable accuracy"""
    predictions = clf.predict(X_test)
    y_test_imputed = np.array([target_map[i] for i in y_test])
    accuracy = accuracy_score(y_test_imputed, predictions)
    assert accuracy > 0.9  # Expecting accuracy greater than 90%


def test_custom_prediction():
    """Test the model with custom input data"""
    custom_data = np.array([[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]])
    custom_predictions = clf.predict(custom_data)
    assert len(custom_predictions) == len(
        custom_data
    )  # Ensure predictions match custom data size
    assert set(custom_predictions).issubset(
        set(target_names)
    )  # Predictions should be valid class indices


def test_prediction_labels():
    """Test that custom predictions map to valid class labels"""
    custom_data = np.array([[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]])
    custom_predictions = clf.predict(custom_data)
    assert len(custom_predictions) == len(
        custom_data
    )  # Ensure labels match custom data size
    assert all(
        isinstance(label, str) for label in custom_predictions
    )  # Ensure labels are strings
