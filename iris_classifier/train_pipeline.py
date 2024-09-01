# loading libraries
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# print(sys.path)
from iris_classifier.processing.data_manager import load_data, save_pipeline


def train_model():

    data = load_data()

    X = data.iloc[:, :-1].values
    y = data.iloc[:, 4].values

    # splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    # create model
    RFClf = RandomForestClassifier(n_estimators=50)

    # train model
    RFClf.fit(X_train, y_train)
    y_pred = RFClf.predict(X_test)

    # Save Model
    save_pipeline(pipeline_to_persist=RFClf)

    # get accuracy
    my_confusion = confusion_matrix(y_test, y_pred)
    my_accuracy = accuracy_score(y_test, y_pred)
    print("this is onfusion matrix", my_confusion)
    print("this is accuracy", my_accuracy)

    return 1


if __name__ == "__main__":
    train_model()
