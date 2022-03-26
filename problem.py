import os
import string
from glob import glob

import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score

import rampwf as rw
from rampwf.score_types.base import BaseScoreType


problem_title = "Hotspot challenge"

Predictions = rw.prediction_types.make_regression()
workflow = rw.workflows.Regressor()

class R_2(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name="R^2", precision=4):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        r_2 = r2_score(y_true, y_pred)
        return r_2

score_types = [
    R_2(name="r_2"),
]

def get_cv(X, y):
    cv = ShuffleSplit(n_splits=10, test_size=0.25, random_state=44)
    return cv.split(X, y)

def _get_data(path='.', split='train'):
    path = os.path.join(path, "data", split, f"{split}.npy")
    with open(path, 'rb') as f:
        X = np.load(f, allow_pickle=True)
        y = np.load(f, allow_pickle=True)
    return X, y

def get_train_data(path="."):
    return _get_data(path, "train")

def get_test_data(path="."):
    return _get_data(path, "test")
