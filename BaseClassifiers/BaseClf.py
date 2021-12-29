import pandas as pd
from xgboost import XGBClassifier
from typing import *

class BaseClf():
    @staticmethod
    def fit(X_train: pd.DataFrame,
            y_train: pd.Series,
            X_test: pd.DataFrame = None,
            y_test: pd.DataFrame = None,
            ):
        pass

    @staticmethod
    def predict(clf: XGBClassifier,
                X: pd.DataFrame
                ):
        pass

    @staticmethod
    def fit_predict(X_train:pd.DataFrame,
                    y_train:pd.Series,
                    X_test:pd.DataFrame,
                    y_test: pd.Series = None
                    ):
        pass

    @staticmethod
    def run_baseline(data:pd.DataFrame, config:Dict):
        pass