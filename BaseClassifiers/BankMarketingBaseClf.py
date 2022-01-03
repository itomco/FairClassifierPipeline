# Importing required libraries
import json
# from google.colab import drive
# import requests
import zipfile
import pandas as pd
import numpy as np
import sklearn
from typing import *
from collections import OrderedDict

#
# %matplotlib inline
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set_context('notebook')
# sns.set_style(style='darkgrid')

# tomer added
from scipy import stats
import statsmodels.formula.api as smf

from collections import defaultdict

np.random.seed(sum(map(ord, "aesthetics")))

from sklearn.datasets import make_classification
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, auc, accuracy_score
from sklearn.model_selection import ShuffleSplit, train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize, StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline as pipe
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn import set_config
from sklearn_pandas import DataFrameMapper
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import fbeta_score, f1_score
from sklearn.metrics import make_scorer
from sklearn.preprocessing import FunctionTransformer

# anomaly detection models
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
import rrcf

# classifier
from xgboost import XGBClassifier

from pprint import pprint

# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# fairlearn
from fairlearn.metrics import (
    MetricFrame,
    true_positive_rate,
    false_positive_rate,
    false_negative_rate,
    selection_rate,
    count,
    equalized_odds_difference
)
import itertools

from BaseClassifiers.BaseClf import BaseClf
from FairClassifierPipeline import Utils as utils
from FairClassifierPipeline import FairPipeline as fair_ppl

class BankBaseClf(BaseClf):
    @staticmethod
    def predict(clf:XGBClassifier,
                X:pd.DataFrame
                ):

        y_pred = clf.predict(X)  # model.best_iteration
        y_pred_proba = clf.predict_proba(X)[:, 1]  # model.best_iteration

        return (y_pred, y_pred_proba)

    @staticmethod
    def fit(X_train: pd.DataFrame,
            y_train: pd.Series,
            X_test:pd.DataFrame = None,
            y_test:pd.Series = None
            ):
        model = XGBClassifier(random_state = 42,use_label_encoder=False)
        model.fit(X=X_train,
                  y=y_train,
                  eval_metric='logloss',
                  verbose=False)

        return (model)

    @staticmethod
    def fit_predict(X_train:pd.DataFrame,
                    y_train:pd.Series,
                    X_test:pd.DataFrame,#X_test is needed for the predict thus MUST NOT BE None !!!
                    y_test: pd.Series = None
                    ):

        model = BankBaseClf.fit(X_train=X_train,
                                y_train=y_train)

        return (model, *BankBaseClf.predict(clf=model, X=X_test))

    @staticmethod
    def run_baseline(data:pd.DataFrame,config:Dict):
        data_baseline = data.copy()

        #Converting object type data into numeric type using One-Hot encoding method which is
        #majorly used for XGBoost (for better accuracy) [Applicable only for non numeric categorical features]
        data_new = pd.get_dummies(data, columns=['job','marital',
                                                 'education','default',
                                                 'housing','loan',
                                                 'contact','month',
                                                 'poutcome'])
        #pd is instance of pandas. Using get_dummies method we can directly convert any type of data into One-Hot encoded format.

        # Since y is a class variable we will have to convert it into binary format. (Since 2 unique class values)

        # data_new.y.value_counts()

        # Checking types of all the columns converted
        # data_new.dtypes

        # Our New dataframe ready for XGBoost
        # data_new.head()

        # Spliting data as X -> features and y -> class variable

        # print(data_X.columns)
        # print(data_y.columns)

        X_train, X_test = fair_ppl.split_data(data=data_new,
                                              config=config)

        X_train[config['label_col']].replace(('yes', 'no'), (1, 0), inplace=True)
        X_test[config['label_col']].replace(('yes', 'no'), (1, 0), inplace=True)

        y_train = X_train[config['label_col']]
        X_train = X_train.drop([config['label_col']], axis=1)

        y_test = X_test[config['label_col']]
        X_test = X_test.drop([config['label_col']], axis=1)

        # Dividing records in training and testing sets along with its shape (rows, cols)
        # X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3, random_state=2, stratify=data_y)
        # print(X_train.shape)
        # print(X_test.shape)
        # print(y_train.shape)
        # print(y_test.shape)

        # Create an XGB classifier and train it on 70% of the data set.
        # y_train = y_train.iloc[:, 0]
        # y_test = y_test.iloc[:, 0]

        return(X_train, X_test, y_train, y_test,
               *BankBaseClf.fit_predict(X_train=utils.to_float_df(X_train),
                                           y_train=utils.to_int_srs(y_train),
                                           X_test=utils.to_float_df(X_test)))