#Importing required libraries
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

#tomer added
from scipy import stats
import statsmodels.formula.api as smf

from collections import defaultdict

np.random.seed(sum(map(ord, "aesthetics")))

from sklearn.datasets import make_classification
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report,confusion_matrix, roc_curve, roc_auc_score, auc, accuracy_score
from sklearn.model_selection import ShuffleSplit,train_test_split, cross_val_score, GridSearchCV
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
from sklearn.metrics import fbeta_score,f1_score
from sklearn.metrics import make_scorer
from sklearn.preprocessing import FunctionTransformer

#anomaly detection models
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
import rrcf

#classifier
from xgboost import XGBClassifier


from pprint import pprint

# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

#fairlearn
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

class GermanBaseClf(BaseClf):
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
        # https://www.kaggle.com/hendraherviawan/predicting-german-credit-default
        # params2 = {
        #     'n_estimators': 3000,  # 1a - compared to 100 in 1a
        #     'objective': 'binary:logistic',  # 1a - same
        #     'learning_rate': 0.005,  # 1a - 0.1 (20 times slower)
        #     # 'gamma':0.01,
        #     'subsample': 0.555,  # 1a - no subsampling
        #     'colsample_bytree': 0.7,  # 1a - no col sumpling
        #     'min_child_weight': 3,  # 1a -
        #     'max_depth': 8,
        #     # 'seed':1024,
        #     'n_jobs': -1
        # }

        XGBClassifier_params = {
            'n_estimators': 3000,
            'objective': 'binary:logistic',
            'learning_rate': 0.005,
            # 'gamma':0.0,
            'subsample': 0.555,
            'colsample_bytree': 0.70,
            'min_child_weight': 3,
            'max_depth': 8,
            # 'seed':1024,
            'n_jobs': -1,
            'use_label_encoder': False
        }
        if X_test is not None and y_test is not None:
            #THIS IS EXACTLY THE ORIGINAL CODE from: https://www.kaggle.com/hendraherviawan/predicting-german-credit-default
            #There seems to be a "small problem" here as they include (X_test, y_test) as the second parameter in the 'eval set' param.
            # following the documentation (https://xgboost.readthedocs.io/en/stable/python/python_api.html), though they
            # specify "early_stopping_rounds" param, the early stopping might happen before based on the second element
            # in the eval_set parameter. This looks as Future data leakage into the fitting stage thus obiviousely wrong pratice!!
            eval_set = [(X_train, y_train), (X_test, y_test)]
            model = XGBClassifier(**XGBClassifier_params).fit(X=X_train, y=y_train,
                                                               eval_set=eval_set,eval_metric='auc',
                                                               early_stopping_rounds = 100, verbose=False)
            ntree_limit = model.best_ntree_limit
        else:
            model = XGBClassifier(**XGBClassifier_params)
            ntree_limit =  XGBClassifier().get_num_boosting_rounds() #avoid future data leak and fitting twice (might be weaker much faster for heavy usage)

        # print(model.best_ntree_limit)
        # print(f"model_.best_ntree_limit:{model_.best_ntree_limit}")
        model.set_params(**{'n_estimators': ntree_limit})
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

        model = GermanBaseClf.fit(X_train=X_train,
                                    y_train=y_train,
                                    X_test=X_test,
                                    y_test=y_test)

        return (model, *GermanBaseClf.predict(clf=model, X=X_test))


    @staticmethod
    def run_baseline(data:pd.DataFrame, config:Dict):
        '''
        :param data: the raw data overwhich to run the baseline model
        :return:
        - X_train:pd.DataFrame - preprocessed
        - Y_train:pd.Series - preprocessed
        - X_test:pd.DataFrame - preprocessed
        - y_test:pd.Series - preprocessed
        - model:XGBClassifier - pre-trained XGBClassifier base model
        - y_pred:pd.Series - pre-trainded model's predict result on X_test
        - y_pred_proba:pd.Series - pre-trainded model's predict_proba result on X_test
        '''
        data_baseline = data.copy()
        utils.save_date_processing_debug(data_baseline,"BASE_data_baseline")

        # Binarize the y output for easier use of e.g. ROC curves -> 0 = 'bad' credit; 1 = 'good' credit
        # Print number of 'good' credits (should be 700) and 'bad credits (should be 300)
        # data_baseline.classification.value_counts()

        # numerical variables labels
        numvars = ['creditamount', 'duration', 'installmentrate', 'residencesince', 'age',
                   'existingcredits', 'peopleliable', 'classification']

        # Standardization
        # numdata_std = pd.DataFrame(StandardScaler().fit_transform(data_baseline[numvars].drop(['classification'], axis=1)))

        # categorical variables labels
        catvars = ['existingchecking', 'credithistory', 'purpose', 'savings', 'employmentsince',
                   'statussex', 'otherdebtors', 'property', 'otherinstallmentplans', 'housing', 'job',
                   'telephone', 'foreignworker']


        # d = defaultdict(LabelEncoder)

        # Encoding the variable
        # lecatdata = data_baseline[catvars].apply(lambda x: d[x.name].fit_transform(x))

        # print transformations
        # for x in range(len(catvars)):
        #     print(catvars[x], ": ", data_baseline[catvars[x]].unique())
        #     print(catvars[x], ": ", lecatdata[catvars[x]].unique())

        # One hot encoding, create dummy variables for every category of every categorical variable
        dummyvars = pd.get_dummies(data_baseline[catvars])

        data_baseline_clean = pd.concat([data_baseline[numvars], dummyvars], axis=1)

        # print(data_baseline_clean.shape)
        utils.save_date_processing_debug(data_baseline_clean,"BASE_data_baseline_clean")
        # Unscaled, unnormalized data

        #create common train,. test split
        X_train, X_test = fair_ppl.split_data(data=data_baseline_clean,
                                                config=config)

        X_train[config['label_col']].replace([1, 2], [1, 0], inplace=True)
        X_test[config['label_col']].replace([1, 2], [1, 0], inplace=True)

        utils.save_date_processing_debug(X_train,'BASE_X_train.csv')
        utils.save_date_processing_debug(X_test,"BASE_X_test.csv")

        X_baseline_train_clean = X_train.drop(config['label_col'], axis=1)
        y_baseline_train_clean = X_train[config['label_col']]

        X_baseline_test_clean = X_test.drop(config['label_col'], axis=1)
        y_baseline_test_clean = X_test[config['label_col']]

        utils.save_date_processing_debug(X_baseline_train_clean,'BASE_X_baseline_train_clean')
        utils.save_date_processing_debug(y_baseline_train_clean,'BASE_y_baseline_train_clean')
        utils.save_date_processing_debug(X_baseline_test_clean,'BASE_X_baseline_test_clean')
        utils.save_date_processing_debug(y_baseline_test_clean,'BASE_y_baseline_test_clean')

        # X_baseline_train_clean, X_baseline_test_clean, y_baseline_train_clean, y_baseline_test_clean = train_test_split(
        #     X_baseline_clean, y_baseline_clean, test_size=0.2, random_state=1)


        return (X_baseline_train_clean, X_baseline_test_clean, y_baseline_train_clean,y_baseline_test_clean,
                *GermanBaseClf.fit_predict(X_train=utils.to_float_df(X_baseline_train_clean),
                                           y_train=utils.to_int_srs(y_baseline_train_clean),
                                           X_test=utils.to_float_df(X_baseline_test_clean),
                                           y_test= utils.to_int_srs(y_baseline_test_clean)))

