#Importing required libraries
import json
# from google.colab import drive
# import requests
import zipfile
import pandas as pd
import numpy as np
import sklearn
from typing import *
from datetime import datetime
from tqdm import tqdm
# tqdm().pandas()
from collections import OrderedDict

#
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('notebook')
sns.set_style(style='darkgrid')

#tomer added
from scipy import stats
import statsmodels.formula.api as smf


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

#fairlearn
# false_positive_rate_difference and true_positive_rate_difference metrics are derived metrics
# created on-the-fly at import stage using make_derived_metric(...) at fairlearn file: _generated_metrics.py
# therefore they are not recognized by pycharm before the import is actually performed
# similar usage official example: https://github.com/fairlearn/fairlearn/blob/main/notebooks/Binary%20Classification%20with%20the%20UCI%20Credit-card%20Default%20Dataset.ipynb
from fairlearn.metrics import (
    MetricFrame,
    true_positive_rate,
    false_positive_rate,
    false_negative_rate,
    selection_rate,
    count,
    false_positive_rate_difference,
    true_positive_rate_difference,
    equalized_odds_difference
)

from FairClassifierPipeline import Utils as utils

def merge_feature_onehot_columns(feature_name: str, data: pd.DataFrame) -> pd.Series:
    # sensitive_feature_name = config['sensitive_feature']

    sensitive_feature_one_hot_columns = [col for col in data.columns if feature_name in col]

    feature_df = pd.DataFrame(data=[''] * data.shape[0], columns=[feature_name])
    for col in sensitive_feature_one_hot_columns:
        feature_df.loc[(data[col].values == 1)] = col

    # sensitive_test_feature = pd.DataFrame(data=['']*preprocessed_test_data.shape[0], columns=[sensitive_feature_name])
    # for col in sensitive_feature_one_hot_columns:
    #     sensitive_test_feature.loc[preprocessed_test_data[col] == 1] = col

    return feature_df[feature_name]



def get_feature_col_from_preprocessed_data(feature_name: str, data: pd.DataFrame) -> pd.Series:
    sensitive_feature_srs = None
    if feature_name not in data.columns:
        sensitive_feature_srs = merge_feature_onehot_columns(feature_name=feature_name, data=data)
    else:
        sensitive_feature_srs = data[feature_name]

    return sensitive_feature_srs



def average_odds_difference(y_true: pd.Series,
                            y_pred: pd.Series,
                            sensitive_feature_arr: np.array):
    fpr_diff = abs(false_positive_rate_difference(utils.to_int_srs(y_true), utils.to_int_srs(pd.Series(y_pred)),
                                                  sensitive_features=sensitive_feature_arr))
    tpr_diff = abs(true_positive_rate_difference(utils.to_int_srs(y_true), utils.to_int_srs(pd.Series(y_pred)),
                                                 sensitive_features=sensitive_feature_arr))
    aod_score = 0.5 * (fpr_diff + tpr_diff)

    return aod_score



def get_fairness_score_for_sensitive_features(sensitive_features_names: List[str],
                                              fairness_metric: str,
                                              y_true: pd.Series,
                                              y_pred: pd.Series,
                                              data: pd.DataFrame):
    fairness_metric = fairness_metric.lower()
    assert fairness_metric.lower() in ['eod', 'aod'], 'currently support eod and aod only'
    assert isinstance(y_true, pd.Series), 'y_true must be of type pd.Series'
    assert isinstance(y_pred, pd.Series), 'y_pred must be of type pd.Series'
    assert isinstance(data, pd.DataFrame), 'data must be of type pd.DataFrame'

    snsftr_frns_scores_dict = {}
    for ftr in sensitive_features_names:
        sensitive_feature_srs = None
        sensitive_feature_srs = get_feature_col_from_preprocessed_data(data=data, feature_name=ftr)
        if fairness_metric.lower() == 'eod':
            result = equalized_odds_difference(y_true=y_true,
                                               y_pred=y_pred,
                                               sensitive_features=sensitive_feature_srs.values)
        else:
            result = average_odds_difference(y_true=y_true,
                                                            y_pred=y_pred,
                                                            sensitive_feature_arr=sensitive_feature_srs.values)

        snsftr_frns_scores_dict[f'{ftr}:{fairness_metric}'] = result

    return snsftr_frns_scores_dict



def get_feature_sub_groups_by_selection_rate(y_true: pd.Series,
                                             y_pred: pd.Series,
                                             sensitive_feature_srs: pd.Series):
    metrics_dict = {"selection_rate": selection_rate}

    mf2 = MetricFrame(
        metrics=metrics_dict,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_feature_srs.values)

    metircs_df = mf2.by_group
    # max_selection_rate = metircs_df.selection_rate.max()
    # min_selection_rate = metircs_df.selection_rate.min()
    # selection_rate_threshold = (((max_selection_rate+min_selection_rate)/2)+avg_selection_rate)/2
    selection_rate_threshold = metircs_df.selection_rate.mean()
    metircs_df['sub_groups'] = (metircs_df.selection_rate > selection_rate_threshold) * 1

    priv = tuple(metircs_df[metircs_df.sub_groups == 1].index.values)
    unpriv = tuple(metircs_df[metircs_df.sub_groups == 0].index.values)

    return mf2.by_group, (unpriv, priv)

def drop_sensitive_feature(sensitive_col_name:str,
                              data:pd.DataFrame,
                              snsftr_slctrt_sub_groups:Tuple[Tuple,Tuple]) -> pd.DataFrame:

    # remove the sensitive name (eg: 'sex')
    if sensitive_col_name in data.columns:
        return data.drop(columns=[sensitive_col_name])
    else:
        # remove the sensitive columns (eg: 'statussex_A91')
        return data.drop(columns=(list(snsftr_slctrt_sub_groups[0])+list(snsftr_slctrt_sub_groups[1])))