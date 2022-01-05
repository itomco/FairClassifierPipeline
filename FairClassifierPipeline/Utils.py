#Importing required libraries
import json
# from google.colab import drive
# import requests
import zipfile
from datetime import datetime
import os
import pandas as pd
import numpy as np
import sklearn
from typing import *
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

def get_datetime_tag():
    return datetime.now().strftime("%y%m%d_%H%M%S")

dt_tag = get_datetime_tag()
debug_data_preprocessing = False
data_files_saved_counter = 0

def save_date_processing_debug(data:Union[pd.Series, pd.DataFrame],data_name):
    if debug_data_preprocessing:
        directory = f"data_processing/{dt_tag}"
        if not os.path.exists(directory):
            os.makedirs(directory)

        global data_files_saved_counter
        data_files_saved_counter += 1
        data.to_csv(f"data_processing/{dt_tag}/[{data_files_saved_counter}]_{data_name}.csv")
        if isinstance(data,pd.DataFrame):
            pd.Series(data.columns).sort_values().to_csv(f"data_processing/{dt_tag}/[{data_files_saved_counter}]_{data_name}_columns.csv")

def to_int_srs(series:pd.Series) -> pd.Series:
    assert isinstance(series,pd.Series), 'df must be of type pd.Series'
    return pd.Series(np.array(series.values,dtype=int))

def to_float_df(df:pd.DataFrame) -> pd.DataFrame:
    assert isinstance(df,pd.DataFrame), 'df must be of type pd.DataFrame'
    return df.astype(float)

def print_confusion_matrix(test_labels, test_pred, y_pred_proba = None, do_plot:bool=False):
    cm = confusion_matrix(test_labels, test_pred)
    if do_plot:
        ax = plt.subplot()
        sns.heatmap(cm, annot=True, ax=ax, cmap='Reds', fmt="d")

        auc_title_tag = ''
        if y_pred_proba is not None:
            AUC = roc_auc_score(test_labels, y_pred_proba)
            auc_title_tag = f" [AUC:{AUC:.4f}]"

        ax.set_title(f'Confusion Matrix{auc_title_tag}')

        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')

        ax.xaxis.set_ticklabels([False, True])
        ax.yaxis.set_ticklabels([False, True])
        plt.show()
    else:
        print(f"Confusion Martix:\n{cm}")

    print('Classification Report:')
    print(classification_report(test_labels, test_pred, digits=4))

    print(f'True Positive Rate [Recall(1)]: {cm[1, 1] / sum(cm[1, :])}')
    print(f'False Positive Rate [1-RCL(0)]: {cm[0, 1] / sum(cm[0, :])}')
    print(f'Accuracy with default threshold: {(cm[0, 0] + cm[1, 1]) * 100 / (sum(cm).sum()):.4f}%\n')


# Function to get roc curve
def get_roc (y_test,y_pred):
    '''fpr, tpr, roc_auc'''
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    #Plot of a ROC curve
    # plt.figure()
    # lw = 2
    # plt.plot(fpr, tpr, color='darkorange',
    #          label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic')
    # plt.legend(loc="upper left")
    # plt.show()
    return(fpr, tpr, roc_auc)

def save_df(file_path:str, df:pd.DataFrame):
    file_dir = os.path.dirname(file_path)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    df.to_csv(file_path)