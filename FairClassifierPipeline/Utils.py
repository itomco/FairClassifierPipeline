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


def print_confusion_matrix(test_labels, test_pred, y_pred_proba = None):
    print('Classification Report:')
    print(classification_report(test_labels, test_pred, digits=4))

    cm = confusion_matrix(test_labels, test_pred)
    print(f'True Positive Rate [Recall(1)]: {cm[1, 1] / sum(cm[1, :])}')
    print(f'False Positive Rate [1-RCL(0)]: {cm[0, 1] / sum(cm[0, :])}\n')

    print(f'Accuracy with default threshold=50%: {(cm[0, 0] + cm[1, 1]) * 100 / (sum(cm).sum()):.4f}%\n')

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
