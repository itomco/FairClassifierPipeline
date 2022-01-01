#Importing required libraries
import json
# from google.colab import drive
# import requests
import zipfile
import pandas as pd
import numpy as np
import sklearn
from typing import *
import datetime
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

class RobustRandomCutForest():
    """
    RobustRandomCutForest.
    The Robust Random Cut Forest (RRCF) algorithm is an ensemble method for detecting outliers in streaming data. RRCF offers a number of features that many competing anomaly detection algorithms lack. Specifically, RRCF:
    Is designed to handle streaming data.
    Performs well on high-dimensional data.
    Reduces the influence of irrelevant dimensions.
    Gracefully handles duplicates and near-duplicates that could otherwise mask the presence of outliers.
    Features an anomaly-scoring algorithm with a clear underlying statistical meaning.
    This repository provides an open-source implementation of the RRCF algorithm and its core data structures for the purposes of facilitating experimentation and enabling future extensions of the RRCF algorithm.
    Parameters
    ----------
    num_trees : int, default=100
        The number of trees.
    tree_size : int, default=512
        The maximum depth of each tree

    Attributes
    ----------

    avg_cod : Compute average CoDisp
              Explanation about CoDisp:
              The likelihood that a point is an outlier is measured by its collusive displacement
              (CoDisp): if including a new point significantly changes the model complexity (i.e. bit depth),
              then that point is more likely to be an outlier.
              tree.codisp('inlier') >>> 1.75
              tree.codisp('outlier') >>> 39.0

    Notes
    -----
    https://github.com/kLabUM/rrcf

    """

    def __init__(self, *, num_trees=100, tree_size=512, contamination:float=0.0):
        # print('>> Init RobustRandomCutForest')
        self.num_trees = num_trees
        self.tree_size = tree_size
        self.contamination = contamination

    def fit(self, X, y=None):
        # do nothing :-)
        # Return the classifier
        return self

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def predict(self, X):
        n = len(X)
        forest = []
        # build trees for RRCF
        while len(forest) < self.num_trees:
            # Select random subsets of points uniformly from point set
            ixs = np.random.choice(n,
                                   size=(1, self.tree_size),  # size=(n // tree_size, tree_size),
                                   replace=True)
            # Add sampled trees to forest
            trees = [rrcf.RCTree(X[ix], index_labels=ix) for ix in ixs]
            forest.extend(trees)

        # Compute average CoDisp
        avg_codisp_d = pd.Series(0.0, index=np.arange(n))
        index = np.zeros(n)
        for tree in forest:
            codisp = pd.Series({leaf: tree.codisp(leaf)
                                for leaf in tree.leaves})
            avg_codisp_d[codisp.index] += codisp
            np.add.at(index, codisp.index.values, 1.0)
        avg_codisp_d /= index
        # self.avg_codisp.append(avg_codisp_d)
        mask=np.percentile(avg_codisp_d,int((1-self.contamination)*100.0))
        avg_codisp_d[avg_codisp_d <= mask] = 1
        avg_codisp_d[avg_codisp_d > mask] = -1

        return avg_codisp_d


