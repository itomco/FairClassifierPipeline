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

def merge_feature_onehot_columns(X_train_df:pd.DataFrame, feature_name:str) -> pd.Series:
    # sensitive_feature_name = config['sensitive_feature']

    sensitive_feature_one_hot_columns = [col for col in X_train_df.columns if feature_name in col]

    feature_df = pd.DataFrame(data=[''] * X_train_df.shape[0], columns=[feature_name])
    for col in sensitive_feature_one_hot_columns:
        feature_df.loc[(X_train_df[col].values == 1)] = col

    # sensitive_test_feature = pd.DataFrame(data=['']*preprocessed_test_data.shape[0], columns=[sensitive_feature_name])
    # for col in sensitive_feature_one_hot_columns:
    #     sensitive_test_feature.loc[preprocessed_test_data[col] == 1] = col

    return feature_df[feature_name]

def get_eod_for_sensitive_features(sensitive_features_names:List[str],
                                   y_true:pd.Series,
                                   y_pred:pd.Series,
                                   X_pd:pd.DataFrame):
    snsftr_eod_dict = {}
    for ftr in sensitive_features_names:
        sensitive_feature_srs = None
        if ftr not in X_pd.columns:
            sensitive_feature_srs = merge_feature_onehot_columns(X_train_df=X_pd,feature_name=ftr)
        else:
            sensitive_feature_srs = X_pd[ftr]

        eod = equalized_odds_difference(y_true=np.array(y_true.values,dtype=int),
                                  y_pred=np.array(y_pred.values,dtype=int),
                                  sensitive_features=sensitive_feature_srs.values)
        snsftr_eod_dict[ftr] = eod

    return snsftr_eod_dict


def get_feature_sub_groups_by_selection_rate(clf_model, X_train_df:pd.DataFrame, y_train:pd.Series, sensitive_feature_srs:pd.Series):
    metrics_dict = {"accuracy": accuracy_score, "selection_rate": selection_rate}
    y_pred = clf_model.predict(X_train_df.astype(float).values)
    # y_pred_proba = model_baseline.predict_proba(X_train, ntree_limit=model_baseline.best_ntree_limit)[:, 1]

    mf2 = MetricFrame(
        metrics=metrics_dict,
        y_true=np.array(y_train.values,dtype=int),
        y_pred=y_pred,
        sensitive_features=sensitive_feature_srs.values)

    # mf2.by_group
    metircs_df = mf2.by_group.reset_index()
    metircs_df = metircs_df.sort_values(by=['selection_rate'])
    priv = list(metircs_df[metircs_df.index + 1 > len(metircs_df) / 2].sensitive_feature_0)
    unpriv = list(metircs_df[metircs_df.index + 1 <= len(metircs_df) / 2].sensitive_feature_0)

    return mf2.by_group, (tuple(unpriv),tuple(priv))


def build_gridsearch_cv_params(X_train_df:pd.DataFrame):
    if_param_grid = {'n_estimators': [100, 150],
                     'max_samples': ['auto', 0.5],
                     'contamination': ['auto'],
                     'max_features': [10, 15],
                     'bootstrap': [True],
                     'n_jobs': [-1]}

    svm_param_grid = {'kernel': ['rbf'],
                      'gamma': ['auto', 1, 0.1, 0.01, 0.001, 0.0001]}

    rc_param_grid = {'random_state': [42]}

    lof_param_grid = {'n_neighbors': [10, 20, 30],
                      'novelty': [True]}

    rrcf_param_grid = {'num_trees': [100],
                       'tree_size': [min(512, int(((X_train_df.shape[0]) * 0.8) / 2))]}

    def dict_product(dicts):
        return (dict(zip(dicts.keys(), x)) for x in itertools.product(*dicts.values()))

    params_sets = []
    models_params_grid = {'IF': if_param_grid, 'SVM': svm_param_grid, 'RC': rc_param_grid, 'LOF': lof_param_grid,
                          'RRCF': rrcf_param_grid}

    for model_name in models_params_grid:
        for params_set in list(dict_product(models_params_grid[model_name])):
            params_sets.append({model_name: params_set})

    return params_sets

def build_gridsearch_cv(X_train_df:pd.DataFrame,
                        y_train:pd.Series,
                        sensitive_feature_name:str,
                        sensitive_feature_srs:pd.Series,
                        snsftr_slctrt_sub_groups:Tuple[Tuple,Tuple]):
    # #################################################################################################################
    # RepeatedStratifiedKFold params
    n_splits = 5
    n_repeats = 1
    random_state = 42
    # #################################################################################################################

    # https://stackoverflow.com/questions/49017257/custom-scoring-on-gridsearchcv-with-fold-dependent-parameter
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    grid_search_idx = {}
    for train_index, test_index in rskf.split(X_train_df.values, y_train.values):
        grid_search_idx[hash(y_train[test_index].tobytes())] = np.copy(test_index)

    def f2_measure(y_true, y_pred):
        return fbeta_score(y_true, y_pred, beta=2)

    def f1_measure(y_true, y_pred):
        return f1_score(y_true, y_pred)

    def eod_measure(y_true, y_pred):
        sensitivie_selected_arr = sensitive_feature_srs.values[grid_search_idx[hash(y_true.tobytes())]]
        eod_score = abs(equalized_odds_difference(y_true,
                                                  y_pred,
                                                  sensitive_features=sensitivie_selected_arr))

        print(f'EOD Score:{eod_score}')
        return eod_score

    estimator = pipe(steps=[
        ('fairxgboost', FairXGBClassifier())])

    # Define the parameter grid space
    param_grid = {
        'fairxgboost__anomalies_per_to_remove': [0.2, 0.3],  # 0.1,0.2 !!!!!!!!!!!!!!!!!!
        'fairxgboost__include_sensitive_feature': [True, False],  # False
        'fairxgboost__sensitive_col_name': sensitive_feature_name,
        'fairxgboost__remove_side': ['only_non_privilaged', 'only_privilaged', 'all'],
        # 'only_privilaged'(A93,A94),'only_non_privilaged'(A91,A92),'all'
        'fairxgboost__data_columns': [tuple(X_train_df.columns)],
        'fairxgboost__anomaly_model_params': build_gridsearch_cv_params(X_train_df),  # global_params_sets !!!!!!!!!!!!!!!!!!!
        'fairxgboost__sensitive_privilage_group': [snsftr_slctrt_sub_groups],
        'fairxgboost__verbose': [True],
    }

    # Initialize the CV object
    pipe_cv = GridSearchCV(estimator=estimator,
                           param_grid=param_grid,
                           cv=rskf,
                           scoring={"AUC": "roc_auc",
                                    "F1": make_scorer(f1_measure, greater_is_better=True),
                                    "EOD": make_scorer(eod_measure, greater_is_better=False)},
                           refit="EOD",
                           n_jobs=1)


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

    def __init__(self, *, num_trees=100, tree_size=512):
        print('>> Init RobustRandomCutForest')
        self.num_trees = num_trees
        self.tree_size = tree_size

    def fit(self, X, y=None):
        """
        Fit estimator.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Use ``dtype=np.float32`` for maximum
            efficiency. Sparse matrices are also supported, use sparse
            ``csc_matrix`` for maximum efficiency.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        print('>> fit RobustRandomCutForest')
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        # self.classes_ = unique_labels(y)

        # RRCF parameters
        self.avg_codisp = []
        self.epsilon = 0.0001

        n = len(X)
        forest = []
        # build trees for RRCF
        while len(forest) < self.num_trees:
            # Select random subsets of points uniformly from point set
            ixs = np.random.choice(n,
                                   size=(1, self.tree_size),  # size=(n // tree_size, tree_size),
                                   replace=True)
            # Add sampled trees to forest
            trees = [rrcf.RCTree(X[ix] + self.epsilon,
                                 index_labels=ix) for ix in ixs]
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
        self.avg_codisp.append(avg_codisp_d)

        # define a threshold for label 1 and 0 in RRCF algorithm
        self.avg_cod = self.avg_codisp[-1]

        # Return the classifier
        return self

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def predict(self, X):
        """
        Predict if a particular sample is an outlier or not.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        Returns
        -------
        is_inlier : ndarray of shape (n_samples,)
            For each observation, tells whether or not (+1 or -1) it should
            be considered as an inlier according to the fitted model.
        """
        check_is_fitted(self)
        decision_func = self.decision_function(X)
        is_inlier = np.ones_like(decision_func, dtype=int)
        is_inlier[decision_func < 0] = -1
        return is_inlier

    def decision_function(self, X):
        """
        Average anomaly score of CoDisp.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The anomaly score of the input samples.
            The lower, the more abnormal. Negative scores represent outliers,
            positive scores represent inliers.
        """
        print('>> decision_function RobustRandomCutForest')
        return self.avg_cod


class FairXGBClassifier(ClassifierMixin, BaseEstimator):
    """ An FairXGBClassifier which implements a Fair XGBoost algorithm.

    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.
    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """

    def __init__(self, anomalies_per_to_remove=None, remove_side=None,
                 include_sensitive_feature=None, sensitive_col_name=None,
                 data_columns=None, anomaly_model_params=None, sensitive_privilage_group=None,
                 verbose=False):

        # XGBClassifier parameters as 1b
        self.XGBClassifier_params = {
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

        self.XGBClassifier = XGBClassifier(**self.XGBClassifier_params)

        self.anomalies_per_to_remove = anomalies_per_to_remove
        self.include_sensitive_feature = include_sensitive_feature
        self.sensitive_col_name = sensitive_col_name
        self.remove_side = remove_side
        self.data_columns = data_columns
        self.anomaly_model_params = anomaly_model_params
        self.sensitive_privilage_group = sensitive_privilage_group
        self.verbose = verbose

    def __plot_histogram(self, score_input):
        n, bins, patches = plt.hist(x=score_input, bins='auto', color='#0504aa',
                                    alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Score Histogram')
        maxfreq = n.max()
        # Set a clean upper y-axis limit.
        plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

    def __drop_sensitive_features(self):
        # remove the sensitive name (eg: 'sex')
        if self.sensitive_col_name in self.data.columns:
            return data.drop(columns=[self.sensitive_col_name])
        else:
            # remove the sensitive columns (eg: 'statussex_A91')
            onehot_sensitive_feature_cols = []
            for col in self.data.columns:
                if col.startswith(self.sensitive_col_name):
                    onehot_sensitive_feature_cols.append(col)
            return self.data.drop(columns=onehot_sensitive_feature_cols)

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"anomalies_per_to_remove": self.anomalies_per_to_remove,
                "include_sensitive_feature": self.include_sensitive_feature,
                "sensitive_col_name": self.sensitive_col_name,
                "remove_side": self.remove_side,
                "data_columns": self.data_columns,
                "anomaly_model_params": self.anomaly_model_params,
                "sensitive_privilage_group": self.sensitive_privilage_group,
                "verbose": self.verbose}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.
        Returns
        -------
        self : object
            Returns self.
        """

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        self.X_, self.y_ = X, y

        # ##################################################################################################
        if self.verbose:
            print('\n>>>Remove Anomalies fit()')

        # build dataframe base on the data_columns input and X
        self.data = pd.DataFrame(data=self.X_, columns=self.data_columns)

        # ##############################################################################################
        # Check improvement in drops the sensitive columns
        # ##############################################################################################
        if self.include_sensitive_feature == False:
            data_arr = self.__drop_sensitive_features()

        data_arr = self.data.to_numpy()

        # ##############################################################################################
        # Outlier detectors from sklean and RRCF
        anomaly_algorithms = {
            "IF": IsolationForest,
            "SVM": svm.OneClassSVM,
            "LOF": LocalOutlierFactor,
            "RC": EllipticEnvelope,
            "RRCF": RobustRandomCutForest}

        # ##############################################################################################
        # fit models
        algorithm_name = list(self.anomaly_model_params.keys())[0]

        unsupervised_model_params = list(self.anomaly_model_params.values())[0]
        if self.verbose:
            print(f'  Selected algorithm:{algorithm_name}')
            print(f'  Algorithm params:{unsupervised_model_params}')
            print(
                f'  Grid params: anomalies_per_to_remove:{self.anomalies_per_to_remove},include_sensitive_feature:{self.include_sensitive_feature},remove_side:{self.remove_side}')

        algorithm = anomaly_algorithms[algorithm_name](**unsupervised_model_params)
        algorithm.fit(data_arr, self.y_)

        # ##############################################################################################
        # get algorithm's score
        score_series = pd.Series(algorithm.decision_function(data_arr)).sort_values(ascending=True)
        score_idx = list(score_series.index)
        if self.verbose:
            print(f'  Algorithm score:{list(score_series.values)}')
            self.__plot_histogram(list(score_series.values))
        # ##############################################################################################
        # remove anomalies by using the selected unsupervised algorithm
        num_remove_samples = int(len(self.data) * self.anomalies_per_to_remove)
        anomalies_idx = None
        if algorithm_name == 'RRCF':
            anomalies_idx = score_idx[(-1 * num_remove_samples):]
        else:
            anomalies_idx = score_idx[:num_remove_samples]

        # if self.verbose:
        #     print(f'anomalies score:{list(score_series[anomalies_idx].values)}')
        # print(f'anomalies indexes:{anomalies_idx}')
        # print(f'{len(anomalies_idx)} anomalies, non anomalies {len(self.data) - len(anomalies_idx)}')
        # print(self.y_)
        # print(np.delete(self.y_, anomalies_idx))
        # ##############################################################################################
        # Run XGBoost without anomalies

        # print(f'sensitive_privilage_group: {self.sensitive_privilage_group}')
        # print(f'sensitive config: {self.sensitive_col_name}')
        sensitive_feature_one_hot_columns = [x for x in self.data.columns if x.startswith(self.sensitive_col_name)]
        # print(f'sensitive columns arr: {sensitive_feature_one_hot_columns}')
        non_privilage_group = list(set(sensitive_feature_one_hot_columns) - set(self.sensitive_privilage_group))
        # print(f'non-privilaged columns: {non_privilage_group}')

        anomalies_idx = set(anomalies_idx)
        filter_sub_sensitive_group = []
        # print(f'indexes before filtering sensitive: {anomalies_idx}')
        if self.remove_side == 'only_privilaged':
            ## filter the rows of the non-privilaged
            for idx_np in self.sensitive_privilage_group:
                # print(f'indexes of non privilaged: {set(list(self.data.index[self.data[idx_np] == 1]))}')
                filter_sub_sensitive_group += (list(self.data.index[self.data[idx_np] == 1]))
                # print(f'filter_sub_sensitive_group:{filter_sub_sensitive_group}')
        elif self.remove_side == 'only_non_privilaged':
            ## filter the rows of the privilaged
            for idx_p in non_privilage_group:
                # print(f'indexes of privilaged: {set(list(self.data.index[self.data[idx_p] == 1]))}')
                filter_sub_sensitive_group += (list(self.data.index[self.data[idx_p] == 1]))
                # print(f'indexes after filtering sensitive: {anomalies_idx}')

        anomalies_idx_to_remove = anomalies_idx & set(filter_sub_sensitive_group)

        if self.verbose:
            # print(f'anomalies_idx_to_remove:{anomalies_idx_to_remove}')
            # print(f'statussex_A91:{list(self.data["statussex_A91"].values)}')
            # print(f'statussex_A92:{list(self.data["statussex_A92"].values)}')
            # print(f'statussex_A93:{list(self.data["statussex_A93"].values)}')
            # print(f'statussex_A94:{list(self.data["statussex_A94"].values)}')
            print(f'  Number of anomalies to be removed:{len(anomalies_idx_to_remove)}')

        self.model = self.XGBClassifier.fit(self.data.drop(list(anomalies_idx_to_remove)),
                                            np.delete(self.y_, list(anomalies_idx_to_remove)))

        # Return the classifier
        return self

    def predict(self, X):
        """ Predict using the classifier.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        self.y_pred = self.model.predict(X)

        return self.y_pred

    def predict_proba(self, X):
        self.y_pred_proba = self.model.predict_proba(X)

        return self.y_pred_proba

    def decision_function(self, X):

        return self.predict_proba(X)
