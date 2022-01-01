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
#classifier
from xgboost import XGBClassifier
from FairClassifierPipeline.RobustRandomCutForest import RobustRandomCutForest as RRCF
from FairClassifierPipeline import FairPipeline as fair_ppl
import FairClassifierPipeline.Utils as utils
from BaseClassifiers.BaseClf import BaseClf


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


class FairClassifier:
    @staticmethod
    def merge_feature_onehot_columns(feature_name:str, data:pd.DataFrame) -> pd.Series:
        # sensitive_feature_name = config['sensitive_feature']

        sensitive_feature_one_hot_columns = [col for col in data.columns if feature_name in col]

        feature_df = pd.DataFrame(data=[''] * data.shape[0], columns=[feature_name])
        for col in sensitive_feature_one_hot_columns:
            feature_df.loc[(data[col].values == 1)] = col

        # sensitive_test_feature = pd.DataFrame(data=['']*preprocessed_test_data.shape[0], columns=[sensitive_feature_name])
        # for col in sensitive_feature_one_hot_columns:
        #     sensitive_test_feature.loc[preprocessed_test_data[col] == 1] = col

        return feature_df[feature_name]

    @staticmethod
    def get_feature_col_from_preprocessed_data(feature_name:str, data:pd.DataFrame) -> pd.Series:
        sensitive_feature_srs = None
        if feature_name not in data.columns:
            sensitive_feature_srs = FairClassifier.merge_feature_onehot_columns(feature_name=feature_name, data=data)
        else:
            sensitive_feature_srs = data[feature_name]

        return sensitive_feature_srs

    @staticmethod
    def average_odds_difference(y_true: pd.Series,
                                y_pred: pd.Series,
                                sensitive_feature_arr: np.array):

        fpr_diff = abs(false_positive_rate_difference(utils.to_int_srs(y_true), utils.to_int_srs(pd.Series(y_pred)),
                                                      sensitive_features=sensitive_feature_arr))
        tpr_diff = abs(true_positive_rate_difference(utils.to_int_srs(y_true), utils.to_int_srs(pd.Series(y_pred)),
                                                     sensitive_features=sensitive_feature_arr))
        aod_score = 0.5 * (fpr_diff + tpr_diff)

        return aod_score


    @staticmethod
    def get_fairness_score_for_sensitive_features(sensitive_features_names:List[str],
                                                  fairness_metric:str,
                                                  y_true:pd.Series,
                                                  y_pred:pd.Series,
                                                  data:pd.DataFrame):
        fairness_metric = fairness_metric.lower()
        assert fairness_metric.lower() in ['eod','aod'], 'currently support eod and aod only'
        assert isinstance(y_true, pd.Series), 'y_true must be of type pd.Series'
        assert isinstance(y_pred, pd.Series), 'y_pred must be of type pd.Series'
        assert isinstance(data, pd.DataFrame), 'data must be of type pd.DataFrame'

        snsftr_frns_scores_dict = {}
        for ftr in sensitive_features_names:
            sensitive_feature_srs = None
            sensitive_feature_srs = FairClassifier.get_feature_col_from_preprocessed_data(data=data,feature_name=ftr)
            if fairness_metric.lower() == 'eod':
                result = equalized_odds_difference(y_true=y_true,
                                                  y_pred=y_pred,
                                                  sensitive_features=sensitive_feature_srs.values)
            else:
                result = FairClassifier.average_odds_difference( y_true=y_true,
                                                                  y_pred=y_pred,
                                                                  sensitive_feature_arr=sensitive_feature_srs.values)

            snsftr_frns_scores_dict[f'{ftr}:{fairness_metric}'] = result

        return snsftr_frns_scores_dict



    @staticmethod
    def get_feature_sub_groups_by_selection_rate(y_true:pd.Series,
                                                 y_pred:pd.Series,
                                                 sensitive_feature_srs:pd.Series):

        metrics_dict = {"accuracy": accuracy_score, "selection_rate": selection_rate}

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
        metircs_df['sub_groups'] = (metircs_df.selection_rate > selection_rate_threshold) *1

        priv = tuple(metircs_df[metircs_df.sub_groups == 1].index.values)
        unpriv = tuple(metircs_df[metircs_df.sub_groups == 0].index.values)

        return mf2.by_group, (unpriv,priv)


    @staticmethod
    def build_gridsearch_cv_params(X_train_df:pd.DataFrame):
        if_param_grid = {'n_estimators': [100, 150],
                         'max_samples': ['auto', 0.5],
                         'contamination': ['auto'],
                         'max_features': [10, 15],
                         'bootstrap': [True],
                         'n_jobs': [-1]}

        svm_param_grid = {'kernel': ['rbf'],
                          'gamma': ['auto', 1, 0.1, 0.01, 0.001, 0.0001]}

        rc_param_grid = {'random_state': [1]} #todo: change back to 42

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

    @staticmethod
    def get_most_biased_sensitive_feature(data:pd.DataFrame,
                                          fairness_metric:str,
                                          base_clf:BaseClf,
                                          config:Dict):

        fairness_metric = fairness_metric.lower()
        assert fairness_metric in ['aod','eod'], "currently only 'aod' and 'eod' fairness metrics are supported"

        sensitive_features_names = config['sensitive_features']
        sf_fairness_scores_dict = {}
        for sf in sensitive_features_names:
            config['sensitive_feature'] = sf

            ppl, preprocessed_train_data, preprocessed_test_data, X_train, X_test, y_train, y_test = fair_ppl.run_fair_data_preprocess_pipeline(data.copy(), config)

            if sf in X_train.columns and fair_ppl.is_categorial(X_train[sf]) == False:
                print(f'sensitive feature {sf} is not categorical thus removed from sensitive features list')
                sensitive_features_names.remove(sf)
                continue

            y_test = utils.to_int_srs(y_test)
            X_test = utils.to_float_df(X_test)
            xgb_clf, y_pred, y_pred_proba = base_clf.fit_predict(X_train=utils.to_float_df(X_train),
                                                                                    y_train=utils.to_int_srs(y_train),
                                                                                    X_test=X_test)


            sf_fairness_scores_dict.update(FairClassifier.get_fairness_score_for_sensitive_features(sensitive_features_names=[sf],
                                                                                                               fairness_metric=fairness_metric,
                                                                                                               y_true=y_test,
                                                                                                               y_pred=pd.Series(y_pred),
                                                                                                               data=X_test))

        sf_fairness_scores_dict_df = pd.DataFrame.from_dict(sf_fairness_scores_dict, orient='index', columns=[fairness_metric])
        sf_fairness_scores_dict_df = sf_fairness_scores_dict_df.sort_values(ascending=False, by=[fairness_metric])
        print(sf_fairness_scores_dict_df)

        return sf_fairness_scores_dict_df.index[0].split(':')[0]


    @staticmethod
    def run_gridsearch_cv(base_clf:BaseClf,
                          X_train:pd.DataFrame,
                          y_train:pd.Series,
                          target_fairness_metric:str,
                          sensitive_feature_name:str,
                          sensitive_feature_srs:pd.Series,
                          snsftr_slctrt_sub_groups:Tuple[Tuple,Tuple],
                          verbose=False):

        assert target_fairness_metric.lower() in ['aod','eod'], 'eod and aod are currntly the only supported fairness metrics'

        # #################################################################################################################
        # RepeatedStratifiedKFold params
        n_splits = 5
        n_repeats = 5
        random_state = 42 #todo: test other random states for RepeatedStratifiedKFold
        # #################################################################################################################
        adapt_pred_thresh_for_best_f1 = False

        # https://stackoverflow.com/questions/49017257/custom-scoring-on-gridsearchcv-with-fold-dependent-parameter
        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        grid_search_idx = {}
        y_train = utils.to_int_srs(y_train)
        X_train = utils.to_float_df(X_train)
        for train_index, test_index in rskf.split(X_train, y_train):
            grid_search_idx[hash(y_train[test_index].values.tobytes())] = np.copy(test_index)

        def aod_measure(y_true: pd.Series, y_pred:np.ndarray) -> float:
            sensitive_selected_arr = sensitive_feature_srs.values[grid_search_idx[hash(y_true.values.tobytes())]]

            try:
                score = FairClassifier.average_odds_difference(y_true=utils.to_int_srs(y_true),
                                                                  y_pred=utils.to_int_srs(pd.Series(y_pred)),
                                                                  sensitive_feature_arr=sensitive_selected_arr)
            except BaseException as e:
                print(f"Exception raised due to insufficient values for some of the sub groups:\n{pd.Series(sensitive_selected_arr).value_counts()}")
                score = 20

            if verbose:
                print(f'AOD Score:{score}')
            return score


        def f1_measure(y_true:pd.Series, y_pred:np.ndarray) -> float:
            score = f1_score(y_true=utils.to_int_srs(y_true),
                            y_pred=utils.to_int_srs(pd.Series(y_pred)),
                            average='macro')
            if verbose:
                print(f'macro f1 Score:{score}')

            return score


        def eod_measure(y_true: pd.Series, y_pred:np.ndarray) -> float:
            sensitive_selected_arr = sensitive_feature_srs.values[grid_search_idx[hash(y_true.values.tobytes())]]

            try:
                score = equalized_odds_difference(utils.to_int_srs(y_true),
                                                          utils.to_int_srs(pd.Series(y_pred)),
                                                          sensitive_features=sensitive_selected_arr)
            except BaseException as e:
                print(
                    f"Exception raised due to insufficient values for some of the sub groups:\n{pd.Series(sensitive_selected_arr).value_counts()}")
                score = 10

            if verbose:
                print(f'EOD Score:{score}')
            return score

        # def eod_measure(y_true:pd.Series, y_pred:np.ndarray) -> float:
        #     sensitive_selected_arr = sensitive_feature_srs.values[grid_search_idx[hash(y_true.values.tobytes())]]
        #     eod_score = abs(equalized_odds_difference(utils.to_int_srs(y_true),
        #                                               utils.to_int_srs(pd.Series(y_pred)),
        #                                               sensitive_features=sensitive_selected_arr))
        #     print(f'EOD Score:{eod_score}')
        #     return eod_score

        estimator = pipe(steps=[
            ('fairxgboost', FairXGBClassifier())])

        # Define the parameter grid space
        param_grid = {
            'fairxgboost__base_clf':[base_clf],
            'fairxgboost__anomalies_per_to_remove': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],  # 0.1,0.2 !!!!!!!!!!!!!!!!!!
            'fairxgboost__include_sensitive_feature': [True, False],  # False
            'fairxgboost__sensitive_col_name': [sensitive_feature_name],
            'fairxgboost__remove_side': ['only_non_privilaged', 'only_privilaged', 'all'],
            # 'only_privilaged'(A93,A94),'only_non_privilaged'(A91,A92),'all'
            'fairxgboost__data_columns': [tuple(X_train.columns)],
            'fairxgboost__anomaly_model_params': FairClassifier.build_gridsearch_cv_params(X_train),  # global_params_sets !!!!!!!!!!!!!!!!!!!
            'fairxgboost__snsftr_slctrt_sub_groups': [snsftr_slctrt_sub_groups],
            'fairxgboost__verbose': [verbose],
        }

        #best standard metirics for imbalanced data is probably fbeta_<x> (f1 is the base case)
        # https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc#2 <--- IMPORTANT REFERENCE !!!

        # Initializethe CV object
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
        # https://scikit-learn.org/stable/modules/grid_search.html#multimetric-grid-search
        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html
        # https://datascience.stackexchange.com/questions/43793/how-to-get-mean-test-scores-from-gridsearchcv-with-multiple-scorers-scikit-lea
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
        # https://hub.gke2.mybinder.org/user/scikit-learn-scikit-learn-s0vnxwm7/lab/tree/notebooks/auto_examples/model_selection/plot_learning_curve.ipynb
        # https://hub.gke2.mybinder.org/user/scikit-learn-scikit-learn-s0vnxwm7/lab/tree/notebooks/auto_examples/model_selection/plot_roc.ipynb

        scorers_dict = {'eod':make_scorer(eod_measure, greater_is_better=False),
                        'aod':make_scorer(aod_measure, greater_is_better=False),
                        'f1':make_scorer(f1_measure,greater_is_better=True)}
        refit = target_fairness_metric.lower()
        assert refit in scorers_dict.keys(), f"gridsearch_cv's parameter 'refit' == '{target_fairness_metric}' (target_fairness_metric) and must be one of the scorrers names"

        pipe_cv = GridSearchCV(estimator=estimator,
                               param_grid=param_grid,
                               cv=rskf,
                               scoring=scorers_dict,
                               refit = refit,#for multiple scorers specify the scorer string id (in the scorers dictionary) to use
                               return_train_score = False,
                               n_jobs=1)

        t0 = datetime.now()

        pipe_cv.fit(X_train, y_train)

        total_time_secs = datetime.now() -t0
        print(f"Gridsearch_cv total run time: {total_time_secs}")

        if verbose:
            print('#' * 100)
            print(f'Gridsearch_cv Best Params:\n{pipe_cv.best_params_}')
            print('#' * 100)

            # print(f'Predict on X_test:\n{pipe_cv.predict_proba(preprocessed_test_data)}')
            # print('#'*100)
            # print(f'Score:\n{pipe_cv.score(preprocessed_test_data, y_test)}')
            # print('#'*100)


        return pipe_cv


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

    def __init__(self, base_clf:BaseClf=None, anomalies_per_to_remove:float=None, remove_side:str=None,
                 include_sensitive_feature:bool=None, sensitive_col_name:str=None,
                 data_columns:List=None, anomaly_model_params:Dict=None, snsftr_slctrt_sub_groups:Tuple[Tuple,Tuple]=None,
                 verbose:bool=True, do_plots:bool=False):

        self.base_clf = base_clf
        self.anomalies_per_to_remove = anomalies_per_to_remove
        self.include_sensitive_feature = include_sensitive_feature
        self.sensitive_col_name = sensitive_col_name
        self.remove_side = remove_side
        self.data_columns = data_columns
        self.anomaly_model_params = anomaly_model_params
        self.snsftr_slctrt_sub_groups = snsftr_slctrt_sub_groups
        self.verbose = verbose
        self.do_plots = do_plots

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
            return self.data.drop(columns=[self.sensitive_col_name])
        else:
            # remove the sensitive columns (eg: 'statussex_A91')
            return self.data.drop(columns=(list(self.snsftr_slctrt_sub_groups[0])+list(self.snsftr_slctrt_sub_groups[1])))

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"base_clf": self.base_clf,
                "anomalies_per_to_remove": self.anomalies_per_to_remove,
                "include_sensitive_feature": self.include_sensitive_feature,
                "sensitive_col_name": self.sensitive_col_name,
                "remove_side": self.remove_side,
                "data_columns": self.data_columns,
                "anomaly_model_params": self.anomaly_model_params,
                "snsftr_slctrt_sub_groups": self.snsftr_slctrt_sub_groups,
                "verbose": self.verbose,
                "do_plots": self.do_plots}

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
            "SVM": OneClassSVM,
            "LOF": LocalOutlierFactor,
            "RC": EllipticEnvelope,
            "RRCF": RRCF}

        # ##############################################################################################
        # fit models
        algorithm_name = list(self.anomaly_model_params.keys())[0]

        unsupervised_model_params = list(self.anomaly_model_params.values())[0]
        if self.verbose:
            print(f'  Selected algorithm:{algorithm_name}')
            print(f'  Algorithm params:{unsupervised_model_params}')
            print(f'  Grid params: anomalies_per_to_remove:{self.anomalies_per_to_remove}, include_sensitive_feature:{self.include_sensitive_feature}, remove_side:{self.remove_side}')

        algorithm = anomaly_algorithms[algorithm_name](**unsupervised_model_params)
        algorithm.fit(data_arr, self.y_)

        # ##############################################################################################
        # get algorithm's score
        score_series = pd.Series(algorithm.decision_function(data_arr)).sort_values(ascending=True)
        score_idx = list(score_series.index)
        if self.verbose:
            print(f'  Algorithm score:{list(score_series.values)}')
            if self.do_plots:
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

        # print(f'snsftr_slctrt_sub_groups: {self.snsftr_slctrt_sub_groups}')
        # print(f'sensitive config: {self.sensitive_col_name}')
        # sensitive_feature_one_hot_columns = [x for x in self.data.columns if x.startswith(self.sensitive_col_name)]
        # print(f'sensitive columns arr: {sensitive_feature_one_hot_columns}')
        non_privilage_group = self.snsftr_slctrt_sub_groups[0]
        privilage_group = self.snsftr_slctrt_sub_groups[1]
        # print(f'non-privilaged columns: {non_privilage_group}')

        anomalies_idx = set(anomalies_idx)
        filter_sub_sensitive_group = []
        sensitive_feature_col = FairClassifier.get_feature_col_from_preprocessed_data(feature_name=self.sensitive_col_name,
                                                                       data=self.data)
        # print(f'indexes before filtering sensitive: {anomalies_idx}')
        if self.remove_side == 'only_privilaged':
            ## filter the rows of the non-privilaged
            for sf_value in privilage_group:
                # print(f'indexes of non privilaged: {set(list(self.data.index[self.data[idx_np] == 1]))}')
                filter_sub_sensitive_group += (list(self.data.index[sensitive_feature_col == sf_value]))
                # print(f'filter_sub_sensitive_group:{filter_sub_sensitive_group}')
        elif self.remove_side == 'only_non_privilaged':
            ## filter the rows of the privilaged
            for sf_value in non_privilage_group:
                # print(f'indexes of privilaged: {set(list(self.data.index[self.data[idx_p] == 1]))}')
                filter_sub_sensitive_group += (list(self.data.index[sensitive_feature_col == sf_value]))
                # print(f'indexes after filtering sensitive: {anomalies_idx}')
        else:# remove all anomalies
            filter_sub_sensitive_group = anomalies_idx

        anomalies_idx_to_remove = anomalies_idx & set(filter_sub_sensitive_group)

        if self.verbose:
            # print(f'anomalies_idx_to_remove:{anomalies_idx_to_remove}')
            # print(f'statussex_A91:{list(self.data["statussex_A91"].values)}')
            # print(f'statussex_A92:{list(self.data["statussex_A92"].values)}')
            # print(f'statussex_A93:{list(self.data["statussex_A93"].values)}')
            # print(f'statussex_A94:{list(self.data["statussex_A94"].values)}')
            print(f'  Number of anomalies to be removed:{len(anomalies_idx_to_remove)}')

        # https: // xgboost.readthedocs.io / en / stable / python / python_api.html
        self.model = self.base_clf.fit(X_train=self.data.drop(list(anomalies_idx_to_remove)),
                                       y_train=pd.Series(np.delete(self.y_, list(anomalies_idx_to_remove))))


        # self.model = self.XGBClassifier.fit(X=self.data.drop(list(anomalies_idx_to_remove)),
        #                                     y=np.delete(self.y_, list(anomalies_idx_to_remove)),
        #                                     eval_metric='logloss',
        #                                     verbose=False)

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
        y_pred, y_pred_proba = self.base_clf.predict(clf= self.model,
                                                            X = X)

        return y_pred

    def predict_proba(self, X):
        y_predict, y_pred_proba = self.base_clf.predict(clf= self.model,
                                                                X = X)
        return y_pred_proba

    def decision_function(self, X):
        return self.predict_proba(X)
