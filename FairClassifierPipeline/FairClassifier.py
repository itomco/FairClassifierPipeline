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
#classifier
from xgboost import XGBClassifier
from FairClassifierPipeline.RobustRandomCutForest import RobustRandomCutForest as RRCF
from FairClassifierPipeline import FairPipeline as fair_ppl
import FairClassifierPipeline.Utils as utils
from BaseClassifiers.BaseClf import BaseClf
from FairClassifierPipeline.FairXGBClassifier import FairXGBClassifier
from FairClassifierPipeline import FairnessUtils as frns_utils
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


class FairClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, target_fairness_metric:str,
                 base_clf:BaseClf, n_splits:int = 5, n_repeats:int = 1, random_state:int = 42, verbose:bool=False):
        self.supported_metrics = ['f1','aod','eod']
        assert target_fairness_metric.lower() in self.supported_metrics, 'eod and aod are currntly the only supported fairness metrics'

        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.target_fairness_metric = target_fairness_metric
        self.base_clf = base_clf
        self.verbose = verbose

        self._pipe_cv = None
        self._is_fitted = False

    @property
    def pipe_cv(self):
        assert self._is_fitted,'FairClassifier is not fitted yet'
        return self._pipe_cv

    @staticmethod
    def build_gridsearch_cv_params(num_samples_in_gridsearch_fold:int):
        # if_param_grid = {'n_estimators': [100, 200, 300],
        #                  'max_samples': ['auto', 0.5],
        #                  'max_features': [5, 10, 15],
        #                  'bootstrap': [True, False],
        #                  'n_jobs': [-1]}
        #
        # svm_param_grid = {'kernel': ['rbf'],
        #                   'gamma': ['auto', 1, 0.1, 0.01, 0.001, 0.0001]}
        #
        # rc_param_grid = {'random_state': [42]}
        #
        # lof_param_grid = {'n_neighbors': [10,20,30,40]}
        #
        # rrcf_param_grid = {'num_trees': [100, 200, 400],
        #                    'tree_size': [min(512, int(num_samples_in_gridsearch_fold/2))]}
        if_param_grid = {'n_estimators': [100],
                         'max_samples': [0.5],
                         'max_features': [10],
                         'bootstrap': [True],
                         'n_jobs': [-1]}

        svm_param_grid = {'kernel': ['rbf'],
                          'gamma': [0.1]}

        rc_param_grid = {'random_state': [42]}

        lof_param_grid = {'n_neighbors': [40]}

        rrcf_param_grid = {'num_trees': [400],
                           'tree_size': [min(512, int(num_samples_in_gridsearch_fold/2))]}

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


            sf_fairness_scores_dict.update(frns_utils.get_fairness_score_for_sensitive_features(sensitive_features_names=[sf],
                                                                                                               fairness_metric=fairness_metric,
                                                                                                               y_true=y_test,
                                                                                                               y_pred=pd.Series(y_pred),
                                                                                                               data=X_test))

        sf_fairness_scores_dict_df = pd.DataFrame.from_dict(sf_fairness_scores_dict, orient='index', columns=[fairness_metric])
        sf_fairness_scores_dict_df = sf_fairness_scores_dict_df.sort_values(ascending=False, by=[fairness_metric])
        print(sf_fairness_scores_dict_df)

        return sf_fairness_scores_dict_df.index[0].split(':')[0]

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"n_splits": self.n_splits,
                "n_repeats": self.n_repeats,
                "random_state": self.random_state,
                "target_fairness_metric": self.target_fairness_metric,
                "base_clf": self.base_clf,
                "verbose": self.verbose}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self




    def fit(self,
            X_train:pd.DataFrame,
            y_train:pd.Series,
            sensitive_feature_name:str,
            snsftr_slctrt_sub_groups:Tuple[Tuple,Tuple],
            ):

        self.sensitive_feature_name = sensitive_feature_name
        self.sensitive_feature_srs = frns_utils.get_feature_col_from_preprocessed_data(feature_name=sensitive_feature_name,
                                                                                data= X_train)
        self.snsftr_slctrt_sub_groups = snsftr_slctrt_sub_groups


        data_portion_in_fold = 1-(1.0/self.n_splits)

        # #################################################################################################################

        # https://stackoverflow.com/questions/49017257/custom-scoring-on-gridsearchcv-with-fold-dependent-parameter
        rskf = RepeatedStratifiedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.random_state)
        grid_search_idx = {}
        y_train = utils.to_int_srs(y_train)
        X_train = utils.to_float_df(X_train)
        for train_index, test_index in rskf.split(X_train, y_train):
            grid_search_idx[hash(y_train[test_index].values.tobytes())] = np.copy(test_index)

        # def eod_measure(y_true:pd.Series, y_pred:np.ndarray) -> float:
        #     sensitive_selected_arr = sensitive_feature_srs.values[grid_search_idx[hash(y_true.values.tobytes())]]
        #     eod_score = abs(equalized_odds_difference(utils.to_int_srs(y_true),
        #                                               utils.to_int_srs(pd.Series(y_pred)),
        #                                               sensitive_features=sensitive_selected_arr))
        #     print(f'EOD Score:{eod_score}')
        #     return eod_score


        # Define the parameter grid space
        # param_grid = {
        #     'fairxgboost__base_clf':[self.base_clf],
        #     'fairxgboost__anomalies_per_to_remove': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],  # 0.1,0.2 !!!!!!!!!!!!!!!!!!
        #     'fairxgboost__include_sensitive_feature': [True, False],  # False
        #     'fairxgboost__sensitive_col_name': [self.sensitive_feature_name],
        #     'fairxgboost__remove_side': ['only_non_privilaged', 'only_privilaged', 'all'],
        #     # 'only_privilaged'(A93,A94),'only_non_privilaged'(A91,A92),'all'
        #     'fairxgboost__data_columns': [tuple(X_train.columns)],
        #     'fairxgboost__anomaly_model_params': FairClassifier.build_gridsearch_cv_params(num_samples_in_gridsearch_fold= int(data_portion_in_fold*X_train.shape[0])),
        #     'fairxgboost__snsftr_slctrt_sub_groups': [self.snsftr_slctrt_sub_groups],
        #     'fairxgboost__verbose': [self.verbose],
        # }
        param_grid = {
            'fairxgboost__base_clf':[self.base_clf],
            'fairxgboost__anomalies_per_to_remove': [0.35],  # 0.1,0.2 !!!!!!!!!!!!!!!!!!
            'fairxgboost__include_sensitive_feature': [True],  # False
            'fairxgboost__sensitive_col_name': [self.sensitive_feature_name],
            'fairxgboost__remove_side': ['all'],
            # 'only_privilaged'(A93,A94),'only_non_privilaged'(A91,A92),'all'
            'fairxgboost__data_columns': [tuple(X_train.columns)],
            'fairxgboost__anomaly_model_params': FairClassifier.build_gridsearch_cv_params(num_samples_in_gridsearch_fold= int(data_portion_in_fold*X_train.shape[0])),
            'fairxgboost__snsftr_slctrt_sub_groups': [self.snsftr_slctrt_sub_groups],
            'fairxgboost__verbose': [self.verbose],
        }
        num_iters = self.n_repeats * self.n_splits
        for params_set in param_grid.values():
            num_iters *= len(params_set)
        print(f"Number of GridSearch fits: {num_iters}")


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
        pipe_cv = None

        with tqdm(total=num_iters) as pbar:

            def aod_measure(y_true: pd.Series, y_pred:np.ndarray) -> float:
                sensitive_selected_arr = self.sensitive_feature_srs.values[grid_search_idx[hash(y_true.values.tobytes())]]

                try:
                    score = frns_utils.average_odds_difference(y_true=utils.to_int_srs(y_true),
                                                                      y_pred=utils.to_int_srs(pd.Series(y_pred)),
                                                                      sensitive_feature_arr=sensitive_selected_arr)
                except BaseException as e:
                    print(f"Exception raised due to insufficient values for some of the sub groups:\n{pd.Series(sensitive_selected_arr).value_counts()}")
                    score = 20

                if self.verbose:
                    print(f'AOD Score:{score}')
                return score


            def f1_measure(y_true:pd.Series, y_pred:np.ndarray) -> float:
                score = f1_score(y_true=utils.to_int_srs(y_true),
                                y_pred=utils.to_int_srs(pd.Series(y_pred)),
                                average='macro')
                if self.verbose:
                    print(f'macro f1 Score:{score}')

                return score


            def eod_measure(y_true: pd.Series, y_pred:np.ndarray) -> float:
                sensitive_selected_arr = self.sensitive_feature_srs.values[grid_search_idx[hash(y_true.values.tobytes())]]

                try:
                    score = equalized_odds_difference(utils.to_int_srs(y_true),
                                                              utils.to_int_srs(pd.Series(y_pred)),
                                                              sensitive_features=sensitive_selected_arr)
                except BaseException as e:
                    print(
                        f"Exception raised due to insufficient values for some of the sub groups:\n{pd.Series(sensitive_selected_arr).value_counts()}")
                    score = 10

                if self.verbose:
                    print(f'EOD Score:{score}')

                pbar.update(1)

                return score


            scorers_dict = {'eod':make_scorer(eod_measure, greater_is_better=False),
                            'aod':make_scorer(aod_measure, greater_is_better=False),
                            'f1':make_scorer(f1_measure,greater_is_better=True)}
            refit = self.target_fairness_metric.lower()
            assert refit in scorers_dict.keys(), f"gridsearch_cv's parameter 'refit' == '{self.target_fairness_metric}' (target_fairness_metric) and must be one of the scorrers names"

            estimator = pipe(steps=[
                ('fairxgboost', FairXGBClassifier())])

            self._pipe_cv = GridSearchCV(estimator=estimator,
                                   param_grid=param_grid,
                                   cv=rskf,
                                   scoring=scorers_dict,
                                   refit = refit,#for multiple scorers specify the scorer string id (in the scorers dictionary) to use
                                   return_train_score = False,
                                   n_jobs=1)

            t0 = datetime.now()

            # gridsearch_cv fit progress bar: https://towardsdatascience.com/progress-bars-in-python-4b44e8a4c482
            self._pipe_cv.fit(X_train, y_train)

            total_time_secs = datetime.now() -t0
        print(f"Gridsearch_cv total run time: {total_time_secs}")

        if self.verbose:
            print('#' * 100)
            print(f'Gridsearch_cv Best Params:\n{self._pipe_cv.best_params_}')
            print('#' * 100)

            # print(f'Predict on X_test:\n{pipe_cv.predict_proba(preprocessed_test_data)}')
            # print('#'*100)
            # print(f'Score:\n{pipe_cv.score(preprocessed_test_data, y_test)}')
            # print('#'*100)

        self._is_fitted = True

        return(self)

    def retrain_top_models_and_get_performance_metrics(  self, X_train:pd.DataFrame,
                                                         y_train:pd.Series,
                                                         X_test:pd.DataFrame,
                                                         y_test:pd.Series,
                                                         target_metrics_thresholds:Dict,
                                                         performance_metrics:List=['aod','eod','f1'],
                                                         max_num_top_models:int=100,
                                                         verbose:bool=False):
        '''

        :param target_metric_name: The name of the performance (/fiarness) metric by which to find the top models in self.pipe_cv.results_
        :param metric_value_threshold: The threshold above/below which to find top models from self.pipe_cv.results_
        :param greater_target_metric_value_is_better: If True, models with score value above the metric_value_threshold are considered. Otherwise if False.
        :param max_num_top_models: Maximum top models to retrain
        :return:List[{model_index_in_pipe_cv:value,metric_name:value,..}]
        '''
        assert self._is_fitted, 'cannot perform refit before fitting'
        performance_metrics = [x.lower() for x in performance_metrics]
        not_supported_metrics = set(performance_metrics) - set(self.supported_metrics)
        assert len(not_supported_metrics) == 0, f'the performance_metrics are not supported: {not_supported_metrics}'
        assert (0 <= np.array(list(target_metrics_thresholds.values()))).all() and (np.array(list(target_metrics_thresholds.values())) <=1).all(), 'target_metrics_thresholds can get values between 0 to 1'

        if max_num_top_models > len(self._pipe_cv.cv_results_):
            max_num_top_models = len(self._pipe_cv.cv_results_)

        top_models_results = pd.DataFrame(self._pipe_cv.cv_results_)
        for target_metric_name, target_metric_value_threshold in target_metrics_thresholds.items():
            target_metric_col = f'mean_test_{target_metric_name.lower()}'

            greater_target_metric_value_is_better = bool(top_models_results[target_metric_col].mean() > 0)
            ascending = False if greater_target_metric_value_is_better else True

            if greater_target_metric_value_is_better:
                top_models_results = top_models_results.loc[top_models_results[target_metric_col] > target_metric_value_threshold]
            if (greater_target_metric_value_is_better == False):
                top_models_results = top_models_results.loc[-1.0*top_models_results[target_metric_col] > target_metric_value_threshold]

            top_models_results = top_models_results.sort_values(by=target_metric_col,ascending=ascending).head(max_num_top_models)

        include_sensitive_feature_col = None
        for col in top_models_results.columns:
            if col.endswith('include_sensitive_feature'):
                include_sensitive_feature_col = col
                break
        assert include_sensitive_feature_col is not None, 'could not find include_sensitive_feature column'

        if top_models_results.shape[0] == 0:
            print("there was no model who mached the target metrics threshold specified")
        elif verbose:
            print(f"Start computing models performance results for {top_models_results.shape[0]} top models ...")

        non_privilage_group = self.snsftr_slctrt_sub_groups[0]
        privilage_group = self.snsftr_slctrt_sub_groups[1]
        X_test_sensitive_feature_srs = frns_utils.get_feature_col_from_preprocessed_data(feature_name=self.sensitive_feature_name,
                                                                                        data=X_test)

        result = []
        for index in list(top_models_results.index):
            #1. remove sensitive feature if required
            include_sensitive_feature = bool(top_models_results[include_sensitive_feature_col][index])

            X_train_to_use = X_train.copy()
            if include_sensitive_feature == False:
                X_train_to_use = frns_utils.drop_sensitive_features(  sensitive_col_name=self.sensitive_feature_name,
                                                                      data=X_train,
                                                                      snsftr_slctrt_sub_groups=self.snsftr_slctrt_sub_groups)
                # X_train_to_use = X_train.drop(columns=self.sensitive_feature_name)

            #2. perform anomaly semples detection and removal strategy on X_train_to_use
            anomaly_model_params = top_models_results['param_fairxgboost__anomaly_model_params'][index]
            anomalies_per_to_remove = top_models_results['param_fairxgboost__anomalies_per_to_remove'][index]
            remove_side = top_models_results['param_fairxgboost__remove_side'][index]

            anomaly_algorithms = {
                "IF": IsolationForest,
                "SVM": OneClassSVM,
                "LOF": LocalOutlierFactor,
                "RC": EllipticEnvelope,
                "RRCF": RRCF}
            algorithm_name = list(anomaly_model_params.keys())[0]
            anomaly_model_params = list(anomaly_model_params.values())[0]
            if verbose:
                print(f'  Selected algorithm:{algorithm_name}')
                print(f'  Algorithm params:{anomaly_model_params}')
                print(f'  Grid params: anomalies_per_to_remove:{anomalies_per_to_remove}, include_sensitive_feature:{include_sensitive_feature}, remove_side:{remove_side}')

            if algorithm_name.lower() == 'svm':
                anomaly_model_params['nu'] = anomalies_per_to_remove
            else:
                anomaly_model_params['contamination'] = anomalies_per_to_remove

            algorithm = anomaly_algorithms[algorithm_name](**anomaly_model_params)

            # #############################################################################################
            # get anomalies by prediction
            if algorithm_name.lower() == 'lof':
                anomalies_idx = np.where(algorithm.fit_predict(X_train_to_use.values) == -1)[0].tolist()
            else:
                algorithm.fit(X_train_to_use.values)
                anomalies_idx = np.where(algorithm.predict(X_train_to_use.values) == -1)[0].tolist()

            anomalies_idx = set(anomalies_idx)
            filter_sub_sensitive_group = []
            if remove_side == 'only_privilaged':
                ## filter the rows of the non-privilaged
                for sf_value in privilage_group:
                    # print(f'indexes of non privilaged: {set(list(self.data.index[self.data[idx_np] == 1]))}')
                    filter_sub_sensitive_group += (list(X_train_to_use.index[self.sensitive_feature_srs == sf_value]))
                    # print(f'filter_sub_sensitive_group:{filter_sub_sensitive_group}')
            elif remove_side == 'only_non_privilaged':
                ## filter the rows of the privilaged
                for sf_value in non_privilage_group:
                    # print(f'indexes of privilaged: {set(list(self.data.index[self.data[idx_p] == 1]))}')
                    filter_sub_sensitive_group += (list(X_train_to_use.index[self.sensitive_feature_srs == sf_value]))
                    # print(f'indexes after filtering sensitive: {anomalies_idx}')
            else:  # remove all anomalies
                filter_sub_sensitive_group = anomalies_idx

            anomalies_idx_to_remove = anomalies_idx & set(filter_sub_sensitive_group)

            #3. fit the base model
            clf = self.base_clf.fit(X_train=X_train_to_use.drop(list(anomalies_idx_to_remove)),
                                    y_train=pd.Series(np.delete(y_train.values, list(anomalies_idx_to_remove))))

            #4. make prediction
            y_pred = self.base_clf.predict(clf=clf, X=X_test)[0]

            #5. create performance results
            model_preformance_results = {'index':index}

            if 'eod' in performance_metrics:
                model_preformance_results['eod'] = equalized_odds_difference(y_true=y_test,
                                                       y_pred=utils.to_int_srs(pd.Series(y_pred)),
                                                       sensitive_features=X_test_sensitive_feature_srs.values)
            if 'aod' in performance_metrics:
                model_preformance_results['aod'] = frns_utils.average_odds_difference(y_true=y_test,
                                                       y_pred=utils.to_int_srs(pd.Series(y_pred)),
                                                       sensitive_feature_arr=X_test_sensitive_feature_srs.values)
            if 'f1' in performance_metrics:
                model_preformance_results['f1'] = f1_score(y_true=y_test,
                                                            y_pred=utils.to_int_srs(pd.Series(y_pred)),
                                                            average='macro')

            if verbose:
                print(f"top model #{index} performance results: {model_preformance_results}")

            result.append(model_preformance_results)

        return result

    def predict(self, X_test:pd.DataFrame) -> pd.Series:
        check_is_fitted(self, '_is_fitted')

        return pd.Series(self._pipe_cv.predict(X=X_test))

