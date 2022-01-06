#Import required libraries
import pandas as pd
import numpy as np
from typing import *
from datetime import datetime

np.random.seed(sum(map(ord, "aesthetics")))

from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline as pipe
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer

from fairlearn.metrics import (
    false_positive_rate_difference,
    true_positive_rate_difference,
)
#classifier
from FairClassifierPipeline import FairPipeline as fair_ppl
import FairClassifierPipeline.Utils as utils
from BaseClassifiers.BaseClf import BaseClf
from FairClassifierPipeline.FairXGBClassifier import FairXGBClassifier
from FairClassifierPipeline import FairnessUtils as frns_utils
from pprint import pprint

#import the proper progress bar - there are different types for iphthon (i.e. jupyter) and simple python interpreter
# from tqdm.notebook import tqdm
# tqdm().pandas()
from tqdm import tqdm

#fairlearn
from fairlearn.metrics import (
    equalized_odds_difference
)
import itertools


class FairClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, target_fairness_metric:str,
                 base_clf:BaseClf, n_splits:int = 5, n_repeats:int = 1, random_state:int = 42, verbose:bool=False):
        self.supported_metrics = ['f1_macro','aod','eod']
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

        #anomaly detection models' Gridsearch params FOR FAST DEBUG
        # if_param_grid = {'n_estimators': [100],
        #                  'max_samples': [0.5],
        #                  'max_features': [10],
        #                  'bootstrap': [True],
        #                  'n_jobs': [-1],
        #                  'random_state':[42]}
        #
        # svm_param_grid = {'kernel': ['rbf'],
        #                   'gamma': [0.1]}
        #
        # rc_param_grid = {'random_state': [42]}
        #
        # lof_param_grid = {'n_neighbors': [40]}
        #
        # rrcf_param_grid = {'num_trees': [400],
        #                    'tree_size': [min(512, int(num_samples_in_gridsearch_fold/2))]}

        # FINAL SLIM anomaly detection models' Gridsearch params
        if_param_grid = {'n_estimators': [200],
                         'max_features': [10],
                         'n_jobs': [-1],
                         'random_state':[42]}

        svm_param_grid = {'kernel': ['rbf'],
                          'gamma': [0.0001, 0.01, 1]}

        rc_param_grid = {'random_state': [42]}

        lof_param_grid = {'n_neighbors': [10, 30]}

        rrcf_param_grid = {'num_trees': [200, 400],
                           'tree_size': [min(512, int(num_samples_in_gridsearch_fold / 2))]}

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

        #select the sensitive feature's larger sub-group to include in the 'remove_side' strategy
        remove_side_list = ['only_privilaged']
        if self.snsftr_slctrt_sub_groups[0][0] > self.snsftr_slctrt_sub_groups[1][0]:
            remove_side_list = ['only_non_privilaged']
        remove_side_list.append('all')

        if self.verbose:
            print(f"Final remove_side policy: {remove_side_list}")

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
        #     'fairxgboost__snsftr_slctrt_sub_groups': [(self.snsftr_slctrt_sub_groups[0][1],self.snsftr_slctrt_sub_groups[1][1])],
        #     'fairxgboost__verbose': [self.verbose],
        # }

        # param_grid = {
        #     'fairxgboost__base_clf':[self.base_clf],
        #     'fairxgboost__anomalies_per_to_remove': [0.35],  # 0.1,0.2 !!!!!!!!!!!!!!!!!!
        #     'fairxgboost__include_sensitive_feature': [True],  # False
        #     'fairxgboost__sensitive_col_name': [self.sensitive_feature_name],
        #     'fairxgboost__remove_side': ['all'],
        #     # 'only_privilaged'(A93,A94),'only_non_privilaged'(A91,A92),'all'
        #     'fairxgboost__data_columns': [tuple(X_train.columns)],
        #     'fairxgboost__anomaly_model_params': [FairClassifier.build_gridsearch_cv_params(num_samples_in_gridsearch_fold= int(data_portion_in_fold*X_train.shape[0]))[0]],
        #     'fairxgboost__snsftr_slctrt_sub_groups': [(self.snsftr_slctrt_sub_groups[0][1],self.snsftr_slctrt_sub_groups[1][1])],
        #     'fairxgboost__verbose': [self.verbose],
        # }

        # FINAL SLIM gridsearch params set
        param_grid = {
            'fairxgboost__base_clf': [self.base_clf],
            'fairxgboost__anomalies_per_to_remove': [0.35, 0.4, 0.45, 0.5],
            'fairxgboost__include_sensitive_feature': [True],  # False
            'fairxgboost__sensitive_col_name': [self.sensitive_feature_name],
            'fairxgboost__remove_side': remove_side_list,
            'fairxgboost__data_columns': [tuple(X_train.columns)],
            'fairxgboost__anomaly_model_params': FairClassifier.build_gridsearch_cv_params(
                num_samples_in_gridsearch_fold=int(data_portion_in_fold * X_train.shape[0])),
            'fairxgboost__snsftr_slctrt_sub_groups': [(self.snsftr_slctrt_sub_groups[0][1],self.snsftr_slctrt_sub_groups[1][1])],
            'fairxgboost__verbose': [self.verbose],
        }

        num_iters = self.n_repeats * self.n_splits
        for params_set in param_grid.values():
            num_iters *= len(params_set)
        print(f"Number of GridSearch fits: {num_iters}")

        pipe_cv = None

        with tqdm(total=num_iters) as pbar:

            def aod_measure(y_true: pd.Series, y_pred:np.ndarray) -> float:
                sensitive_selected_arr = self.sensitive_feature_srs.values[grid_search_idx[hash(y_true.values.tobytes())]]

                try:
                    score = frns_utils.average_odds_difference(y_true=utils.to_int_srs(y_true),
                                                                      y_pred=utils.to_int_srs(pd.Series(y_pred)),
                                                                      sensitive_feature_arr=sensitive_selected_arr)
                except BaseException as e:
                    if self.verbose:
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
                    if self.verbose:
                        print(f"Exception raised due to insufficient values for some of the sub groups:\n{pd.Series(sensitive_selected_arr).value_counts()}")

                    score = 10

                if self.verbose:
                    print(f'EOD Score:{score}')

                pbar.update(1)

                return score


            scorers_dict = {'eod':make_scorer(eod_measure, greater_is_better=False),
                            'aod':make_scorer(aod_measure, greater_is_better=False),
                            'f1_macro':make_scorer(f1_measure,greater_is_better=True)}
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
                                                         performance_metrics:List=('aod','eod','f1_macro','tpr_diff','fpr_diff'),
                                                         max_num_top_models:int=50,
                                                         verbose:bool=False):
        '''

        :param target_metric_name: The name of the performance (/fiarness) metric by which to find the top models in self.pipe_cv.results_
        :param metric_value_threshold: The threshold above/below which to find top models from self.pipe_cv.results_
        :param greater_target_metric_value_is_better: If True, models with score value above the metric_value_threshold are considered. Otherwise if False.
        :param max_num_top_models: Maximum top models to retrain
        :return:List[{model_index_in_pipe_cv:value,metric_name:value,..}]
        '''
        assert self._is_fitted, 'cannot perform refit before fitting'

        supported_metrics = list(set(self.supported_metrics + ['tpr_diff', 'fpr_diff']))
        if performance_metrics is None:
            performance_metrics = supported_metrics
        else:
            performance_metrics = [x.lower() for x in performance_metrics]
        not_supported_metrics = set(performance_metrics) - set(supported_metrics)

        assert len(not_supported_metrics) == 0, f'the performance_metrics are not supported: {not_supported_metrics}'
        assert (0 <= np.array(list(target_metrics_thresholds.values()))).all() and (np.array(list(target_metrics_thresholds.values())) <=1).all(), 'target_metrics_thresholds can get values between 0 to 1'

        top_models_results = pd.DataFrame(self._pipe_cv.cv_results_)

        if max_num_top_models > len(list(self._pipe_cv.cv_results_.values())[0]):
            max_num_top_models = len(list(self._pipe_cv.cv_results_.values())[0])

        for target_metric_name, target_metric_value_threshold in target_metrics_thresholds.items():
            target_metric_col = f'mean_test_{target_metric_name.lower()}'

            greater_target_metric_value_is_better = bool(top_models_results[target_metric_col].mean() > 0)
            ascending = False if greater_target_metric_value_is_better else True

            if greater_target_metric_value_is_better and target_metric_value_threshold > 0.0:
                top_models_results = top_models_results.loc[top_models_results[target_metric_col] > target_metric_value_threshold]
            if (greater_target_metric_value_is_better == False) and target_metric_value_threshold < 1.0:
                top_models_results = top_models_results.loc[-1.0*top_models_results[target_metric_col] < target_metric_value_threshold]

            top_models_results = top_models_results.sort_values(by=target_metric_col,ascending=ascending).head(max_num_top_models)


        if top_models_results.shape[0] == 0:
            print("there was no model who mached the target metrics threshold specified")
        elif verbose:
            print(f"Start computing models performance results for {top_models_results.shape[0]} top models ...")

        X_test_sensitive_feature_srs = frns_utils.get_feature_col_from_preprocessed_data(feature_name=self.sensitive_feature_name,
                                                                                        data=X_test)

        result = []
        num_iters = top_models_results.shape[0]
        print(f"Check performance of top {num_iters} fairness aware trained models:")
        with tqdm(total=num_iters) as pbar:
            for index in list(top_models_results.index):
                params = {}
                for col in list(top_models_results.columns):
                    # if col == 'param_fairxgboost__base_clf':
                    #     params['base_clf'] = self.base_clf
                    # elif col == 'param_fairxgboost__anomaly_model_params':
                    #     params['anomaly_model_params'] = top_models_results[col][index]
                    if col.startswith('param_fairxgboost__'):
                        param_name = col.split('param_fairxgboost__')[1]
                        params[param_name] = top_models_results[col][index]

                fair_clf = FairXGBClassifier()
                fair_clf.set_params(**params)
                fair_clf.fit(X_train, y_train)

                #4. make prediction
                y_pred = fair_clf.predict(X=X_test)

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
                if 'f1_macro' in performance_metrics:
                    model_preformance_results['f1_macro'] = f1_score(y_true=y_test,
                                                                y_pred=utils.to_int_srs(pd.Series(y_pred)),
                                                                average='macro')

                if 'tpr_diff' in performance_metrics:
                    model_preformance_results['tpr_diff'] = true_positive_rate_difference(y_true=y_test,
                                                                                     y_pred=utils.to_int_srs(pd.Series(y_pred)),
                                                                                     sensitive_features=X_test_sensitive_feature_srs.values)

                if 'fpr_diff' in performance_metrics:
                    model_preformance_results['fpr_diff'] = false_positive_rate_difference(y_true=y_test,
                                                                                      y_pred=utils.to_int_srs(pd.Series(y_pred)),
                                                                                      sensitive_features=X_test_sensitive_feature_srs.values)


                if verbose:
                    print(f"top model #{index} performance results: {model_preformance_results}")

                result.append(model_preformance_results)
                pbar.update(1)

        return result

    def predict(self, X_test:pd.DataFrame) -> pd.Series:
        check_is_fitted(self, '_is_fitted')

        return pd.Series(self._pipe_cv.best_estimator_.predict(X=X_test))


    def predict_proba(self, X_test:pd.DataFrame) -> pd.Series:
        check_is_fitted(self, '_is_fitted')

        return pd.Series(self._pipe_cv.predict_proba(X=X_test))