#Import required libraries
import pandas as pd
import numpy as np
from typing import *

import matplotlib.pyplot as plt

np.random.seed(sum(map(ord, "aesthetics")))

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

#anomaly detection models
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

from FairClassifierPipeline.RobustRandomCutForest import RobustRandomCutForest as RRCF

from BaseClassifiers.BaseClf import BaseClf
from FairClassifierPipeline import FairnessUtils as frns_utils


class FairXGBClassifier(ClassifierMixin, BaseEstimator):
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
        data_arr = self.data.to_numpy()
        if self.include_sensitive_feature == False:
            data_arr = frns_utils.drop_sensitive_feature( sensitive_col_name=self.sensitive_col_name,
                                                           data=self.data,
                                                           snsftr_slctrt_sub_groups = self.snsftr_slctrt_sub_groups)
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

        if algorithm_name.lower() == 'svm':
            unsupervised_model_params['nu'] = self.anomalies_per_to_remove
        else:
            unsupervised_model_params['contamination'] = self.anomalies_per_to_remove

        algorithm = anomaly_algorithms[algorithm_name](**unsupervised_model_params)

        # #############################################################################################
        # get anomalies by prediction
        if algorithm_name.lower() == 'lof':
            anomalies_idx = np.where(algorithm.fit_predict(data_arr) == -1)[0].tolist()
        else:
            algorithm.fit(data_arr)
            anomalies_idx = np.where(algorithm.predict(data_arr) == -1)[0].tolist()
        # ##############################################################################################
        # Run XGBoost without anomalies

        # print(f'snsftr_slctrt_sub_groups: {self.snsftr_slctrt_sub_groups}')
        # print(f'sensitive config: {self.sensitive_col_name}')
        # print(f'sensitive columns arr: {sensitive_feature_one_hot_columns}')
        non_privilage_group = self.snsftr_slctrt_sub_groups[0]
        privilage_group = self.snsftr_slctrt_sub_groups[1]
        # print(f'non-privilaged columns: {non_privilage_group}')

        anomalies_idx = set(anomalies_idx)
        filter_sub_sensitive_group = []
        sensitive_feature_col = frns_utils.get_feature_col_from_preprocessed_data(feature_name=self.sensitive_col_name,
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

        self.model = self.base_clf.fit(X_train=self.data.drop(list(anomalies_idx_to_remove)),
                                       y_train=pd.Series(np.delete(self.y_, list(anomalies_idx_to_remove))))

        return self


    def predict(self, X):
        y_pred, y_pred_proba = self.base_clf.predict(clf= self.model,
                                                     X = X)

        return y_pred

    def predict_proba(self, X):
        y_predict, y_pred_proba = self.base_clf.predict(clf= self.model,
                                                        X = X)
        return y_pred_proba
