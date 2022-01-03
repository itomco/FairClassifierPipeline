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
from datetime import datetime
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
import xgboost
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

from FairClassifierPipeline import FairPipeline as fair_ppl
from FairClassifierPipeline import Utils as utils
from BaseClassifiers.BaseClf import BaseClf
from BaseClassifiers import BaseClfCreator
# from BaseClassifiers import GermanCreditBaseClf as base_clf
# from BaseClassifiers.GermanCreditBaseClf import GermanBaseClf as base_clf_class
from FairClassifierPipeline.FairClassifier import FairClassifier
from FairClassifierPipeline import FairnessUtils as frns_utils
from Configs import Configurator as cfg
from Data import DataLoader as data_loader
# print(xgboost.__version__)

create_config = True
if create_config:
    cfg.create_configs()

def load_config(config_name:str) -> Dict:
    with open(f'Configs/{config_name}.json', 'r', encoding='utf-8') as f:
        config_reloaded = json.load(f)
    return config_reloaded

def showcase_pipeline_impact_on_base_model(config:Dict,
                                           fairness_metrics:List,
                                           base_clf:BaseClf,
                                           data:pd.DataFrame
                                           ):

    sensitive_features_names = config['sensitive_features']


    for sf in sensitive_features_names:
        if sf in data.columns and fair_ppl.is_categorial(data[sf]) == False:
            #skip sensitive feature with continues values as base model's data preprocessing
            # does not convert it to categorical feature as our pipeline does
            continue

        snsftr_eods_w_base_preprocess = {}
        snsftr_eods_w_fair_pipeline = {}
        snsftr_f1_w_base_preprocess = {}
        snsftr_f1_w_fair_pipeline = {}

        config['sensitive_feature'] = sf

        #Execute Baseline XGBoost Classifier
        base_X_train, base_X_test, base_y_train, base_y_test, base_model, base_y_pred, base_y_pred_proba = \
                                                                    base_clf.run_baseline(data=data.copy(),
                                                                                          config=config)

        base_fpr, base_tpr, base_auc = utils.get_roc(y_test=base_y_test,
                                                     y_pred=base_y_pred)

        print(f"Base model AUC: {base_auc}")
        utils.print_confusion_matrix(base_y_test,base_y_pred, base_y_pred_proba)

        #base model
        for frns_mtrc in fairness_metrics:
            frns_mtrc = frns_mtrc.lower()
            snsftr_eods_w_base_preprocess.update(frns_utils.get_fairness_score_for_sensitive_features(sensitive_features_names= [sf],
                                                                                                    fairness_metric=frns_mtrc,
                                                                                                    y_true=base_y_test,
                                                                                                    y_pred=pd.Series(base_y_pred),
                                                                                                    data=base_X_test))

        clsf_rprt = classification_report(base_y_test, pd.Series(base_y_pred), digits=4, output_dict=True)
        snsftr_f1_w_base_preprocess.update({f'{sf}:accuracy':clsf_rprt['accuracy'],
                                            f'{sf}:macro_avg-precision':clsf_rprt['macro avg']['precision'],
                                            f'{sf}:macro_avg-recall':clsf_rprt['macro avg']['recall'],
                                            f'{sf}:macro_avg-f1-score':clsf_rprt['macro avg']['f1-score']})



        #initial
        ppl, preprocessed_train_data, preprocessed_test_data, initial_X_train, initial_X_test, initial_y_train, initial_y_test = \
                                                                                        fair_ppl.run_fair_data_preprocess_pipeline(data=data.copy(), config=config)

        #### Pipeline Stracture Graph plot
        #set_config(display='diagram')
        # ppl

        #### Execute Baseline XGBoost Classifier over fairly preprocessed data
        initial_y_test = utils.to_int_srs(initial_y_test)
        initial_model, initial_y_pred, initial_y_pred_proba = base_clf.fit_predict(X_train= utils.to_float_df(initial_X_train),
                                                                                  y_train= utils.to_int_srs(initial_y_train),
                                                                                  X_test= utils.to_float_df(initial_X_test))

        initial_fpr, initial_tpr, initial_auc = utils.get_roc(y_test= initial_y_test,
                                                                 y_pred= initial_y_pred)

        print(f"Initial model AUC: {initial_auc}")
        utils.print_confusion_matrix(utils.to_int_srs(initial_y_test),initial_y_pred, initial_y_pred_proba)

        for frns_mtrc in fairness_metrics:
            snsftr_eods_w_fair_pipeline.update(frns_utils.get_fairness_score_for_sensitive_features(sensitive_features_names = [sf],
                                                                                                  fairness_metric=frns_mtrc,
                                                                                                  y_true=initial_y_test,
                                                                                                  y_pred=pd.Series(initial_y_pred),
                                                                                                  data=initial_X_test))

        clsf_rprt = classification_report(initial_y_test, pd.Series(initial_y_pred), digits=4, output_dict=True)
        snsftr_f1_w_fair_pipeline.update({f'{sf}:accuracy':clsf_rprt['accuracy'],
                                          f'{sf}:macro_avg-precision':clsf_rprt['macro avg']['precision'],
                                          f'{sf}:macro_avg-recall':clsf_rprt['macro avg']['recall'],
                                          f'{sf}:macro_avg-f1-score':clsf_rprt['macro avg']['f1-score']})


        base_vs_initial_eod_results = pd.DataFrame([snsftr_eods_w_base_preprocess,
                                                    snsftr_eods_w_fair_pipeline]).T

        base_vs_initial_macro_avg_cf_resuls = pd.DataFrame([snsftr_f1_w_base_preprocess,
                                                           snsftr_f1_w_fair_pipeline]).T

        base_vs_initial_eod_results.columns = ['base','initial']
        base_vs_initial_macro_avg_cf_resuls.columns = ['base','initial']
        print(f"Base model vs Initial Model for sensitive feature '{sf}':\n{pd.concat([base_vs_initial_eod_results,base_vs_initial_macro_avg_cf_resuls],axis=0)}")

if __name__ == '__main__':
    for project_mode in ['german']:
        # project_mode = 'german' # select 'bank' or 'german'

        ####-0 select config
        config = load_config(config_name=project_mode)

        fairness_metrics = config['fairness_metrics']
        target_fairness_metric = config['target_fairness_metric']

        ####-1. Load data
        data = data_loader.load_data(project_mode=project_mode, data_path=config['data_path'])
        print(data.head(3))

        ####-2. Execute Baseline XGBoost Classifier
        base_clf:BaseClf = BaseClfCreator.create_base_clf(project_mode=project_mode)

        ####-3. Check fair pipeline impact on base model
        showcase_pipeline_impact_on_base_model(config=config,
                                               fairness_metrics = fairness_metrics,
                                               base_clf=base_clf,
                                               data=data.copy())

        ####-4. search for most fairness biased sensitive feature

        sensitive_feature = FairClassifier.get_most_biased_sensitive_feature(data=data.copy(),
                                                                       fairness_metric=target_fairness_metric,
                                                                       base_clf=base_clf,
                                                                       config=config)

        print(f"Sensitive feature with highest un-fair bias based on fairness metric '{target_fairness_metric}' is: {sensitive_feature} ")


        ####-5. find privileged and unprivileged groups in sensitive feature
        config = load_config(config_name=project_mode)
        config['sensitive_feature'] = sensitive_feature

        ppl, preprocessed_train_data, preprocessed_test_data, X_train, X_test, y_train, y_test = \
                                                                fair_ppl.run_fair_data_preprocess_pipeline(data=data.copy(), config=config)

        X_train = utils.to_float_df(X_train)
        y_train = utils.to_int_srs(y_train)
        X_test = utils.to_float_df(X_test)
        y_test = utils.to_int_srs(y_test)

        xgb_clf = base_clf.fit(X_train=X_train,
                                y_train=y_train,
                                X_test=X_test)

        y_pred, y_pred_proba = base_clf.predict(clf=xgb_clf,
                                                      X=X_train)
        y_pred = utils.to_int_srs(pd.Series(y_pred))

        sensitive_feature_srs = frns_utils.get_feature_col_from_preprocessed_data(feature_name=sensitive_feature,
                                                                                data= X_train)
        snsftr_groups_slctnrt_and_acc, snsftr_slctrt_sub_groups = \
            frns_utils.get_feature_sub_groups_by_selection_rate( y_true= y_train,
                                                                 y_pred= y_pred,
                                                                 sensitive_feature_srs = sensitive_feature_srs)

        print(f"snsftr_slctrt_sub_groups: {snsftr_slctrt_sub_groups}\n")
        print(f"snsftr_groups_slctnrt_and_acc:\n{snsftr_groups_slctnrt_and_acc}\n")

        ####-6. run gridsearch_cv with anomaly samples removal
        fair_clf = FairClassifier(target_fairness_metric = target_fairness_metric,
                                   base_clf=base_clf,
                                   verbose=False)

        fair_clf.fit(X_train = X_train,
                     y_train = y_train,
                     sensitive_feature_name=sensitive_feature,
                     snsftr_slctrt_sub_groups=snsftr_slctrt_sub_groups)

        pipe_cv = fair_clf.pipe_cv
        results = pd.DataFrame(pipe_cv.cv_results_)
        datetime_tag = datetime.now().strftime("%y%m%d_%H%M%S")
        results.to_csv(f'./gscv_results/{datetime_tag}_{project_mode}_{target_fairness_metric}_pipe_cv.cv_results_.csv')
        print(f'results:\n{results}')

        y_pred = fair_clf.predict(X_test)
        # y_pred = utils.to_int_srs(pd.Series(y_pred))

        best_fair_clf_model = frns_utils.get_fairness_score_for_sensitive_features(sensitive_features_names=[sensitive_feature],
                                                                           fairness_metric=target_fairness_metric,
                                                                           y_true=y_test,
                                                                           y_pred=y_pred,
                                                                           data=X_test)

        print(f"best_fair_clf_model: {best_fair_clf_model}")
        print(classification_report(y_test, pd.Series(y_pred), digits=4))

        top_models_scores_on_test = fair_clf.retrain_top_models_and_get_performance_metrics(X_train=X_train,
                                                                                            y_train=y_train,
                                                                                            X_test=X_test,
                                                                                            y_test=y_test,
                                                                                            max_num_top_models=180,
                                                                                            target_metrics_thresholds={target_fairness_metric:1.0,
                                                                                                                       'f1':0.0})

        print(top_models_scores_on_test)
        top_models_scores_on_test_df = pd.DataFrame(top_models_scores_on_test).sort_values(by=[target_fairness_metric.lower(),'f1'])
        print(top_models_scores_on_test_df)
        top_models_scores_on_test_df.to_csv(f'./gscv_results/{datetime_tag}_{project_mode}_{target_fairness_metric}_top_models_scores_on_test_df.csv')