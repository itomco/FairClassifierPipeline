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
from BaseClassifiers import BaseClf
from BaseClassifiers import GermanCreditBaseClf as base_clf
from BaseClassifiers.GermanCreditBaseClf import GermanBaseClf as base_clf_class
from FairClassifierPipeline import FairClassifier as fair_clf

# print(xgboost.__version__)

def create_config() -> Dict:
    config = {}
    config['sensitive_features'] = ['statussex']
    config['label_col'] = 'classification'
    config['label_ordered_classes'] = ([1,2], [1,0]) #relevant for Fairness Positive Label Value

    # numerical_str_features_stubs enable the following comfort generic settings:
    # 1. in all cases where numerical value is sent as a string with or with out a prefix and / or a suffix, this would 'clean' this features values and convert to
    #    int or float according to the actual column's value type
    # 2. easier approach to define ordinal categorical features in case they are already properly ordered buy has some prefix and / or suffix to all values
    config['numerical_str_features_stubs'] =    {
                                                'otherinstallmentplans':('A14',None),
                                                }

    # https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#
    config['ordinal_categorial_features'] = {
                                            'existingchecking':['A14','A11', 'A12', 'A13'],
                                            'credithistory':['A30','A31','A32','A33','A34'],
                                            'savings':['A65','A61','A62','A63','A64'],
                                            'employmentsince':['A71','A72','A73','A74','A75'],
                                            # 'installmentrate':[1,2,3,4],#8
                                            'otherdebtors':['A101','A102','A103'],
                                            # 'residencesince':[1,2,3,4],#11
                                            'property':['A121','A122','A123','A124'],
                                            # 'otherinstallmentplans':['A141','A142','A143'],
                                            'housing':['A151','A152','A153'],
                                            # 'existingcredits':[1,2,3,4],#16
                                            'job':['A171','A172','A173','A174'],
                                            # 'peopleliable':[1,2],#18
                                            }

    config['numeric_to_ord_cat'] = {'age':[25,50]}

    config['max_sparse_col_threshold'] = 0.8

    return config

def load_data():
    names = ['existingchecking', 'duration', 'credithistory', 'purpose', 'creditamount',
             'savings', 'employmentsince', 'installmentrate', 'statussex', 'otherdebtors',
             'residencesince', 'property', 'age', 'otherinstallmentplans', 'housing',
             'existingcredits', 'job', 'peopleliable', 'telephone', 'foreignworker', 'classification']

    return pd.read_csv('./Data/german.data', names=names, delimiter=' ')

if __name__ == '__main__':
    ####-1. Load data
    data = load_data()
    # print(data.head())

    ####-2. Execute Baseline XGBoost Classifier
    base_X_test, base_y_test, base_model, base_y_pred, base_y_pred_proba = base_clf.run_baseline_clf(data.copy())
    base_fpr, base_tpr, base_auc = base_clf.get_roc(y_test=base_y_test,
                                                    y_pred=base_y_pred)

    print(f"Base model AUC: {base_auc}")
    utils.print_confusion_matrix(base_y_test,base_y_pred, base_y_pred_proba)

    ####-3. Check fair pipeline impact on base model

    ####Run the Data PreProcessing Pipeline on Train Dataset
    config = create_config()
    sensitive_features_names = config['sensitive_features']
    snsftr_eods_w_base_preprocess = {}
    snsftr_eods_w_fair_pipeline = {}
    snsftr_f1_w_base_preprocess = {}
    snsftr_f1_w_fair_pipeline = {}

    for sf in sensitive_features_names:
        if sf in base_X_test.columns and fair_ppl.is_categorial(base_X_test[sf]) == False:
            continue

        config = create_config()
        config['sensitive_feature'] = sf

        #base model
        snsftr_eods_w_base_preprocess.update(fair_clf.get_eod_for_sensitive_features(sensitive_features_names= [sf],
                                                                                y_true=base_y_test,
                                                                                y_pred=pd.Series(base_y_pred),
                                                                                data=base_X_test))

        clsf_rprt = classification_report(base_y_test, pd.Series(base_y_pred), digits=4, output_dict=True)
        snsftr_f1_w_base_preprocess.update({'accuracy':clsf_rprt['accuracy'],
                                            'precision':clsf_rprt['macro avg']['precision'],
                                            'recall':clsf_rprt['macro avg']['recall'],
                                            'f1-score':clsf_rprt['macro avg']['f1-score']})



        #initial
        ppl, preprocessed_train_data, preprocessed_test_data, initial_X_train, initial_X_test, initial_y_train, initial_y_test = \
                                                                                        fair_ppl.run_fair_data_preprocess_pipeline(data.copy(), config)

        #### Pipeline Stracture Graph plot
        #set_config(display='diagram')
        # ppl

        #### Execute Baseline XGBoost Classifier over fairly preprocessed data
        initial_y_test = utils.to_int_srs(initial_y_test)
        initial_model, initial_y_pred, initial_y_pred_proba = base_clf_class.fit_predict(X_train= utils.to_float_df(initial_X_train),
                                                                                                      y_train= utils.to_int_srs(initial_y_train),
                                                                                                      X_test= utils.to_float_df(initial_X_test),
                                                                                                      y_test= initial_y_test)

        initial_fpr, initial_tpr, initial_auc = base_clf.get_roc(y_test= initial_y_test,
                                                                 y_pred= initial_y_pred)

        print(f"Initial model AUC: {initial_auc}")
        utils.print_confusion_matrix(utils.to_int_srs(initial_y_test),initial_y_pred, initial_y_pred_proba)

        snsftr_eods_w_fair_pipeline.update(fair_clf.get_eod_for_sensitive_features(sensitive_features_names = [sf],
                                                                              y_true=initial_y_test,
                                                                              y_pred=pd.Series(initial_y_pred),
                                                                              data=initial_X_test))

        clsf_rprt = classification_report(initial_y_test, pd.Series(initial_y_pred), digits=4, output_dict=True)
        snsftr_f1_w_fair_pipeline.update({'accuracy':clsf_rprt['accuracy'],
                                          'precision':clsf_rprt['macro avg']['precision'],
                                          'recall':clsf_rprt['macro avg']['recall'],
                                          'f1-score':clsf_rprt['macro avg']['f1-score']})


    base_vs_initial_eod_results = pd.DataFrame([snsftr_eods_w_base_preprocess,
                                                snsftr_eods_w_fair_pipeline]).T

    base_vs_initial_macro_avg_cf_resuls = pd.DataFrame([snsftr_f1_w_base_preprocess,
                                                       snsftr_f1_w_fair_pipeline]).T

    base_vs_initial_eod_results.columns = ['base','initial']
    base_vs_initial_macro_avg_cf_resuls.columns = ['base','initial']
    print(pd.concat([base_vs_initial_eod_results,base_vs_initial_macro_avg_cf_resuls],axis=0))

    ####-4. search for most fairness biased sensitive feature
    sensitive_features_eod_scores_dict = {}
    for sf in sensitive_features_names:
        config = create_config()
        config['sensitive_feature'] = sf

        ppl, preprocessed_train_data, preprocessed_test_data, X_train, X_test, y_train, y_test = fair_ppl.run_fair_data_preprocess_pipeline(data.copy(), config)

        if sf in X_train.columns and fair_ppl.is_categorial(X_train[sf]) == False:
            print(f'sensitive feature {sf} is not categorical thus removed from sensitive features list')
            sensitive_features_names.remove(sf)
            continue

        y_test = utils.to_int_srs(y_test)
        X_test = utils.to_float_df(X_test)
        xgb_clf, y_pred, y_pred_proba = base_clf_class.fit_predict(X_train=utils.to_float_df(X_train),
                                                                                y_train=utils.to_int_srs(y_train),
                                                                                X_test=X_test,
                                                                                y_test=y_test)


        sensitive_features_eod_scores_dict.update(fair_clf.get_eod_for_sensitive_features(sensitive_features_names=[sf],
                                                                                     y_true=y_test,
                                                                                     y_pred=pd.Series(y_pred),
                                                                                     data=X_test))

    sensitive_features_eod_scores_df = pd.DataFrame.from_dict(sensitive_features_eod_scores_dict, orient='index', columns=['eod'])
    sensitive_features_eod_scores_df = sensitive_features_eod_scores_df.sort_values(ascending=False, by=['eod'])
    print(sensitive_features_eod_scores_df)

    sensitive_feature = sensitive_features_eod_scores_df.index[0].split('_')[1]



    ####-5. find privileged and unprivileged groups in sensitive feature
    config = create_config()
    config['sensitive_feature'] = sensitive_feature

    ppl, preprocessed_train_data, preprocessed_test_data, X_train, X_test, y_train, y_test = \
                                                            fair_ppl.run_fair_data_preprocess_pipeline(data.copy(), config, stratify_mode='full')

    X_train = utils.to_float_df(X_train)
    y_train = utils.to_int_srs(y_train)
    xgb_clf = base_clf_class.fit(X_train=X_train,
                                                y_train=y_train,
                                                X_test=utils.to_float_df(X_test),
                                                y_test=utils.to_int_srs(y_test))

    y_pred, y_pred_proba = base_clf_class.predict(clf=xgb_clf,
                                                  X=X_train)

    sensitive_feature_srs = fair_clf.get_feature_col_from_preprocessed_data(feature_name=sensitive_feature,
                                                                            data= X_train)
    snsftr_groups_slctnrt_and_acc, snsftr_slctrt_sub_groups = \
        fair_clf.get_feature_sub_groups_by_selection_rate(   y_true= y_train,
                                                             y_pred= utils.to_int_srs(pd.Series(y_pred)),
                                                             sensitive_feature_srs = sensitive_feature_srs)



    ####-6. run gridsearch_cv with anomaly samples removal
    gridsearch_cv = fair_clf.run_gridsearch_cv(base_clf_class=base_clf_class,
                                               X_train = X_train,
                                               y_train = y_train,
                                               sensitive_feature_name = sensitive_feature,
                                               sensitive_feature_srs = sensitive_feature_srs,
                                               snsftr_slctrt_sub_groups = snsftr_slctrt_sub_groups)

