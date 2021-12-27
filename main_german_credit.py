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

from FairClassifierPipeline import FairPipeline as fp
from FairClassifierPipeline import Utils as utils
from BaseClassifiers import GermanCreditBaseClf as base_clf
from FairClassifierPipeline import FairClassifier as fair_clf

print(xgboost.__version__)

def create_config() -> Dict:
    config = {}
    config['sensitive_features'] = ['age','statussex']
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

def run_baseline_clf_on_preprocessed_data(X_baseline_train_clean, X_baseline_test_clean, y_baseline_train_clean, y_baseline_test_clean):

    params2 = {
        'n_estimators': 3000,
        'objective': 'binary:logistic',
        'learning_rate': 0.005,
        # 'gamma':0.01,
        'subsample': 0.555,
        'colsample_bytree': 0.7,
        'min_child_weight': 3,
        'max_depth': 8,
        'use_label_encoder' : False,
        # 'seed':1024,
        'n_jobs': -1
    }

    return base_clf.xgbclf(params2, X_baseline_train_clean, y_baseline_train_clean, X_baseline_test_clean, y_baseline_test_clean)

def run_baseline_clf(data:pd.DataFrame):
    data_baseline = data.copy()
    # Binarize the y output for easier use of e.g. ROC curves -> 0 = 'bad' credit; 1 = 'good' credit
    data_baseline.classification.replace([1, 2], [1, 0], inplace=True)
    # Print number of 'good' credits (should be 700) and 'bad credits (should be 300)
    data_baseline.classification.value_counts()

    # numerical variables labels
    numvars = ['creditamount', 'duration', 'installmentrate', 'residencesince', 'age',
               'existingcredits', 'peopleliable', 'classification']

    # Standardization
    numdata_std = pd.DataFrame(StandardScaler().fit_transform(data_baseline[numvars].drop(['classification'], axis=1)))

    # categorical variables labels
    catvars = ['existingchecking', 'credithistory', 'purpose', 'savings', 'employmentsince',
               'statussex', 'otherdebtors', 'property', 'otherinstallmentplans', 'housing', 'job',
               'telephone', 'foreignworker']

    d = defaultdict(LabelEncoder)

    # Encoding the variable
    lecatdata = data_baseline[catvars].apply(lambda x: d[x.name].fit_transform(x))

    # print transformations
    for x in range(len(catvars)):
        print(catvars[x], ": ", data_baseline[catvars[x]].unique())
        print(catvars[x], ": ", lecatdata[catvars[x]].unique())

    # One hot encoding, create dummy variables for every category of every categorical variable
    dummyvars = pd.get_dummies(data_baseline[catvars])

    data_baseline_clean = pd.concat([data_baseline[numvars], dummyvars], axis=1)

    print(data_baseline_clean.shape)

    # Unscaled, unnormalized data
    X_baseline_clean = data_baseline_clean.drop('classification', axis=1)
    y_baseline_clean = data_baseline_clean['classification']

    X_baseline_train_clean, X_baseline_test_clean, y_baseline_train_clean, y_baseline_test_clean = train_test_split(
        X_baseline_clean, y_baseline_clean, test_size=0.2, random_state=1)

    return (X_baseline_test_clean,*run_baseline_clf_on_preprocessed_data(X_baseline_train_clean, X_baseline_test_clean, y_baseline_train_clean, y_baseline_test_clean))


def run_fair_data_preprocess_pipeline(data:pd.DataFrame, config:Dict):
    # categorical_numerical_preprocessor.fit_transform(X_train, y_train)
    preprocessed_train_data = ppl.fit_transform(train_df)
    final_columns = fp.get_pipeline_final_columns(ppl)

    preprocessed_train_data = pd.DataFrame(data=preprocessed_train_data, columns=final_columns)

    # https://datatofish.com/numpy-array-to-pandas-dataframe/ - following this tutorial, a numpy array containing multiple types of columns values results with all object array.
    # https://stackoverflow.com/questions/61346021/create-a-mixed-type-pandas-dataframe-using-an-numpy-array-of-type-object
    preprocessed_train_data = preprocessed_train_data.convert_dtypes()

    X_train, y_train = preprocessed_train_data.drop(columns=[config['label_col']], axis=1), preprocessed_train_data[config['label_col']]
    print(preprocessed_train_data.head(3))

    ####Run the Data PreProcessing Pipeline on Test Dataset

    preprocessed_test_data = ppl.transform(test_df)
    preprocessed_test_data = pd.DataFrame(data=preprocessed_test_data, columns=final_columns)
    preprocessed_test_data = preprocessed_test_data.convert_dtypes()
    X_test, y_test = preprocessed_test_data.drop(columns=[config['label_col']], axis=1), preprocessed_test_data[
        config['label_col']]

    print(preprocessed_test_data.head(3))

    return(ppl, preprocessed_train_data, preprocessed_test_data, X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    #### Load data
    names = ['existingchecking', 'duration', 'credithistory', 'purpose', 'creditamount',
             'savings', 'employmentsince', 'installmentrate', 'statussex', 'otherdebtors',
             'residencesince', 'property', 'age', 'otherinstallmentplans', 'housing',
             'existingcredits', 'job', 'peopleliable', 'telephone', 'foreignworker', 'classification']

    data = pd.read_csv('./Data/german.data', names=names, delimiter=' ')
    print(data.head())

    #### Execute Baseline XGBoost Classifier
    base_X_test, base_model, base_y_test, base_y_pred, base_y_pred_proba, base_fpr, base_tpr, base_auc = run_baseline_clf(data.copy())
    print(f"Base model AUC: {base_auc}")
    utils.print_confusion_matrix(base_y_test,base_y_pred, base_y_pred_proba)

    #### Execute Fair Pipeline
    train_df, test_df = train_test_split(data, test_size=0.2)

    ####Run the Data PreProcessing Pipeline on Train Dataset
    config = create_config()
    sensitive_features_names = config['sensitive_features']
    snsftr_eods_w_base_preprocess = {}
    snsftr_eods_w_fair_pipeline = {}
    for sf in sensitive_features_names:
        if sf in base_X_test.columns and fp.is_categorial(base_X_test[sf]) == False:
            continue

        config = create_config()
        config['sensitive_feature'] = sf
        ppl = fp.create_pipeline(config)
        # print(ppl)

        ppl, preprocessed_train_data, preprocessed_test_data, X_train, X_test, y_train, y_test = run_fair_data_preprocess_pipeline(data.copy(), config)

        #### Pipeline Stracture Graph plot
        #set_config(display='diagram')
        # ppl

        #### Execute Baseline XGBoost Classifier over fairly preprocessed data
        initial_model, initial_y_test, initial_y_pred, initial_y_pred_proba, initial_fpr, initial_tpr, initial_auc = \
            run_baseline_clf_on_preprocessed_data(X_train.astype(float), X_test.astype(float), pd.Series(np.array(y_train.values,dtype=int)), pd.Series(np.array(y_test.values,dtype=int)))

        print(f"Initial model AUC: {initial_auc}")
        utils.print_confusion_matrix(initial_y_test,initial_y_pred, initial_y_pred_proba)

        #compare base vs initial results
        snsftr_eods_w_base_preprocess.update(fair_clf.get_eod_for_sensitive_features(sensitive_features_names= [sf],
                                                                                y_true=base_y_test,
                                                                                y_pred=pd.Series(base_y_pred),
                                                                                X_pd=base_X_test))


        snsftr_eods_w_fair_pipeline.update(fair_clf.get_eod_for_sensitive_features(sensitive_features_names = [sf],
                                                                              y_true=initial_y_test,
                                                                              y_pred=pd.Series(initial_y_pred),
                                                                              X_pd=X_test))

    base_vs_initial_eod_results = pd.DataFrame([snsftr_eods_w_base_preprocess,snsftr_eods_w_fair_pipeline]).T
    base_vs_initial_eod_results.columns = ['base','initial']
    print(base_vs_initial_eod_results)

    # if sensitive_feature_name not in X_train.columns:
    #     sensitive_feature_srs = fair_clf.merge_feature_onehot_columns(X_train_df=X_train, feature_name=sensitive_feature_name)
    # else:
    #     sensitive_feature_srs = X_train[sensitive_feature_name]
    #
    # snsftr_groups_slctnrt_and_acc, snsftr_slctrt_sub_groups = fair_clf.get_feature_sub_groups_by_selection_rate( clf_model = initial_model,
    #                                                                                                              X_train_df = X_train,
    #                                                                                                              y_train = y_train,
    #                                                                                                              sensitive_feature_srs = sensitive_feature_srs)
    #
    #
    # gridsearch_cv = fair_clf.build_gridsearch_cv( X_train_df = X_train,
    #                                              y_train = y_train,
    #                                              sensitive_feature_name = sensitive_feature_name,
    #                                              sensitive_feature_srs = sensitive_feature_srs,
    #                                              snsftr_slctrt_sub_groups = snsftr_slctrt_sub_groups)
    #
