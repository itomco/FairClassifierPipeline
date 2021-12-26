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

from FairClassifierPipeline import FairPipeline as fp


config = {}
config['sensitive_feature'] = 'age'
config['label_col'] = 'y'
config['label_ordered_classes'] = (['yes', 'no'], [1, 0]) #relevant for Fairness Positive Label Value

config['numerical_str_features_stubs'] =    {}
# config['numerical_far_feature'] =    {'pdays':[-1]}
config['ordinal_categorial_features'] = {
                                        #'education':['unknown','primary', 'secondary', 'tertiary'],
                                        'month':['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],
                                        #'contact':['unknown','telephone','cellular'],
                                        #'poutcome':['failure','other','unknown','success']
                                        }

config['numeric_to_ord_cat'] = {'age': [30,60]}

config['max_sparse_col_threshold'] = 0.8

if __name__ == '__main__':
    ppl = fp.create_pipeline(config)
    print(ppl)
    data = pd.read_csv('./Data/bank.csv', delimiter=";", header='infer')
    print(data.head())
    train_df, test_df = train_test_split(data, test_size=0.2)

    ####Run the Data PreProcessing Pipeline on Train Dataset

    # categorical_numerical_preprocessor.fit_transform(X_train, y_train)
    preprocessed_train_data = ppl.fit_transform(train_df)
    final_columns = fp.get_pipeline_final_columns(ppl)

    preprocessed_train_data = pd.DataFrame(data=preprocessed_train_data, columns=final_columns)

    # https://datatofish.com/numpy-array-to-pandas-dataframe/ - following this tutorial, a numpy array containing multiple types of columns values results with all object array.
    # https://stackoverflow.com/questions/61346021/create-a-mixed-type-pandas-dataframe-using-an-numpy-array-of-type-object
    preprocessed_train_data = preprocessed_train_data.convert_dtypes()

    X_train, y_train = preprocessed_train_data.drop(columns=[config['label_col']], axis=1).values, \
                       preprocessed_train_data[config['label_col']].values
    print(preprocessed_train_data.head(3))

    ####Run the Data PreProcessing Pipeline on Test Dataset

    preprocessed_test_data = ppl.transform(test_df)
    preprocessed_test_data = pd.DataFrame(data=preprocessed_test_data, columns=final_columns)
    preprocessed_test_data = preprocessed_test_data.convert_dtypes()
    X_test, y_test = preprocessed_test_data.drop(columns=[config['label_col']], axis=1).values, preprocessed_test_data[
        config['label_col']].values

    print(preprocessed_test_data.head(3))

    #### Pipeline Stracture Graph plot
    set_config(display='diagram')
    ppl