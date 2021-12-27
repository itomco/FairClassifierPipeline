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

# Function to get roc curve
def get_roc (y_test,y_pred):
    '''fpr, tpr, roc_auc'''
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    #Plot of a ROC curve
    # plt.figure()
    # lw = 2
    # plt.plot(fpr, tpr, color='darkorange',
    #          label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic')
    # plt.legend(loc="upper left")
    # plt.show()
    return(fpr, tpr, roc_auc)

# fit, train and cross validate Decision Tree with training and test data
def xgbclf(params, X_train, y_train, X_test, y_test):
    '''return model,y_test, y_pred, y_pred_proba, fpr,tpr,auc'''
    eval_set = [(X_train, y_train), (X_test, y_test)]

    model_ = XGBClassifier(**params)
    model_.fit(X_train, y_train, eval_set=eval_set,
            eval_metric='auc', early_stopping_rounds=100, verbose=100)

    # print(model.best_ntree_limit)
    model = XGBClassifier(**params)
    model.set_params(**{'n_estimators': model_.best_ntree_limit})
    model.fit(X_train, y_train)
    # print(model,'\n')

    # Predict target variables y for test data
    y_pred = model.predict(X_test, ntree_limit=model.best_ntree_limit)  # model.best_iteration
    # print(y_pred)

    # Get Cross Validation and Confusion matrix
    # get_eval(model, X_train, y_train)
    # get_eval2(model, X_train, y_train,X_test, y_test)

    # Create and print confusion matrix
    abclf_cm = confusion_matrix(y_test, y_pred)
    print(abclf_cm)

    # y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print('\n')
    print("Model Final Generalization Accuracy: %.6f" % accuracy_score(y_test, y_pred))

    # Predict probabilities target variables y for test data
    y_pred_proba = model.predict_proba(X_test, ntree_limit=model.best_ntree_limit)[:, 1]  # model.best_iteration

    return (model, y_test, y_pred, y_pred_proba, *get_roc(y_test, y_pred_proba)) #return model,fpr,tpr,auc

