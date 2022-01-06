#Import required libraries
import pandas as pd
import numpy as np
import sklearn
from typing import *

np.random.seed(sum(map(ord, "aesthetics")))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline as pipe
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn_pandas import DataFrameMapper
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import FunctionTransformer

import warnings
warnings.filterwarnings("ignore", 'The covariance matrix associated to your dataset is not full rank')

from FairClassifierPipeline import Utils as utils
from FairClassifierPipeline import FairnessUtils as frns_utils

################# Types handling functions ###########################

def remove_prefix(value, prefix):
    '''Python 3.9+ has str.removeprefix. We implement it assuming our python version is lower than 3.9
    '''
    # print(value)
    # print(prefix)
    if isinstance(prefix, str) and value.startswith(prefix):
        return value[len(prefix):]

    return value

def remove_suffix(value, suffix):
    '''Python 3.9+ has str.removesuffix. We implement it assuming our python version is lower than 3.9
    '''
    if isinstance(suffix, str) and value.endswith(suffix):
        return x[:-len(suffix)]

    return value
    
    
def is_int_str(feature_values: pd.Series) -> bool:
    # print(f"feature_values.name:{feature_values.name}")
    # print(f"feature_values.dtype:{feature_values.dtype}")
    # print(f"is_str_ftype(feature_values.dtype):{is_str_ftype(feature_values.dtype)}")
    # print(f"type(list(feature_values)[0]):{type(list(feature_values)[0])}")
    return is_str_ftype(feature_values.dtype) and all(map(str.isnumeric,feature_values[feature_values.notnull()].str.lstrip('-').str.replace(',','', regex=False)))

def is_float_str(feature_values: pd.Series) -> bool:
    return (is_str_ftype(feature_values.dtype) and
            (is_int_str(feature_values) == False) and 
            all(map(str.isdigit,feature_values[feature_values.notnull()].str.lstrip('-').str.replace(',','', regex=False).str.replace('.','',1, regex=False))))


def is_number_str(feature_values: pd.Series) -> bool:
    return is_int_str(feature_values) or is_float_str(feature_values)
    
def is_str_ftype(dtype) -> bool:
    return dtype in ['O','S','a','U'] or ('str' in str(dtype).lower())

def is_bool_ftype(dtype) -> bool:
    return (dtype == 'b') or ('bool' in str(dtype).lower())

#https://stackoverflow.com/questions/37561991/what-is-dtypeo-in-pandas
def is_numeric_ftype(feature_values: pd.Series, check_number_str: bool = False) -> bool:
    dtype = feature_values.dtype
    return ((dtype =='i') or ('int' in str(dtype).lower()) or 
            (dtype =='f') or ('float' in str(dtype).lower()) or 
            all([isinstance(x,int) for x in feature_values[feature_values.notnull()]]) or #check if all notnull values are of type int
            all([isinstance(x,float) for x in feature_values[feature_values.notnull()]]) or #check if all notnull values are of type int
            (check_number_str and is_number_str(feature_values)))

def is_categorial(feature_values: pd.Series, check_number_str:bool = True) -> bool:
    '''Check if the received series is a categorical feature (incorporate some huristics regarding the number of distinct values)
    Parameters:
    - feature_values: the series to check
    - check_number_str: for string type which actually stores valid numerical values, if check_number_str == True, then treat it as if
                        it is a numerical value. Otherwise, treat such string feature as any other (non numerical) string feature.

    Returns True if any of the following scenario applies (otherwise return False):
    I. check if this is a string value feature with up to 25 different values (i.e. discards unique personal string features like user name) 
        and if check_number_str == True verify it is not numerical string (int or float) 
    II. check if this is a boolean value feature is obviousely categorical (we transform such feature to a single 1/0 column)
    III. check if this is a numerical feature with up to 15 unique values. If check_number_str == True include situration where the column 
        type is string but actually represents numeric values (e.g. '-3.2')
    '''
    return ((len(feature_values.unique()) <= 25 and (is_str_ftype(feature_values.dtype) and ((check_number_str==False) or (is_number_str(feature_values) == False)))) or 
            is_bool_ftype(feature_values.dtype) or # in case of a boolean feature
            (len(feature_values.unique()) <= 15 and is_numeric_ftype(feature_values,check_number_str=check_number_str))) # in case of a numerical feature (may be in string values)
            
            
############################## Custom Sklearn Pipeline Components #################################

#remove columns with too high ratio of missing data
def remove_low_data_columns(df : pd.DataFrame, threshold = 0.2) -> pd.DataFrame:
    return df.drop(columns=list(df.loc[:,list((100*(df.isnull().sum()/len(df.index)) >= threshold))].columns), axis=1)


def remove_rows_wo_value_for_column(df : pd.DataFrame, label_col_name:str) -> pd.DataFrame:
    return df.loc[~df[label_col_name].isna()].copy()

def convert_numeric_cols_to_ord_cat(df : pd.DataFrame, cont_to_cat_cols_settings:Dict[str,List[int]]):
    df = df.copy()
    for col in cont_to_cat_cols_settings:
        for i, split_point in enumerate(cont_to_cat_cols_settings[col]):
            # print(f"{i}:{split_point}")
            if i == 0:
                df.loc[df[col]< split_point,col] = i
            else:
                # print(f"{i}:{cont_to_cat_cols_settings[col][i-1]}-{split_point}")
                df.loc[(cont_to_cat_cols_settings[col][i-1] <= df[col]) & (df[col]< split_point),col] = i

            if i == (len(cont_to_cat_cols_settings[col])-1):
                df.loc[split_point <= df[col],col] = i+1
    
    return df

def numeric_str_feature_encoder(df : pd.DataFrame, numerical_str_features_stubs : Dict, unknown_handling = 'ignore', non_numeric_handling='ignore'):
    unknown_numeric_columns = set(numerical_str_features_stubs.keys())-set(df.columns)
    if len(unknown_numeric_columns) > 0 and unknown_handling != 'ignore':
            raise ValueError(f"Unknown column/s {unknown_numeric_columns}")

    columns_to_encode = list(set(numerical_str_features_stubs.keys()) & set(df.columns))

    df = df.copy()
    
    for col in columns_to_encode:
        # print(f"col:{col}")
        original_values = None
        #check if current column (col) has ordinal categorical feature values transformation settings
        settings = numerical_str_features_stubs[col]
        if settings[0] is not None and settings[0] != '':
            original_values = df.loc[:,col] #temporarly save a copy of the column's original values
            #remove the prefix
            df.loc[:,col] = df[col].apply(remove_prefix, prefix = (settings[0]))

        if settings[1] is not None and settings[1] != '':
            if original_values is None:#original_values may have aleady been initiated when removing prefix
                original_values = df.loc[:,col] #temporarly save a copy of the column's original values
            #remove suffix
            df.loc[:,col] = df[col].apply(remove_suffix, suffix = (settings[1]))

        #convert the column's values into int or float
        # https://towardsdatascience.com/converting-data-to-a-numeric-type-in-pandas-db9415caab0b
        if is_int_str(df[col]):
            # print(f"converting str column {col} into 'int' values")
            df.loc[:,col] = df[col].astype(int)
            # print(X[col].dtype)
        elif is_float_str(df[col]):
            # print(f"converting str column {col} into 'float' values")
            df.loc[:,col] = df[col].astype(float)
        else:
            if non_numeric_handling == 'ignore':
                if original_values is not None:#if we've chaned the columns original values the roll it back
                    df.loc[:,col] = original_values
            else:    
                raise ValueError(f"Data in column {col} is not int nor float value type")

    return df
    
######################################### columns extractors ######################################

class LabelColExtractor(BaseEstimator, TransformerMixin ):
    def __init__(self, label_col_name:str):
        super().__init__()
        self.feature_names = []
        self.label_col_name = label_col_name

    def fit(self, X, y= None):
        #nothing to do :-)
        return(self)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names

    def transform(self, X, y= None):
        self.feature_names = [self.label_col_name]

        # print(f"extract_categorical_cols columns: {categorical_cols}")
        return X.loc[:,[self.label_col_name]].copy()

# https://newbedev.com/calling-parent-class-init-with-multiple-inheritance-what-s-the-right-way
class ColumnsSelector():
    def is_relevant_col(self,col:str, X:pd.DataFrame):
        pass            
    
class OneHotCategoricalColsExtractor(BaseEstimator, TransformerMixin, ColumnsSelector ):
    def __init__(self, non_relevant_cols : List[str], 
                 label_col_name:str, include_label_col:bool=False,
                 must_included_cols:List[str]=None):
        self.non_relevant_cols = non_relevant_cols
        self.feature_names = []
        self.categorical_cols = []
        self.label_col_name = label_col_name
        self.include_label_col = include_label_col
        self.must_included_cols = must_included_cols

    def is_relevant_col(self,col:str, X:pd.DataFrame) -> bool:
        return is_categorial(X[col]) and (col not in self.non_relevant_cols) and (is_numeric_ftype(X[col]) == False)

    def fit(self, X, y= None):
        self.categorical_cols = [col for col in X.columns if self.is_relevant_col(col,X)]

        if self.include_label_col:
            self.categorical_cols = list(set(self.categorical_cols+[self.label_col_name]))
        else:
            self.categorical_cols = list(set(self.categorical_cols) - set(self.label_col_name))

        if self.must_included_cols is not None:
            self.categorical_cols = list(set(self.categorical_cols+self.must_included_cols))

        return(self)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names

    def transform(self, X, y= None):
        self.feature_names = self.categorical_cols

        # print(f"extract_categorical_cols columns: {categorical_cols}")
        return X.loc[:,self.categorical_cols].copy()

class OrdinalCategoricalColsExtractor(BaseEstimator, TransformerMixin, ColumnsSelector ):
    def __init__(self,relevant_cols : List[str], 
                 label_col_name:str, include_label_col:bool=False,
                 must_included_cols:List[str]=None):
        self.relevant_cols = relevant_cols
        self.feature_names = []
        self.categorical_cols = []
        self.label_col_name = label_col_name
        self.include_label_col = include_label_col
        self.must_included_cols = must_included_cols

    def is_relevant_col(self,col:str, X:pd.DataFrame) -> bool:
        return is_categorial(X[col]) and ((col in self.relevant_cols) or is_numeric_ftype(X[col]))

    def fit(self, X, y= None):
        self.categorical_cols = [col for col in X.columns if self.is_relevant_col(col,X)]

        if self.include_label_col:
            self.categorical_cols = list(set(self.categorical_cols+[self.label_col_name]))
        else:
            self.categorical_cols = list(set(self.categorical_cols) - set(self.label_col_name))

        if self.must_included_cols is not None:
            self.categorical_cols = list(set(self.categorical_cols+self.must_included_cols))

        return(self)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names

    def transform(self, X, y= None):
        self.feature_names = self.categorical_cols

        # print(f"extract_categorical_cols columns: {categorical_cols}")
        return X.loc[:,self.categorical_cols].copy()


class NonCategoricalColsExtractor(BaseEstimator, TransformerMixin ):
    def __init__(self, label_col_name:str, include_label_col:bool=False,
                 must_included_cols:List[str]=None):        
        super().__init__()
        self.feature_names = []
        self.non_categorical_cols = []
        self.label_col_name = label_col_name
        self.include_label_col = include_label_col
        self.must_included_cols = must_included_cols

    def fit(self, X, y= None):
        self.non_categorical_cols = [col for col in X.columns if is_categorial(X[col]) == False]

        if self.include_label_col:
            self.non_categorical_cols = list(set(self.non_categorical_cols+[self.label_col_name]))
        else:
            self.non_categorical_cols = list(set(self.non_categorical_cols) - set(self.label_col_name))


        if self.must_included_cols is not None:
            self.non_categorical_cols = list(set(self.non_categorical_cols+self.must_included_cols))

        return(self)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names

    def transform(self, X, y=None):
        self.feature_names = self.non_categorical_cols

        # print(f"extract_non_categorical_cols columns: {self.non_categorical_cols}")
        return X.loc[:,self.non_categorical_cols].copy()

###################### pipeline transformers ################################

class LabelCatgoricalEncoder(BaseEstimator, TransformerMixin ):
    def __init__(self, label_col:str, label_col_cat_settings:Tuple[List,List]):        
        super().__init__()
        self.label_col = label_col
        self.label_col_cat_settings = label_col_cat_settings

    def fit(self, X, y= None):
        #nothing to do :-)
        return(self)

    def get_feature_names_out(self, input_features=None):
        return [self.label_col]

    def transform(self, X, y=None):
        return X[[self.label_col]].replace(self.label_col_cat_settings[0], self.label_col_cat_settings[1], inplace=False)#inplace=False thus no need to copy()

class OrdinalCatEncoder(BaseEstimator, TransformerMixin ):
    def __init__(self,ordinal_cols_selector:ColumnsSelector, ordinal_cat_settings:Dict):      #  config['ordinal_categorial_features']
        super().__init__()
        self.ordinal_cat_cols = []
        self.data_frame_mapper = None
        self.ordinal_cols_selector = ordinal_cols_selector
        self.ordinal_cat_settings = ordinal_cat_settings
        self.passthrough_ord_cols = []

    def fit(self, X, y= None):
        self.ordinal_cat_cols = list(X.columns)
        valid_ord_cat_cols = [col for col in X.columns if self.ordinal_cols_selector.is_relevant_col(col,X)]
        # print(f"OrdinalCatEncoder fit X.columns:{X.columns }")
        # print(f"OrdinalCatEncoder fit valid_ord_cat_cols:{valid_ord_cat_cols }")
        # print(f"OrdinalCatEncoder fit mutual ord cols:{list(set(valid_ord_cat_cols) & set(self.ordinal_cat_cols)) }")
        assert len(list(set(valid_ord_cat_cols) & set(self.ordinal_cat_cols))) == len(self.ordinal_cat_cols), 'X columns must be ALL the valid ordinal columns'

        self.passthrough_ord_cols = list(set(self.ordinal_cat_cols) - set(self.ordinal_cat_settings.keys()))

        ordinal_encoder_datamapper_cat_list = [([col],OrdinalEncoder(categories=[self.ordinal_cat_settings[col]], handle_unknown='use_encoded_value', unknown_value=-1))\
                                       for col in self.ordinal_cat_settings]


        self.data_frame_mapper = DataFrameMapper(ordinal_encoder_datamapper_cat_list, input_df=True, df_out=True)                                       
        self.data_frame_mapper.fit(X)
        return(self)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out

    def transform(self, X, y=None):
        X = X.copy()
        result = pd.concat([self.data_frame_mapper.transform(X), X[self.passthrough_ord_cols]], axis=1)
        self.feature_names_out = list(result.columns)

        return result

# https://towardsdatascience.com/coding-a-custom-imputer-in-scikit-learn-31bd68e541de
class AggregatedSubGroupsImputer(BaseEstimator, TransformerMixin):
    '''
    Class used for imputing missing values in a pd.DataFrame using either mean or median of a group.
    '''
    def __init__(self, group_cols:Union[None,List[str]], 
                 columns_to_impute:Union[None,List[str]] = None, 
                 label_col_name = None,
                 group_cols_to_discard:List[str]=None, 
                 strategy:str='mean',
                 add_indicator:bool=False,
                 cols_selector:ColumnsSelector = None):
        
        assert strategy in ['mean', 'median','most_frequent'], 'Unrecognized value for strategy, should be mean/median for numerical columns or most_frequent for categorical imputation'
        assert group_cols is None or isinstance(group_cols,list), 'group_cols should be a list of columns'
        assert columns_to_impute is None or isinstance(columns_to_impute,list), 'columns_to_impute should be of type list'
        common_cols = [] if (group_cols is None or columns_to_impute is None) else (set(group_cols) & set(columns_to_impute))
        assert len(common_cols) == 0, f'group_cols and columns_to_impute must not share same columns - {common_cols}'
        
        self.group_cols = group_cols if group_cols is not None else []
        self.label_col_name = label_col_name
        self.columns_to_impute = columns_to_impute if columns_to_impute is not None else []
        self.strategy = strategy
        self.add_indicator = add_indicator
        self.group_cols_to_discard = group_cols_to_discard if group_cols_to_discard is not None else []
        self.cols_selector = cols_selector

        self.feature_names_out = []   
        self.simple_imputers = {}

        # group_cols_to_discard are set of columns that are columns that are not part of the columns to participate in this imputer's operation but 
        # were initially added to X just to enable better group imputation for those columns that should be handled by this imputer.
        # That is, when transform is completed, and all required imputations are done, these columns are removed from the result just before
        # returned to the pipeline.
        # Therefore, the only reason for these columns to be included in the input X dataframe is for the group imputation. So they must be
        # included in the group_cols list.
        assert len(set(self.group_cols_to_discard) - set(self.group_cols)) == 0, 'columns names to discard on return must be in group_cols'


        # print(f"AggregatedSubGroupsImputer: init: self.columns_to_impute:{self.columns_to_impute} ")
        # print(f"AggregatedSubGroupsImputer: init: self.group_cols_to_discard:{self.group_cols_to_discard} ")
     

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out    

    def _get_most_frequent(self,series):
        # print(f'_get_most_frequent call result: {series.value_counts().idxmax()}, type:{type(series.value_counts().idxmax())}')
        # return series.mode()

        #remove nulls from the given series before searching for the most frequent value
        series = series[series.isna() == False] 

        if series is None or len(series)== 0:
            return None
        else:
            # print(series)
            # https://datascienceparichay.com/article/most-frequent-value-in-a-pandas-column/
            return series.value_counts().idxmax()

    def fit(self, X, y=None):
        # print('doing fit ....')
        # print(f"fit start self.group_cols:{self.group_cols}")
        # print(f"fit start X.columns:{X.columns}")
        # print(f"fit start set(self.group_cols) - set(X.columns):{set(self.group_cols) - set(X.columns)}")

        assert isinstance(X, pd.DataFrame), 'X should be of type pandas DataFrame'
        assert len(set(self.group_cols) - set(X.columns)) == 0,'all columns in group_cols must be included in X.columns'
        assert len(set(self.columns_to_impute) - set(X.columns)) == 0,'all columns in columns_to_impute must be included in X.columns'
        non_categoric_group_cols = [col for col in self.group_cols if is_categorial(X[col]) == False]
        assert len(non_categoric_group_cols) == 0,f'not all group_cols are of categorical type ({non_categoric_group_cols})'
        assert self.group_cols == [] or pd.isnull(X[self.group_cols]).any(axis=None) == False, 'There are missing (null) values in X[group_cols]'

        # print(f"fit start self.include_label_in_group:{self.include_label_in_group}")
        # print(f"fit start X.shape[0]:{X.shape[0]}")
        
        if len(self.columns_to_impute) == 0:
            self.columns_to_impute = list(set(X.columns) - set(self.group_cols))
            if self.label_col_name in self.columns_to_impute:
                self.columns_to_impute = list(set(self.columns_to_impute) - {self.label_col_name})

        # print(f"self.columns_to_impute:{self.columns_to_impute}")
        X_ = X.copy()

        # print(f"columns_to_impute:{self.columns_to_impute}")

        if len(self.group_cols) > 0:
            impute_maps = {}
            for column in self.columns_to_impute:
                if self.strategy == 'most_frequent':
                    impute_maps[column] = X_.groupby(self.group_cols)[column].agg(self._get_most_frequent).reset_index(drop=False)
                else:
                    impute_maps[column] = X_.groupby(self.group_cols)[column].agg(self.strategy).reset_index(drop=False)

            self.impute_maps_ = impute_maps

        for col in self.columns_to_impute: 
            self.simple_imputers[col] = SimpleImputer(strategy=self.strategy,add_indicator=self.add_indicator)
            self.simple_imputers[col].fit(X_[[col]].values)

        # print(f"fit end X.columns:{X.columns}")
        
        self.is_fitted = True

        return self 
    
    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame), 'X should be of type pandas DataFrame'

        # print(f"transform: start X.columns: {X.columns}")
        initial_columns = X.columns
        
        # make sure that the imputer was fitted
        check_is_fitted(self, 'is_fitted')

        # print(f"transform: after is_fitted check X.columns: {X.columns}")

        X = X.copy()

        #create list of columns with null values in X
        columns_null_status = pd.isnull(X).any() 
        columns_with_null_values = list(columns_null_status[columns_null_status==True].index)
        #print(columns_with_null_values)
        cols_with_nulls_to_impute = list(set(self.columns_to_impute) & set(columns_with_null_values))
        #check if there is anything to impute

        # print(f"transform: cols_with_nulls_to_impute:{cols_with_nulls_to_impute}")

        if len(cols_with_nulls_to_impute) > 0:
            if len(self.group_cols) > 0: #self.impute_maps_ is None when len(self.group_cols) == 0 
                for column, impute_map_ in self.impute_maps_.items():
                    if column not in cols_with_nulls_to_impute:
                        continue

                    # print(f"transform: column to impute:{column}")

                    for index, row in impute_map_.iterrows():
                        ind = (X[self.group_cols] == row[self.group_cols]).all(axis=1)
                        # print(row[column])
                        # X.loc[ind, f'{column}_imp'] = 1
                        if self.add_indicator:
                            X.loc[ind, f'{column}_imp'] = [int(x) for x in X.loc[ind, column].isna()]
                        X.loc[ind, column] = X.loc[ind, column].fillna(row[column])

            # print(f"before simple imputers X.columns: {X.columns}")

            #in case of samples that could not be imputed as there are no such similar rows (even not one) 
            # with the same values in the group_cols for which there is a value for the target feature
            if X[self.columns_to_impute].isnull().sum().sum() > 0:
                # print(f'there are null values in: {self.columns_to_impute}')
                # print(f'X[self.columns_to_impute]: {X[self.columns_to_impute]}')
                #loop over the columns which still include null values and impute one by one as we need to 
                #merge the add_indicator column with the one added previousely and any ways we want to be capable of returnning a dataframe rather than a np.array
                columns_null_status = pd.isnull(X).any() 
                columns_with_null_values = list(columns_null_status[columns_null_status==True].index)
                cols_with_nulls_to_impute = list(set(self.columns_to_impute) & set(columns_with_null_values))
                for column in cols_with_nulls_to_impute:
                    # print(np.expand_dims(X[column].values, axis=0))
                    result = np.squeeze(self.simple_imputers[column].transform(X[[column]].values), axis=None)
                    result = pd.DataFrame(result)
                    # print(result.iloc[:,0])
                    X.loc[:,column] = result.iloc[:,0]
                    if self.add_indicator:
                        # print(list(X.columns))
                        if f'{column}_imp' in list(X.columns):
                            #if this column has already been partially imputed in prev impute considering the group_cols
                            X.loc[:,f'{column}_imp'] = [int(x == 1 or z == 1) for x,z in zip(X[f'{column}_imp'].values,result.iloc[:,1])]
                        else:
                            # print(f"column:{column}")
                            # print(f"result:{result}")
                            # print(f"X:{X}")
                            X.loc[:,f'{column}_imp'] = result.iloc[:,1]


        #drop all non required columns from X
        if len(self.group_cols_to_discard) > 0:
            X = X.drop(self.group_cols_to_discard,axis=1)
        
        if self.cols_selector is not None:
            other_non_relevant_cols = [col for col in X.columns if self.cols_selector.is_relevant_col(col,X)==False]
            # print(f"transform: other_non_relevant_cols:{other_non_relevant_cols}")
            if len(other_non_relevant_cols) > 0:
                X = X.drop(other_non_relevant_cols,axis=1)

        self.feature_names_out = list(X.columns)

        # print(f"impute transform end initial_columns:{initial_columns}, out columns: {X.columns}")

        return X
        
########################## Debug Transformer #####################
# def pipeline_debug(df:pd.DataFrame, caller_msg:str=None):
#     caller_msg = '' if caller_msg is None else f" '{caller_msg}'"
#     # print(f"debug{caller_msg}: df.columns:{df.columns}\n")
#     return df

# FunctionTransformer(pipeline_debug, kw_args={'caller_msg':'label'}).fit_transform(pd.DataFrame({'a':[1,2]}))


################################ Pipeline Building Functions #########################
def get_columns_transformer(config:Dict) -> FeatureUnion:
    onehot_cat_cols_extractor = OneHotCategoricalColsExtractor(non_relevant_cols = list(config['ordinal_categorial_features'].keys()), 
                                                       label_col_name = config['label_col'],
                                                       include_label_col = True,
                                                       must_included_cols = [config['sensitive_feature']])

    ordinal_cat_cols_extractor = OrdinalCategoricalColsExtractor(relevant_cols = list(config['ordinal_categorial_features'].keys()),
                                                                         label_col_name = config['label_col'],
                                                                         include_label_col = True,
                                                                         must_included_cols = [config['sensitive_feature']])

    categorical_numerical_preprocessor = FeatureUnion( 
        [
            ('label', pipe(steps=[
                    ('extract_label_col',LabelColExtractor(label_col_name = config['label_col'])),
                    ('label_encoder',LabelCatgoricalEncoder(label_col=config['label_col'],label_col_cat_settings=config['label_ordered_classes'])),
            ])),
            ('num', pipe(steps=[
                    ('extract_non_categorical_cols',NonCategoricalColsExtractor(label_col_name = config['label_col'],
                                                                                include_label_col = True,
                                                                                must_included_cols = [config['sensitive_feature']])), 
                    ('num_ags_imputer',AggregatedSubGroupsImputer(strategy='mean', 
                                                              group_cols = [config['sensitive_feature'],config['label_col']], 
                                                              group_cols_to_discard=[config['sensitive_feature'],config['label_col']], 
                                                              add_indicator=True)),
                    ('num_minmax_scaler',MinMaxScaler()),
            ])),
            ('cat_onehot', pipe(steps=[
                    ('onehot_cat_extractor',onehot_cat_cols_extractor),
                    ('onehot_cat_ags_imputer',
                        AggregatedSubGroupsImputer(strategy='most_frequent',
                                                   group_cols = [config['sensitive_feature'],config['label_col']], 
                                                   group_cols_to_discard=[config['label_col']], 
                                                   add_indicator=True,
                                                   cols_selector = onehot_cat_cols_extractor)),
                    ('onehot_cat_encoder',
                        OneHotEncoder(handle_unknown='ignore', drop='if_binary', sparse=False)),
            ])),
            ('cat_ordinal', pipe(steps=[
                    ('ord_cat_extractor',ordinal_cat_cols_extractor),
                    ('ord_cat_ags_imputer',AggregatedSubGroupsImputer(strategy='most_frequent', 
                                                              group_cols = [config['sensitive_feature'],config['label_col']], 
                                                              group_cols_to_discard=[config['label_col']], 
                                                              add_indicator=True,
                                                              cols_selector = ordinal_cat_cols_extractor)),
                    ('ordinal_encoder',OrdinalCatEncoder(ordinal_cols_selector = ordinal_cat_cols_extractor, ordinal_cat_settings=config['ordinal_categorial_features'])),
            ]))
        ])

    return categorical_numerical_preprocessor

def create_pipeline(config:Dict) -> pipe:
    # SUPER IMPORTANT !! - remove_low_data_columns and numeric_str_feature_encoder returns
    # pandas DataFrame structure, rather than a numpy array. This way we manage to reach
    # categorical_numerical_preprocessor with named columns data structure that, when using
    # "FeatureUnion" for the categorical_numerical_preprocessor, it enables CategoricalColsExtractor 
    # and NonCategoricalColsExtractor to extract the relevant columns (categorical and non categorical) 
    # directly from the received DataFrame passed internally in the pipeline process, rather than from the 
    # initial data DataFrame given in the initial phase of the pipeline (that is, using this approach 
    # we're not relying on the assumption that the already processed data structure, passed in the pipline,
    # has the same columns as in the preprocessed (i.e.initial) data DataFrame).
    #This way our code is much more generic and enables further transformations prior to 
    # categorical_numerical_preprocessor even such that changes the columns set (e.g. add / remove)
    ppl = pipe(steps=
           [
           ('remove_rows_wo_label_value', FunctionTransformer(remove_rows_wo_value_for_column,
                                                        kw_args={'label_col_name': config['label_col']})),
           ('remove_sparse_columns', FunctionTransformer(remove_low_data_columns,
                                                         kw_args={'threshold': config['max_sparse_col_threshold']})),
            ('trim_numeric_str_stabs',FunctionTransformer(numeric_str_feature_encoder,
                                                          kw_args={'numerical_str_features_stubs':config['numerical_str_features_stubs']})),
            ('convert_numeric_cols_to_ord_cat',FunctionTransformer(convert_numeric_cols_to_ord_cat,
                                                                   kw_args={'cont_to_cat_cols_settings':config['numeric_to_ord_cat']})),
            #we remove rows with out value (i.e. null) in the sensitive feature column but in
            # some cases it might be relevant to consider using AggregatedSubGroupsImputer to
            # impute values based on the label (i.e. classification groups) as follows:
            # ('imputer_sf',AggregatedSubGroupsImputer(strategy='most_frequent',
            #                                          group_cols = [config['label_col']],
            #                                          columns_to_impute=[config['sensitive_feature']],
            #                                          add_indicator=True)),
            # ('categorical_numerical_preprocessor', get_columns_transformer(config))
           ('remove_rows_wo_sensitive_feature_value',FunctionTransformer(remove_rows_wo_value_for_column,
                                                           kw_args={'label_col_name': config['sensitive_feature']})),
           ('cat_and_cont_cols_preprocessor', get_columns_transformer(config))
           ])
    
    return ppl

################################ Post data preprocessing #########################

def get_pipeline_final_columns(ppl):
    final_columns = []
    if type(ppl[-1]) == sklearn.pipeline.FeatureUnion:
        print('extract final_columns from FeatureUnion ...')
        transformers_list = ppl[-1].transformer_list
    if type(ppl[-1]) == sklearn.compose.ColumnTransformer:
        print('extract final_columns from ColumnTransformer ...')
        transformers_list = ppl[-1].transformers_

    for i in range(len(transformers_list)):
        initial_columns = []
        for j in range(len(transformers_list[i][1])-1):
            if hasattr(transformers_list[i][1][j],'get_feature_names_out'):
                initial_columns = transformers_list[i][1][j].get_feature_names_out()

        # print(f"{i}: initial columns:{initial_columns}")
        
        last_estimator_in_col_group = transformers_list[i][1][-1]
        if hasattr(last_estimator_in_col_group,'get_feature_names_out'):
            if hasattr(last_estimator_in_col_group,'feature_names_in_'):
                # print(f"{i}: {initial_columns}")
                initial_columns = getattr(last_estimator_in_col_group, "feature_names_in_", None)
            final_columns += list(last_estimator_in_col_group.get_feature_names_out(initial_columns))

        elif hasattr(last_estimator_in_col_group,'transformed_names_'):
            # print(last_estimator_in_col_group.transformed_names_   )
            final_columns += list(last_estimator_in_col_group.transformed_names_)

        else:
            raise Exception('unknown type')

        # print(final_columns)

    return final_columns

def split_data(data:pd.DataFrame, config:Dict, stratify_mode:str='full'):
    assert stratify_mode in ['no_stratify', 'full', 'sensitive_feature', 'label'], \
        "stratify_mode value must be one of the folowing options: ['no_stratify', 'all', 'sensitive_feature', 'label']"

    data = data.copy()

    # shift column 'Name' to first position
    label_col = data.pop(config['label_col'])
    data.insert(0, config['label_col'], label_col)

    rs = 111
    sensitive_feature_col = None
    if stratify_mode in ['sensitive_feature','full']:
        sensitive_feature_col = frns_utils.get_feature_col_from_preprocessed_data(feature_name=config['sensitive_feature'],
                                                                            data=data)

    # data['stratify_col'] = [str(lbl)+str(sstv) for lbl, sstv in zip(data[config['label_col']],data[config['sensitive_feature']])]
    if stratify_mode == 'full':
        data['stratify_col'] = data[config['label_col']].astype(str) + sensitive_feature_col.astype(str)
        try:
            train_df, test_df = train_test_split(data, test_size=0.2,random_state=rs, stratify=data['stratify_col'])
        except BaseException as e:
            train_df, test_df = train_test_split(data, test_size=0.2,random_state=rs, stratify=data[config['label_col']])

        train_df.drop(columns=['stratify_col'],inplace=True)
        test_df.drop(columns=['stratify_col'],inplace=True)
    elif stratify_mode == 'sensitive_feature':
        try:
            train_df, test_df = train_test_split(data, test_size=0.2,random_state=rs, stratify=sensitive_feature_col)
        except BaseException as e:
            train_df, test_df = train_test_split(data, test_size=0.2,random_state=rs)
    elif stratify_mode == 'label':
        train_df, test_df = train_test_split(data, test_size=0.2, random_state=rs, stratify=data[config['label_col']])
    else:
        train_df, test_df = train_test_split(data, test_size=0.2, random_state=rs)

    return(train_df, test_df)


def run_fair_data_preprocess_pipeline(data:pd.DataFrame, config:Dict):
    ppl = create_pipeline(config)
    # categorical_numerical_preprocessor.fit_transform(X_train, y_train)
    utils.save_date_processing_debug(data,"INITIAL_data")

    train_df, test_df = split_data( data=data,
                                    config=config)

    preprocessed_train_data = ppl.fit_transform(train_df)
    final_columns = get_pipeline_final_columns(ppl)

    preprocessed_train_data = pd.DataFrame(data=preprocessed_train_data, columns=final_columns)


    # https://datatofish.com/numpy-array-to-pandas-dataframe/ - following this tutorial, a numpy array containing multiple types of columns values results with all object array.
    # https://stackoverflow.com/questions/61346021/create-a-mixed-type-pandas-dataframe-using-an-numpy-array-of-type-object
    preprocessed_train_data = preprocessed_train_data.convert_dtypes()
    preprocessed_train_data = preprocessed_train_data[sorted(list(preprocessed_train_data.columns), reverse=True)]
    utils.save_date_processing_debug(preprocessed_train_data,"INITIAL_preprocessed_train_data")

    X_train = preprocessed_train_data.drop(columns=[config['label_col']], axis=1)
    y_train = preprocessed_train_data[config['label_col']]
    # print(preprocessed_train_data.head(3))

    ####Run the Data PreProcessing Pipeline on Test Dataset

    preprocessed_test_data = ppl.transform(test_df)
    preprocessed_test_data = pd.DataFrame(data=preprocessed_test_data, columns=final_columns)
    preprocessed_test_data = preprocessed_test_data.convert_dtypes()
    preprocessed_test_data = preprocessed_test_data[sorted(list(preprocessed_test_data.columns), reverse=True)]
    utils.save_date_processing_debug(preprocessed_test_data,"INITIAL_preprocessed_test_data")

    X_test = preprocessed_test_data.drop(columns=[config['label_col']], axis=1)
    y_test = preprocessed_test_data[config['label_col']]

    # print(preprocessed_test_data.head(3))

    return(ppl, preprocessed_train_data, preprocessed_test_data, X_train, X_test, y_train, y_test)

