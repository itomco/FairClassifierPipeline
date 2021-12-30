import json
from typing import *

def create_german_config(config_name:str='german'):
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
    config['data_path'] = './Data/german.data'

    save_config(config_name=config_name,config=config)

def create_bank_config(config_name:str='bank'):
    config = {}
    config['sensitive_features'] = ["age","marital"]
    config['label_col'] = 'y'
    config['label_ordered_classes'] = (['yes', 'no'], [1, 0])  # relevant for Fairness Positive Label Value

    config['numerical_str_features_stubs'] = {}
    # config['numerical_far_feature'] =    {'pdays':[-1]}
    config['ordinal_categorial_features'] = {
        # 'education':['unknown','primary', 'secondary', 'tertiary'],
        'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
        # 'contact':['unknown','telephone','cellular'],
        # 'poutcome':['failure','other','unknown','success']
    }

    config['numeric_to_ord_cat'] = {'age': [30, 60]}

    config['max_sparse_col_threshold'] = 0.8
    config['data_path'] = './Data/bank.csv'

    save_config(config_name=config_name,config=config)

def save_config(config_name:str,config:Dict):
    with open(f'configs/{config_name}.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


def create_configs(bank_cfg_name:str='bank',
                   german_cfg_name:str='german'):
    create_german_config(config_name=german_cfg_name)
    create_bank_config(config_name=bank_cfg_name)