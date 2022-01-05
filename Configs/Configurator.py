import json
from typing import *


def create_german_config(config_name: str = 'german'):
    config = {}
    config['sensitive_features'] = ['statussex', 'age']
    config['label_col'] = 'classification'
    config['label_ordered_classes'] = ([1, 2], [1, 0])  # relevant for Fairness Positive Label Value
    config['numerical_str_features_stubs'] = {
        'otherinstallmentplans': ('A14', None),
        'credithistory': ('A3', None),
        'employmentsince': ('A7', None),
        'otherdebtors': ('A10', None),
        'property': ('A12', None),
        'housing': ('A15', None),
        'job': ('A17', None),
    }
    config['ordinal_categorial_features'] = {'existingchecking': ['A14', 'A11', 'A12', 'A13'],
                                            'savings': ['A65', 'A61', 'A62', 'A63', 'A64']}
    config['numeric_to_ord_cat'] = {'age': [25, 50]}
    config['fairness_metrics'] = ['AOD', 'EOD']
    config['target_fairness_metric'] = 'AOD'
    config['max_sparse_col_threshold'] = 0.2
    config['data_path'] = './Data/german.data'

    save_config(config_name=config_name, config=config)

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

    config['numeric_to_ord_cat'] = {'age': [28, 38, 48, 58, 68, 78]}

    config['fairness_metrics'] = ['AOD', 'EOD']
    config['target_fairness_metric'] = 'AOD'

    config['max_sparse_col_threshold'] = 0.2
    config['data_path'] = './Data/bank.csv'

    save_config(config_name=config_name,config=config)

def save_config(config_name:str,config:Dict):
    with open(f'Configs/{config_name}.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


def create_configs(bank_cfg_name:str='bank',
                   german_cfg_name:str='german'):
    create_german_config(config_name=german_cfg_name)
    create_bank_config(config_name=bank_cfg_name)