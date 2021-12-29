import pandas as pd

def load_bank_data(data_path:str):
    return pd.read_csv(data_path, delimiter=";",header='infer')

def load_german_data(data_path:str):
    names = ['existingchecking', 'duration', 'credithistory', 'purpose', 'creditamount',
             'savings', 'employmentsince', 'installmentrate', 'statussex', 'otherdebtors',
             'residencesince', 'property', 'age', 'otherinstallmentplans', 'housing',
             'existingcredits', 'job', 'peopleliable', 'telephone', 'foreignworker', 'classification']

    return pd.read_csv(data_path, names=names, delimiter=' ')


def load_data(project_mode:str, data_path:str):
    if project_mode == 'german':
        data = load_german_data(data_path=data_path)
    else:
        data = load_bank_data(data_path=data_path)
    return data