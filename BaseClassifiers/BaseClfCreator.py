from BaseClassifiers.BaseClf import BaseClf
from BaseClassifiers.GermanCreditBaseClf import GermanBaseClf
from BaseClassifiers.BankMarketingBaseClf import BankBaseClf

def create_base_clf(project_mode:str) -> BaseClf:
    assert project_mode in ['german', 'bank'], 'unknown project mode'

    if project_mode == 'german':
        return GermanBaseClf()
    else:
        return BankBaseClf()