#Importing required libraries
import json
import pandas as pd
import numpy as np
from typing import *
from datetime import datetime

np.random.seed(sum(map(ord, "aesthetics")))

from sklearn.metrics import classification_report
from FairClassifierPipeline import FairPipeline as fair_ppl
from FairClassifierPipeline import Utils as utils
from BaseClassifiers.BaseClf import BaseClf
from BaseClassifiers import BaseClfCreator
from FairClassifierPipeline.FairClassifier import FairClassifier
from FairClassifierPipeline import FairnessUtils as frns_utils
from Configs import Configurator as cfg
from Data import DataLoader as data_loader

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
                                           data:pd.DataFrame,
                                           do_plots:bool=True
                                           ):

    sensitive_features_names = config['sensitive_features']
    results = {}

    for sf in sensitive_features_names:
        print(f'Showcase pipeline impact on base model performance considering Sensitive Feature: ###- {sf.upper()} -###')

        snsftr_frns_mtrcs_w_base_preprocess = {}
        snsftr_frns_mtrcs_w_fair_pipeline = {}
        snsftr_cm_w_base_preprocess = {}
        snsftr_cm_w_fair_pipeline = {}

        config['sensitive_feature'] = sf

        #Execute Baseline XGBoost Classifier
        base_X_train, base_X_test, base_y_train, base_y_test, base_model, base_y_pred, base_y_pred_proba = \
                                                                    base_clf.run_baseline(data=data.copy(),
                                                                                          config=config)

        _, _, base_auc = utils.get_roc(y_test=base_y_test,
                                                     y_pred=base_y_pred)

        print('####### Base model ######### Confusion Matirx:')
        utils.print_confusion_matrix(base_y_test,base_y_pred, base_y_pred_proba, do_plot=do_plots)

        clsf_rprt = classification_report(base_y_test, pd.Series(base_y_pred), digits=4, output_dict=True)
        snsftr_cm_w_base_preprocess.update({f'{sf}:accuracy': clsf_rprt['accuracy'],
                                            f"{sf}:TPR['1']": clsf_rprt['1']['recall'],
                                            f"{sf}:FPR['1']": 1-clsf_rprt['0']['recall'],
                                            f'{sf}:macro_avg-precision': clsf_rprt['macro avg']['precision'],
                                            f'{sf}:macro_avg-recall': clsf_rprt['macro avg']['recall'],
                                            f'{sf}:macro_avg-f1-score': clsf_rprt['macro avg']['f1-score'],
                                            f'{sf}:AUC': base_auc})

        if sf in data.columns and fair_ppl.is_categorial(data[sf]):
            for frns_mtrc in fairness_metrics:
                frns_mtrc = frns_mtrc.lower()
                snsftr_frns_mtrcs_w_base_preprocess.update(frns_utils.get_fairness_score_for_sensitive_features(sensitive_features_names= [sf],
                                                                                                        fairness_metric=frns_mtrc,
                                                                                                        y_true=base_y_test,
                                                                                                        y_pred=pd.Series(base_y_pred),
                                                                                                        data=base_X_test))
        else:
            # skip sensitive feature with continues values as base model's data preprocessing
            # does not convert it to categorical feature as our pipeline does
            for frns_mtrc in fairness_metrics:
                frns_mtrc = frns_mtrc.lower()
                snsftr_frns_mtrcs_w_base_preprocess[f'{sf}:{frns_mtrc}'] = -1 #not measured




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

        _, _, initial_auc = utils.get_roc(y_test= initial_y_test,
                                                                 y_pred= initial_y_pred)

        print('####### Initial model ######### Confusion Matirx:')
        utils.print_confusion_matrix(utils.to_int_srs(initial_y_test),initial_y_pred, initial_y_pred_proba, do_plot=do_plots)

        clsf_rprt = classification_report(initial_y_test, pd.Series(initial_y_pred), digits=4, output_dict=True)
        snsftr_cm_w_fair_pipeline.update({f'{sf}:accuracy':clsf_rprt['accuracy'],
                                          f"{sf}:TPR['1']": clsf_rprt['1']['recall'],
                                          f"{sf}:FPR['1']": 1-clsf_rprt['0']['recall'],
                                          f'{sf}:macro_avg-precision':clsf_rprt['macro avg']['precision'],
                                          f'{sf}:macro_avg-recall':clsf_rprt['macro avg']['recall'],
                                          f'{sf}:macro_avg-f1-score':clsf_rprt['macro avg']['f1-score'],
                                          f'{sf}:AUC': initial_auc})


        for frns_mtrc in fairness_metrics:
            snsftr_frns_mtrcs_w_fair_pipeline.update(frns_utils.get_fairness_score_for_sensitive_features(sensitive_features_names = [sf],
                                                                                                  fairness_metric=frns_mtrc,
                                                                                                  y_true=initial_y_test,
                                                                                                  y_pred=pd.Series(initial_y_pred),
                                                                                                  data=initial_X_test))

        base_vs_initial_eod_results = pd.DataFrame([snsftr_frns_mtrcs_w_base_preprocess,
                                                    snsftr_frns_mtrcs_w_fair_pipeline]).T

        base_vs_initial_macro_avg_cm_resuls = pd.DataFrame([snsftr_cm_w_base_preprocess,
                                                           snsftr_cm_w_fair_pipeline]).T

        base_vs_initial_eod_results.columns = ['base','initial']
        base_vs_initial_macro_avg_cm_resuls.columns = ['base','initial']
        result = pd.concat([base_vs_initial_eod_results,base_vs_initial_macro_avg_cm_resuls],axis=0)
        print(f"Base model vs Initial model performance comparison for sensitive feature '{sf}':\n{result}")
        print('\n######################################################################################################################################\n')

        results[sf] = result

    return results

if __name__ == '__main__':
    do_plots = False
    for project_mode in ['bank','german']:
        # project_mode = 'german' # select 'bank' or 'german'

        ####-0 select config
        config = load_config(config_name=project_mode)

        fairness_metrics = config['fairness_metrics']
        target_fairness_metric = config['target_fairness_metric']

        ####-1. Load data
        print('\n####### I. Load Data ################################################################################################################ \n')
        data = data_loader.load_data(project_mode=project_mode, data_path=config['data_path'])
        print(f'{project_mode.upper()} data loaded:')
        print(data.head(3))

        base_clf:BaseClf = BaseClfCreator.create_base_clf(project_mode=project_mode)

        ####-2. Check fair pipeline impact on base model
        print('\n####### II. Check fair pipeline impact on base model #################################################################################### \n')
        base_vs_initial_macro_avg_cm_results = showcase_pipeline_impact_on_base_model(config=config,
                                                                                       fairness_metrics = fairness_metrics,
                                                                                       base_clf=base_clf,
                                                                                       data=data.copy(),
                                                                                        do_plots = do_plots)

        ####-3. search for most fairness biased sensitive feature
        print('\n####### III. search for most fairness biased sensitive feature ############################################################################ \n')

        sensitive_feature = FairClassifier.get_most_biased_sensitive_feature(data=data.copy(),
                                                                       fairness_metric=target_fairness_metric,
                                                                       base_clf=base_clf,
                                                                       config=config)

        print(f"Sensitive feature with highest un-fair bias based on fairness metric '{target_fairness_metric}' is: {sensitive_feature} ")


        ####-4. find privileged and unprivileged groups in sensitive feature
        print('\n####### IV. find privileged and unprivileged groups in sensitive feature ################################################################### \n')

        config = load_config(config_name=project_mode)
        config['sensitive_feature'] = sensitive_feature

        ppl, preprocessed_train_data, preprocessed_test_data, X_train, X_test, y_train, y_test = \
                                                                fair_ppl.run_fair_data_preprocess_pipeline(data=data.copy(), config=config)

        X_train = utils.to_float_df(X_train)
        y_train = utils.to_int_srs(y_train)
        X_test = utils.to_float_df(X_test)
        y_test = utils.to_int_srs(y_test)

        xgb_clf = base_clf.fit(X_train=X_train,
                                y_train=y_train)

        y_pred, y_pred_proba = base_clf.predict(clf=xgb_clf,
                                                      X=X_train)
        y_pred = utils.to_int_srs(pd.Series(y_pred))

        sensitive_feature_srs = frns_utils.get_feature_col_from_preprocessed_data(feature_name=sensitive_feature,
                                                                                data= X_train)
        snsftr_groups_slctnrt, snsftr_slctrt_sub_groups = \
            frns_utils.get_feature_sub_groups_by_selection_rate( y_true= y_train,
                                                                 y_pred= y_pred,
                                                                 sensitive_feature_srs = sensitive_feature_srs)
        unprev_size = len(sensitive_feature_srs[sensitive_feature_srs.isin(pd.Series(list(snsftr_slctrt_sub_groups[0])))])
        prev_size = len(sensitive_feature_srs[sensitive_feature_srs.isin(pd.Series(list(snsftr_slctrt_sub_groups[1])))])
        snsftr_sub_groups = ((unprev_size,snsftr_slctrt_sub_groups[0]),(prev_size,snsftr_slctrt_sub_groups[1]))

        print(f"snsftr_slctrt_sub_groups: prev=[{snsftr_slctrt_sub_groups[1]},{prev_size}], unprev=[{snsftr_slctrt_sub_groups[0]},{unprev_size}]\n")
        print(f"snsftr_groups_slctnrt:\n{snsftr_groups_slctnrt}\n")

        ####-5. run gridsearch_cv with anomaly samples removal
        print('\n####### V. Create & Fit a FairClassifier model ################################################################################################ \n')
        fair_clf = FairClassifier(target_fairness_metric = target_fairness_metric,
                                   base_clf=base_clf,
                                   verbose=False)

        fair_clf.fit(X_train = X_train,
                     y_train = y_train,
                     sensitive_feature_name=sensitive_feature,
                     snsftr_slctrt_sub_groups=snsftr_sub_groups)

        pipe_cv = fair_clf.pipe_cv
        results = pd.DataFrame(pipe_cv.cv_results_)
        datetime_tag = datetime.now().strftime("%y%m%d_%H%M%S")
        results.to_csv(f'./gscv_results/{datetime_tag}_{project_mode}_{target_fairness_metric}_pipe_cv.cv_results_.csv')
        print(f'results:\n{results.head(3)}')

        fair_clf_y_pred = fair_clf.predict(X_test)
        fair_clf_y_pred_proba = fair_clf.predict_proba(X_test)


        ####-6. Check top fairness aware models' performance on un-seen (Test) data
        print("\n####### VI. Check top fairness aware models' performance on un-seen (Test) data #################################################################### \n")
        top_models_scores_on_test = fair_clf.retrain_top_models_and_get_performance_metrics(X_train=X_train,
                                                                                            y_train=y_train,
                                                                                            X_test=X_test,
                                                                                            y_test=y_test,
                                                                                            max_num_top_models=results.shape[0],
                                                                                            target_metrics_thresholds={target_fairness_metric:1.0,'f1_macro':0.0})



        # print(top_models_scores_on_test)
        top_models_scores_on_test_df = pd.DataFrame(top_models_scores_on_test).sort_values(by=[target_fairness_metric.lower(),'f1_macro'], ignore_index=True)
        print(top_models_scores_on_test_df.head(10))
        top_models_scores_on_test_df.to_csv(f'./gscv_results/{datetime_tag}_{project_mode}_{target_fairness_metric}_top_models_scores_on_test_df.csv')


        ####-7. print faire classifier final results
        _, _, fair_clf_auc = utils.get_roc(y_test= y_test,
                                           y_pred= fair_clf_y_pred)

        print('\n\n####### Fair Classifier model ######### Confusion Matirx:')
        utils.print_confusion_matrix(y_test,fair_clf_y_pred, fair_clf_y_pred_proba, do_plot=do_plots)


        best_fair_clf_aod = top_models_scores_on_test_df['aod'][0]
        best_fair_clf_eod = top_models_scores_on_test_df['eod'][0]
        fair_clf_results = {f'{sensitive_feature}:aod':best_fair_clf_aod, f'{sensitive_feature}:eod':best_fair_clf_eod}

        clsf_rprt = classification_report(y_test, pd.Series(fair_clf_y_pred), digits=4, output_dict=True)
        fair_clf_results.update({f'{sensitive_feature}:accuracy':clsf_rprt['accuracy'],
                              f"{sensitive_feature}:TPR['1']": clsf_rprt['1']['recall'],
                              f"{sensitive_feature}:FPR['1']": 1-clsf_rprt['0']['recall'],
                              f'{sensitive_feature}:macro_avg-precision':clsf_rprt['macro avg']['precision'],
                              f'{sensitive_feature}:macro_avg-recall':clsf_rprt['macro avg']['recall'],
                              f'{sensitive_feature}:macro_avg-f1-score':clsf_rprt['macro avg']['f1-score'],
                              f'{sensitive_feature}:AUC': fair_clf_auc})

        base_vs_initial_vs_fair_clf = base_vs_initial_macro_avg_cm_results[sensitive_feature].copy()
        base_vs_initial_vs_fair_clf['FairClf'] = pd.Series(fair_clf_results)

        print(base_vs_initial_vs_fair_clf)

        print('\n\n####################################################################################################################################\n\n')
