import warnings
import argparse, os
from multiprocessing import Pool, cpu_count
from joblib import load
import numpy as np
import pandas as pd
from scipy.stats import entropy
from numpy.linalg import norm

warnings.filterwarnings('ignore')

def sum_cc_diff(data_name, seeds, models, res_path='../intermediate/models/',
                       eval_path='eval/'):

    repo_dir = res_path.replace('intermediate/models/', '')
    eval_path = repo_dir + eval_path

    cur_dir = res_path + data_name + '/'
    # res_df = pd.DataFrame(columns=['data', 'seed', 'test_n', 'g0_n', 'model', 'case', 'correct_n', 'correct_g0_n'])
    # diff_n
    # cases_cols = [['Y_pred', 'Y_pred_A'], ['Y_pred', 'Y_pred_min'], ['Y_pred', 'Y_pred_w1'], ['Y_pred', 'Y_pred_w2']]
    # cases_names = ['sep_orig', 'mcc_min_orig', 'mcc_w1_orig', 'mcc_w2_orig']

    # diff_n_sep
    # cases_cols = [['Y_pred', 'Y_pred_A'], ['Y_pred_A', 'Y_pred_min'], ['Y_pred_A', 'Y_pred_w1'], ['Y_pred_A', 'Y_pred_w2']]
    # cases_names = ['sep_orig', 'mcc_min_sep', 'mcc_w1_sep', 'mcc_w2_sep']

    res_df = pd.DataFrame(columns=['data', 'seed', 'comparison', 'distance'])

    for seed in seeds:

        # compare between groups

        for label_i in range(2):
            weight_g0 = pd.read_csv('{}weights-cc-{}-G0-L{}.csv'.format(cur_dir, seed, label_i))
            weight_g1 = pd.read_csv('{}weights-cc-{}-G1-L{}.csv'.format(cur_dir, seed, label_i))
            # todo: update to compute average stds of the learned CC rules



                for group_i in [0, 1]:
                    group_model_file = '{}{}-{}-G{}.joblib'.format(cur_dir, model_name, seed, group_i)
                    if os.path.exists(group_model_file):
                        g_pipe = load(group_model_file)
                        g_model = g_pipe.best_estimator_.named_steps['learner']
                        g_weights = g_model.coef_[0] + g_model.intercept_

                        norm_g = normalize_weights_distribution(g_weights)
                        dist_g = JSD(norm_base, norm_g)
                        res_df.loc[res_df.shape[0]] = [data_name, seed, model_name, 'G{}'.format(group_i), dist_g]
                    else:
                        print('--> No group model ', group_model_file)
                # test_df = pd.read_csv(test_file)
                # base_row = [data_name, seed, test_df.shape[0]]
                #
                # for case_i, cols_i in zip(cases_names, cases_cols):
                #     test_df[case_i] = test_df[cols_i+['Y']].apply(lambda x: get_correct_n(x), axis=1)
                #     case_correct_n = sum(test_df[case_i])
                #
                #     group_df = test_df.query('A==0')
                #     case_g_correct_n = sum(group_df[case_i])
                #
                #     res_df.loc[res_df.shape[0]] = base_row + [group_df.shape[0], model_name, case_i, case_correct_n, case_g_correct_n]
            else:
                print('--> No single model ', model_file)
    res_df.to_csv('{}dist-{}.csv'.format(eval_path, data_name), index=False)
    print('Model distance is saved at', '{}dist-{}.csv'.format(eval_path, data_name))
