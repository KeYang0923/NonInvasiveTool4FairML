import warnings
import argparse, os
from multiprocessing import Pool, cpu_count
from joblib import load
import numpy as np
import pandas as pd
from scipy.stats import entropy
from numpy.linalg import norm

warnings.filterwarnings('ignore')

def JSD(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

# def get_correct_n(x):
#     # x is [focus col, compared col, ground truth col]
#     # if focus col is wrong and compared col is correct, return 1. Otherwise, return 0
#     if x.iloc[0] != x.iloc[1] and x.iloc[0] != x.iloc[2] and x.iloc[2]==1:
#         return 1
#     else:
#         return 0
#
# def get_correct_n_sim(x):
#     # x is [focus col, compared col, ground truth col]
#     # if focus col is wrong and compared col is correct, return 1. Otherwise, return 0
#     if x.iloc[0] == 0 and x.iloc[1] ==1 and x.iloc[2]==1:
#         return 1
#     else:
#         return 0
def normalize_weights_distribution(input_weights):
    min_value = min(input_weights)
    shifted_weights = [x + abs(min_value) for x in input_weights]
    max_base = max(shifted_weights)
    return np.array([x / max_base for x in shifted_weights])

def eval_model_diff(data_name, seeds, models, res_path='../intermediate/models/',
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

    res_df = pd.DataFrame(columns=['data', 'seed', 'model', 'group', 'weights_divergence'])

    for seed in seeds:
        for model_name in models:
            # multi model case
            # test_file = '{}test-{}-{}-multi.csv'.format(cur_dir, model_name, seed)
            model_file = '{}{}-{}.joblib'.format(cur_dir, model_name, seed)


            if os.path.exists(model_file):
                base_pipe = load(model_file)
                base_model = base_pipe.best_estimator_.named_steps['learner']
                base_weights = base_model.coef_[0] + base_model.intercept_
                norm_base = normalize_weights_distribution(base_weights)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract evaluation results for difference between models")
    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    parser.add_argument("--data", type=str, default='all',
                        help="extract results for all the datasets as default. Otherwise, only extract the results for the input dataset.")
    parser.add_argument("--model", type=str, default='lr',
                        help="extract results for all the models as default. Otherwise, only extract the results for the input model from ['lr', 'tr'].")
    parser.add_argument("--exec_n", type=int, default=20,
                        help="number of executions with different random seeds. Default is 20.")
    args = parser.parse_args()

    seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 7777, 100,
             923]
    models = ['lr', 'tr']

    datasets = ['lsac', 'cardio', 'bank', 'meps16', 'credit', 'ACSE', 'ACSP', 'ACSH', 'ACSM', 'ACSI']
    if args.data == 'all':
        pass
    elif args.data in datasets:
        datasets = [args.data]
    else:
        raise ValueError('The input "data" is not valid. CHOOSE FROM ["lsac", "cardio", "bank", "meps16", "credit", "ACSE", "ACSP", "ACSH", "ACSM", "ACSI"].')

    if args.model == 'all':
        pass
    elif args.model in models:
        models = [args.model]
    else:
        raise ValueError('The input "model" is not valid. CHOOSE FROM ["lr", "tr"].')

    if args.exec_n is None:
        raise ValueError('The input "exec_n" is requried. Use "--exec_n 1" for a single execution.')
    elif type(args.exec_n) == str:
        raise ValueError('The input "exec_n" requires integer. Use "--exec_n 1" for a single execution.')
    else:
        n_exec = int(args.exec_n)
        seeds = seeds[:n_exec]

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    res_path = repo_dir + '/intermediate/models/'

    if args.run == 'parallel':
        tasks = []
        for data_name in datasets:
            tasks.append([data_name, seeds, models, res_path])
        with Pool(cpu_count()//2) as pool:
            pool.starmap(eval_model_diff, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')