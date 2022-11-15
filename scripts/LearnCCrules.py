# Optimize the CC's input and learn rules from training data using CCs

import warnings

import argparse
from multiprocessing import Pool, cpu_count
import pandas as pd
import os, json
from sklearn.neighbors import KernelDensity
import prose.datainsights as di


warnings.filterwarnings('ignore')

def read_json(file_name_with_path):
    if os.path.isfile(file_name_with_path):
        with open(file_name_with_path) as f:
            res = json.load(f)
        return res
    else:
        raise ValueError('Not exist', file_name_with_path)

def combine_violation(x):
    idx_map = {'00': 0, '01': 1, '10': 2, '11': 3}
    cur = '{}{}'.format(int(x.iloc[0]), int(x.iloc[1]))
    return x.iloc[2 + idx_map[cur]]


def learn_cc_models(data_name, seed, dense_kernal='guassian',
                    res_path='../intermediate/models/',
                    data_path='../data/processed/',
                    set_suffix='S_1',
                    n_groups=2, n_labels=2, sensi_col='A', y_col='Y',
                    dense_n=0.2, dense_h=0.1, algorithm='auto'):


    cur_dir = res_path + data_name + '/'

    train_df = pd.read_csv(cur_dir + '-'.join(['train', str(seed), set_suffix]) + '.csv')
    test_df = pd.read_csv(cur_dir + '-'.join(['test', str(seed), set_suffix]) + '.csv')
    meta_info = read_json(data_path + data_name + '.json')
    n_cond_features = len(meta_info['continuous_features'])

    cc_cols = ['X{}'.format(i) for i in range(1, n_cond_features+1)]
    cc_df = train_df.copy()
    cc_df[cc_cols] = (cc_df[cc_cols] - cc_df[cc_cols].mean()) / cc_df[cc_cols].std()

    cc_test_df = test_df.copy()
    cc_test_df[cc_cols] = (cc_test_df[cc_cols] - cc_test_df[cc_cols].mean()) / cc_test_df[cc_cols].std()

    for group_i in range(n_groups):
        for label_i in range(n_labels):
            group_input = cc_df[(cc_df[sensi_col] == group_i) & (cc_df[y_col] == label_i)]

            group_X = group_input[cc_cols].to_numpy()
            kde = KernelDensity(bandwidth=dense_h, kernel=dense_kernal, algorithm=algorithm)
            kde.fit(group_X)

            group_input['density'] = kde.score_samples(group_X)
            group_input.sort_values(by=['density'], ascending=False, inplace=True)
            cc_input = group_input.head(int(dense_n * group_input.shape[0]))

            group_cc_rules = di.learn_assertions(cc_input[cc_cols], max_self_violation=1.0)
            train_cc_res = group_cc_rules.evaluate(cc_df[cc_cols], explanation=True, normalizeViolation=True)
            train_df['vio_G{}_L{}'.format(group_i, label_i)] = train_cc_res.row_wise_violation_summary['violation']

            test_cc_res = group_cc_rules.evaluate(cc_test_df[cc_cols], explanation=True, normalizeViolation=True)
            test_df['vio_G{}_L{}'.format(group_i, label_i)] = test_cc_res.row_wise_violation_summary['violation']

    train_df['vio_cc'] = train_df[[sensi_col, y_col, 'vio_G0_L0', 'vio_G0_L1', 'vio_G1_L0', 'vio_G1_L1']].apply(lambda x: combine_violation(x), axis=1)

    train_df.to_csv(cur_dir + '-'.join(['train_vio', str(seed)]) + '.csv', index=False)
    test_df.to_csv(cur_dir+'-'.join(['test_vio', str(seed)])+'.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Optimize CC and learn rules from training data using CCs")

    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    # parameters for running over smaller number of datasets and few number of executions
    parser.add_argument("--set_n", type=int, default=9,
                        help="number of datasets over which the script is running. Default is 9 for all the datasets.")
    parser.add_argument("--exec_n", type=int, default=5,
                        help="number of executions with different random seeds. Default is 20.")
    args = parser.parse_args()

    datasets = ['cardio', 'bank', 'meps16', 'lsac', 'credit', 'ACSE', 'ACSP', 'ACSM', 'ACSI']
    # datasets = ['lsac']

    seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 0xdeadcafe,
             0xdeadbeef, 0xbeefcafe]

    if args.set_n is None:
        raise ValueError(
            'The input "set_n" is requried. Use "--set_n 1" for running over a single dataset.')
    elif type(args.set_n) == str:
        raise ValueError(
            'The input "set_n" requires integer. Use "--set_n 1" for running over a single dataset.')
    else:
        n_datasets = int(args.set_n)
        if n_datasets == -1:
            datasets = datasets[n_datasets:]
        else:
            datasets = datasets[:n_datasets]

    if args.exec_n is None:
        raise ValueError(
            'The input "exec_n" is requried. Use "--exec_n 1" for a single execution.')
    elif type(args.exec_n) == str:
        raise ValueError(
            'The input "exec_n" requires integer. Use "--exec_n 1" for a single execution.')
    else:
        n_exec = int(args.exec_n)
        seeds = seeds[:n_exec]

    res_path = '../intermediate/models/'

    if args.run == 'parallel':
        tasks = []
        for data_name in datasets:
            if data_name in ['cardio', 'ACSM', 'ACSI']:
                kernel_name = 'gaussian'
            elif data_name in ['bank', 'lsac', 'meps16', 'ACSP']:
                kernel_name = 'exponential'
            elif data_name in ['credit', 'ACSE']:
                kernel_name = 'tophat'
            else: # for ACSH
                kernel_name = 'epanechnikov'
            for seed in seeds:
                tasks.append([data_name, seed, kernel_name, res_path])
        with Pool(cpu_count()//2) as pool:
            pool.starmap(learn_cc_models, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')