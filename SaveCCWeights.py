import warnings
import argparse
from multiprocessing import Pool, cpu_count
import pandas as pd
import os
from sklearn.neighbors import KernelDensity
import prose.datainsights as di
from PrepareData import read_json

warnings.filterwarnings('ignore')

def save_cc_weights(data_name, seed, dense_kernal='guassian',
                    res_path='../intermediate/models/',
                    data_path='data/processed/',
                    n_groups=2, n_labels=2, sensi_col='A', y_col='Y',
                    dense_n=0.2, dense_h=0.1, algorithm='auto'):
    repo_dir = res_path.replace('intermediate/models/', '')
    cur_dir = res_path + data_name + '/'

    train_df = pd.read_csv('{}train-{}.csv'.format(cur_dir, seed))

    meta_info = read_json('{}/{}{}.json'.format(repo_dir, data_path, data_name))
    n_cond_features = len(meta_info['continuous_features'])

    cc_cols = ['X{}'.format(i) for i in range(1, n_cond_features+1)]
    cc_df = train_df.copy()
    cc_df[cc_cols] = (cc_df[cc_cols] - cc_df[cc_cols].mean()) / cc_df[cc_cols].std()

    for group_i in range(n_groups):
        for label_i in range(n_labels):
            group_input = cc_df[(cc_df[sensi_col] == group_i) & (cc_df[y_col] == label_i)]

            group_X = group_input[cc_cols].to_numpy()
            kde = KernelDensity(bandwidth=dense_h, kernel=dense_kernal, algorithm=algorithm)
            kde.fit(group_X)

            group_input['density'] = kde.score_samples(group_X)
            group_input.sort_values(by=['density'], ascending=False, inplace=True)
            cc_input = group_input.head(int(dense_n * group_input.shape[0]))
            cc_input.to_csv('{}input-cc-{}.csv'.format(cur_dir, seed), index=False)

            group_cc_rules = di.learn_assertions(cc_input[cc_cols], max_self_violation=1.0)
            train_cc_res = group_cc_rules.evaluate(cc_df[cc_cols], explanation=True, normalizeViolation=True)

            weights_matrix = train_cc_res.assertions.constrained_invariants[0].data_assertion.inv_matrix.T
            weights_cols = train_cc_res.assertions.features + ['intercept']
            weights_df = pd.DataFrame(columns=weights_cols, data=weights_matrix)
            weights_df.to_csv('{}weights-cc-{}-G{}-L{}.csv'.format(cur_dir, seed, group_i, label_i), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Learn CC rules from training data")

    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    # parameters for running over smaller number of datasets and few number of executions
    parser.add_argument("--data", type=str, default='all',
                        help="name of datasets over which the script is running. Default is for all the datasets.")
    parser.add_argument("--set_n", type=int, default=None,
                        help="number of datasets over which the script is running. Default is 10.")
    parser.add_argument("--exec_n", type=int, default=20,
                        help="number of executions with different random seeds. Default is 20.")
    args = parser.parse_args()

    datasets = ['meps16', 'lsac', 'bank', 'cardio', 'ACSM', 'ACSP', 'credit', 'ACSE', 'ACSH', 'ACSI']

    seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 7777, 100, 923]

    if args.data == 'all':
        pass
    elif args.data in datasets:
        datasets = [args.data]
    else:
        raise ValueError('The input "data" is not valid. CHOOSE FROM ["lsac", "cardio", "bank", "meps16", "credit", "ACSE", "ACSP", "ACSH", "ACSM", "ACSI"].')


    if args.set_n is not None:
        if type(args.set_n) == str:
            raise ValueError(
                'The input "set_n" requires integer. Use "--set_n 1" for running over a single dataset.')
        else:
            n_datasets = int(args.set_n)
            if n_datasets < 0:
                datasets = datasets[n_datasets:]
            elif n_datasets > 0:
                datasets = datasets[:n_datasets]
            else:
                raise ValueError(
                    'The input "set_n" requires non-zero integer. Use "--set_n 1" for running over a single dataset.')
    if args.exec_n is None:
        raise ValueError(
            'The input "exec_n" is requried. Use "--exec_n 1" for a single execution.')
    elif type(args.exec_n) == str:
        raise ValueError(
            'The input "exec_n" requires integer. Use "--exec_n 1" for a single execution.')
    else:
        n_exec = int(args.exec_n)
        seeds = seeds[:n_exec]

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    res_path = repo_dir + '/intermediate/models/'

    if args.run == 'parallel':
        tasks = []
        for data_name in datasets:
            if data_name in ['cardio', 'ACSI']:
                kernel_name = 'gaussian'
            elif data_name in ['credit', 'bank', 'lsac', 'meps16', 'ACSM', 'ACSE', 'ACSP', 'ACSH']: #current optimal
                kernel_name = 'exponential'
            else:
                raise ValueError('The input "data" is not valid. CHOOSE FROM ["lsac", "cardio", "bank", "meps16", "credit", "ACSE", "ACSP", "ACSH", "ACSM", "ACSI"].')

            for seed in seeds:
                tasks.append([data_name, seed, kernel_name, res_path])
        with Pool(cpu_count()//2) as pool:
            pool.starmap(save_cc_weights, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')