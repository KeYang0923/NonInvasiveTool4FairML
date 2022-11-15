# Simulate erroneous records in real datasets for the evaluation of multiCC under a fixed error rate

import warnings

import argparse
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
from utils import save_json

warnings.filterwarnings('ignore')

def simulate_drift_data_train(data_name, seed, y_col, sensi_col, res_path='../intermediate/models/', error_k=0.2):

    cur_dir = res_path + data_name + '/'

    train_df = pd.read_csv(cur_dir + '-'.join(['train', str(seed)]) + '.csv')
    train_df.drop(columns=['Unnamed: 0'], inplace=True)

    np.random.seed(seed)
    sampled_indices = np.random.rand(len(train_df)) < error_k

    error_indexes = list(train_df.loc[sampled_indices].index)


    # switch the values of the sensitive attribute
    def reverse_sensi_col(x, cand_list):
        if x.iloc[0] in cand_list:
            return abs(x.iloc[1] - 1)
        else:
            return x.iloc[1]

    train_df['ID_temp'] = train_df.index
    train_df[sensi_col] = train_df[['ID_temp', sensi_col]].apply(lambda x: reverse_sensi_col(x, error_indexes), axis=1)
    train_df.drop(columns=['ID_temp'], inplace=True)

    # print(len(error_indexes), sum(list(test_df[sensi_col] != test_vio_df[sensi_col])))

    save_json({'Error_ID': error_indexes}, '{}train_errorID-{}-error{:.2f}.csv'.format(cur_dir, seed, error_k))


    train_df.to_csv('{}train-{}-error{:.2f}.csv'.format(cur_dir, seed, error_k))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simulate errors in test set under a fixed rate for all the datasets")
    parser.add_argument("--setting", type=str,
                        help="error rate of multiCC. When is not None, multiCC is run on test set with errors in sensitive attribute. Choose from [error0.01, error0.05, error0.1, error0.15, error0.2] for different rates of errors in the test set.")

    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    # parameters for running over smaller number of datasets and few number of executions
    parser.add_argument("--set_n", type=int, default=1,
                        help="number of datasets over which the script is running. Default is 9 for all the datasets.")
    parser.add_argument("--exec_n", type=int, default=20,
                        help="number of executions with different random seeds. Default is 20.")
    args = parser.parse_args()

    # datasets = ['adult', 'german', 'compas', 'cardio', 'bank', 'meps16', 'lawgpa', 'credit', 'UFRGS']
    # y_cols = ['Income Binary', 'credit', 'two_year_recid'] + ['Y' for i in range(6)]
    # sensi_cols = ['sex', 'age', 'race'] + ['C0' for i in range(6)]

    datasets = ['lawgpa']
    y_cols = ['Y']
    sensi_cols = ['C0']

    # seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 0xdeadcafe, 0xdeadbeef, 0xbeefcafe]
    seeds = [6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 0xdeadcafe, 0xdeadbeef, 0xbeefcafe]

    res_path = '../intermediate/models/'
    if 'error' not in args.setting:
        raise ValueError('The input "setting" is not supported. Use "error"+0.1 for 10% errors simulated in the test set.')
    else:
        try:
            error_k = float(args.setting.replace('error', ''))
            if error_k >= 1:
                print("The input error rate is not supported!")
        except IOError as err:
            print("The input error rate is not supported!")

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
            y_cols = y_cols[n_datasets:]
            sensi_cols = sensi_cols[n_datasets:]
        else:
            datasets = datasets[:n_datasets]
            y_cols = y_cols[:n_datasets]
            sensi_cols = sensi_cols[:n_datasets]

    if args.exec_n is None:
        raise ValueError(
            'The input "exec_n" is requried. Use "--exec_n 1" for a single execution.')
    elif type(args.exec_n) == str:
        raise ValueError(
            'The input "exec_n" requires integer. Use "--exec_n 1" for a single execution.')
    else:
        n_exec = int(args.exec_n)
        seeds = seeds[:n_exec]


    if args.run == 'parallel':
        tasks = []
        for data_name, y_col, sensi_col in zip(datasets, y_cols, sensi_cols):
            for seed in seeds:
                tasks.append([data_name, seed, y_col, sensi_col, res_path, error_k])
        with Pool(cpu_count()//2) as pool:
            pool.starmap(simulate_drift_data_train, tasks)
    else:
        for data_name, y_col, sensi_col in zip(datasets, y_cols, sensi_cols):
            for seed in seeds:
                simulate_drift_data_train(data_name, seed, y_col, sensi_col, res_path, error_k)
