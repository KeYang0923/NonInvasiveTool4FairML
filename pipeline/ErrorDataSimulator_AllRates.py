# Simulate erroneous records in a real dataset for the evaluation of multiCC under multiple error rates

import warnings

import argparse
from multiprocessing import Pool, cpu_count
import pandas as pd
from utils import save_json

warnings.filterwarnings('ignore')

def simulate_drift_data(data_name, seed, y_col, sensi_col, res_path='../intermediate/models/', error_k=0.2,
                    n_groups=2, n_labels=2):
    cur_dir = res_path + data_name + '/'
    test_vio_df = pd.read_csv(cur_dir + '-'.join(['test_violation', str(seed)])+'.csv')
    test_df = pd.read_csv(cur_dir + '-'.join(['test', str(seed)]) + '.csv')


    # choose k percentage records from test set
    default_k = int(error_k*test_vio_df.shape[0]/(n_labels*n_groups))

    error_indexes = []
    for group_i in range(n_groups):
        for label_i in range(n_labels):
            opt_vio_col = 'vio_by_' + '_'.join(['density', 'G' + str(abs(group_i-1)), 'L' + str(abs(label_i-1))])
            gl_df = test_vio_df[(test_vio_df[sensi_col] == group_i) & (test_vio_df[y_col] == label_i)]
            sort_gl = gl_df.sort_values(by=opt_vio_col, ascending=True)
            if sort_gl.shape[0] < default_k:
                cur_k = sort_gl.shape[0]
            else:
                cur_k = default_k
            error_indexes += list(sort_gl.head(cur_k).index)

    # switch the values of the sensitive attribute
    def reverse_sensi_col(x, cand_list):
        if x.iloc[0] in cand_list:
            return abs(x.iloc[1] - 1)
        else:
            return x.iloc[1]

    test_df['ID_temp'] = test_df.index
    test_df[sensi_col] = test_df[['ID_temp', sensi_col]].apply(lambda x: reverse_sensi_col(x, error_indexes), axis=1)
    test_df.drop(columns=['ID_temp'], inplace=True)

    test_vio_df['ID_temp'] = test_vio_df.index
    test_vio_df[sensi_col] = test_vio_df[['ID_temp', sensi_col]].apply(
        lambda x: reverse_sensi_col(x, error_indexes), axis=1)
    test_vio_df.drop(columns=['ID_temp'], inplace=True)

    # print(len(error_indexes), sum(list(test_df[sensi_col] != test_vio_df[sensi_col])))

    save_json({'Error_ID': error_indexes}, '{}test_errorID-{}-error{:.2f}.csv'.format(cur_dir, seed, error_k))

    test_vio_df.to_csv('{}test_violation-{}-error{:.2f}.csv'.format(cur_dir, seed, error_k), index=False)
    test_df.to_csv('{}test-{}-error{:.2f}.csv'.format(cur_dir, seed, error_k), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simulate errors in test set under multiple rates for a dataset")
    parser.add_argument("--data", type=str,
                        help="dataset to simulate all the error rates. Choose from [adult, german, compas, cardio, bank, meps16, lawgpa, credit] for different datasets.")

    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    # parameters for running few number of executions and few error rates
    parser.add_argument("--exec_n", type=int, default=20,
                        help="number of executions with different random seeds. Default is 20.")
    parser.add_argument("--exec_k", type=int, default=30,
                        help="number of error rates. Default is 30.")

    args = parser.parse_args()


    datasets = ['adult', 'german', 'compas', 'cardio', 'bank', 'meps16', 'lawgpa', 'credit', 'UFRGS']
    y_cols_mapping = {'adult': 'Income Binary', 'german': 'credit', 'compas': 'two_year_recid',
                      'cardio': 'Y', 'bank': 'Y', 'meps16': 'Y', 'lawgpa': 'Y', 'credit': 'Y', 'UFRGS': 'Y'}

    sensi_cols_mapping = {'adult': 'sex', 'german': 'age', 'compas': 'race',
                          'cardio': 'C0', 'bank': 'C0', 'meps16': 'C0', 'lawgpa': 'C0', 'credit': 'C0', 'UFRGS': 'C0'}


    seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 0xdeadcafe, 0xdeadbeef, 0xbeefcafe]

    res_path = '../intermediate/models/'
    if args.data not in datasets:
        raise ValueError('The input "data" is not supported. Choose from [adult, german, compas, cardio, bank, meps16, lawgpa, credit, UFRGS] for different datasets.')
    else:
        data_name = args.data

    sensi_col = sensi_cols_mapping[data_name]
    y_col = y_cols_mapping[data_name]

    errors_k = [x / 100 for x in range(10, 50)]

    if args.exec_k is None:
        raise ValueError(
            'The input "exec_k" is requried. Use "--exec_k 1" for a single error rate.')
    elif type(args.exec_k) == str:
        raise ValueError(
            'The input "exec_k" requires integer. Use "--exec_k 1" for a single error rate.')
    else:
        k_exec = int(args.exec_k)
        errors_k = errors_k[:k_exec]

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
        for error_k in errors_k:
            for seed in seeds:
                tasks.append([data_name, seed, y_col, sensi_col, res_path, error_k])
        with Pool(cpu_count()//2) as pool:
            pool.starmap(simulate_drift_data, tasks)
    else:
        for error_k in errors_k:
            for seed in seeds:
                simulate_drift_data(data_name, seed, y_col, sensi_col, res_path, error_k)
