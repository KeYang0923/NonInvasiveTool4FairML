# Simulate erroneous records in real datasets for the evaluation of multiCC under a fixed error rate

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

    save_json({'Error_ID': error_indexes}, '{}test_errorID-{}-drift{:.2f}.csv'.format(cur_dir, seed, error_k))

    test_vio_df.to_csv('{}test_violation-{}-drift{:.2f}.csv'.format(cur_dir, seed, error_k), index=False)
    test_df.to_csv('{}test-{}-drift{:.2f}.csv'.format(cur_dir, seed, error_k), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simulate errors in test set under a fixed rate for all the datasets")
    parser.add_argument("--setting", type=str,
                        help="error rate of multiCC. When is not None, multiCC is run on test set with errors in sensitive attribute. Choose from [drift0.01, drift0.05, drift0.1, drift0.15, drift0.2] for different rates of errors in the test set.")

    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    args = parser.parse_args()


    datasets = ['adult', 'german', 'compas', 'cardio', 'bank', 'meps16', 'lawgpa', 'credit']
    y_cols = ['Income Binary', 'credit', 'two_year_recid'] + ['Y' for i in range(5)]
    sensi_cols = ['sex', 'age', 'race'] + ['C0' for i in range(5)]

    seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 0xdeadcafe, 0xdeadbeef, 0xbeefcafe]

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

    if args.run == 'parallel':
        tasks = []
        for data_name, y_col, sensi_col in zip(datasets[1:2], y_cols[1:2], sensi_cols[1:2]):
            for seed in seeds[:3]:
                tasks.append([data_name, seed, y_col, sensi_col, res_path, error_k])
        with Pool(cpu_count()) as pool:
            pool.starmap(simulate_drift_data, tasks)
    else:
        for data_name, y_col, sensi_col in zip(datasets, y_cols, sensi_cols):
            for seed in seeds:
                simulate_drift_data(data_name, seed, y_col, sensi_col, res_path, error_k)
