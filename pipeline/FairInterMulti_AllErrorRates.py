# Use MultiCC and multi-model to produce predictions on the test data with errors in the sensitive attribute
import warnings

import argparse
from multiprocessing import Pool, cpu_count
from FairInterMulti import apply_models_by_cc

warnings.filterwarnings('ignore')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Apply MultiCC and multi-model to predict on erroneous test data")

    parser.add_argument("--data", type=str,
                        help="dataset to simulate all the error rates. Choose from [adult, german, compas, cardio, bank, meps16, lawgpa, credit] for different datasets.")

    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    args = parser.parse_args()

    datasets = ['adult', 'german', 'compas', 'cardio', 'bank', 'meps16', 'lawgpa', 'credit']
    y_cols_mapping = {'adult': 'Income Binary', 'german': 'credit', 'compas': 'two_year_recid',
                      'cardio': 'Y', 'bank': 'Y', 'meps16': 'Y', 'lawgpa': 'Y', 'credit': 'Y'}

    sensi_cols_mapping = {'adult': 'sex', 'german': 'age', 'compas': 'race',
                          'cardio': 'C0', 'bank': 'C0', 'meps16': 'C0', 'lawgpa': 'C0', 'credit': 'C0'}

    seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 0xdeadcafe, 0xdeadbeef, 0xbeefcafe]


    res_path = '../intermediate/models/'

    if args.data not in datasets:
        raise ValueError(
            'The input "data" is not supported. Choose from [adult, german, compas, cardio, bank, meps16, lawgpa, credit] for different datasets.')
    else:
        data_name = args.data

    sensi_col = sensi_cols_mapping[data_name]
    y_col = y_cols_mapping[data_name]
    errors_k = [x / 100 for x in range(1, 30)]

    if args.run == 'parallel':
        tasks = []
        for error_k in errors_k:
            for seed in seeds:
                tasks.append([data_name, seed, y_col, sensi_col, res_path, 'error{:.2f}'.format(error_k)])
        with Pool(cpu_count()) as pool:
            pool.starmap(apply_models_by_cc, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')


