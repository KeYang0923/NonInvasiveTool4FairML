# Evaluate multi-model and MultiCC over original test data
import warnings

import argparse
from multiprocessing import Pool, cpu_count
import pandas as pd
from utils import save_json, eval_settings
warnings.filterwarnings('ignore')


def eval_multi_models(data_name, seed, sensi_col, res_path='../intermediate/models/'):
    cur_dir = res_path + data_name + '/'

    test_df = pd.read_csv(cur_dir + '-'.join(['test_eval_multi', str(seed)]) + '.csv')

    setting_output_drift = {}
    for pred_col, setting in zip(['Y_pred', 'Y_pred_G0', 'Y_pred_G1', 'Y_pred_en', 'Y_pred_en_cc'],
                                 ['A', 'A0', 'A1', 'E', 'F']):
        setting_output_drift[setting] = eval_settings(test_df, sensi_col, pred_col)

    save_json(setting_output_drift, cur_dir + '-'.join(['Multi_eval', str(seed), 'orig']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate multi-model and MultiCC over original test data")

    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    args = parser.parse_args()


    datasets = ['adult', 'german', 'compas', 'cardio', 'bank', 'meps16', 'lawgpa', 'credit']
    y_cols = ['Income Binary', 'credit', 'two_year_recid'] + ['Y' for i in range(5)]
    sensi_cols = ['sex', 'age', 'race'] + ['C0' for i in range(5)]

    seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead,
                                                                0xdeadcafe, 0xdeadbeef, 0xbeefcafe]


    res_path = '../intermediate/models/'

    if args.run == 'parallel':
        tasks = []
        for data_name, y_col, sensi_col in zip(datasets, y_cols, sensi_cols):
            for seed in seeds:
                tasks.append([data_name, seed, sensi_col, res_path])
        with Pool(cpu_count()) as pool:
            pool.starmap(eval_multi_models, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')
