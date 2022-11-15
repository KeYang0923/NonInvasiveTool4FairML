# Evaluate SingleCC, KAM-CAL, SingleCC+KAM-CAL over original test data
import warnings

import argparse
from multiprocessing import Pool, cpu_count
import pandas as pd
from joblib import load
from utils import read_json, save_json, eval_settings
from ModelTrainer import generate_model_predictions
warnings.filterwarnings('ignore')


def load_model_lr(dir, seed, sensi_suffix, group_suffix=None):
    if group_suffix is not None:
        model = load(dir + '-'.join(['model', str(seed), group_suffix, sensi_suffix]) + '.joblib')
    else:
        model = load(dir + '-'.join(['model', str(seed), sensi_suffix]) + '.joblib')
    return model


def eval_single_models(data_name, seed, y_col, sensi_col, fair_setting, sensi_col_in_training, same_thres=False, res_path='../intermediate/models/'):
    cur_dir = res_path + data_name + '/'
    set_suffix = 'S_{}'.format(sensi_col_in_training)
    test_df = pd.read_csv(cur_dir + '-'.join(['test', str(seed), set_suffix]) + '.csv')
    if sensi_col_in_training:
        features = [x for x in test_df.columns if x != y_col]
    else:
        features = [x for x in test_df.columns if x != y_col and x != sensi_col]


    orig_model = load_model_lr(cur_dir, seed, set_suffix)
    orig_thres = read_json(cur_dir + '-'.join(['Thres', str(seed), set_suffix]))['thres']

    cur_model = load_model_lr(cur_dir, seed, set_suffix, fair_setting)
    if same_thres:
        cur_thres = orig_thres
    else:
        cur_thres = read_json(cur_dir + '-'.join(['Thres', str(seed), fair_setting, set_suffix]))['thres']
    test_data = test_df[features]
    test_df['Y_pred_cc'] = generate_model_predictions(data_name, cur_model, test_data, cur_thres)
    test_df['Y_pred'] = generate_model_predictions(data_name, orig_model, test_data, orig_thres)
    test_df['Y'] = test_df[y_col]
    test_df[[sensi_col, 'Y', 'Y_pred', 'Y_pred_cc']].to_csv(cur_dir + '-'.join(['test_eval_single', str(seed), fair_setting, set_suffix]) + '.csv', index=False)

    setting_output = {}
    for pred_col, setting in zip(['Y_pred', 'Y_pred_cc'], ['A', 'C']):
        setting_output[setting] = eval_settings(test_df, sensi_col, pred_col)
    save_json(setting_output, cur_dir + '-'.join(['Single_eval', str(seed), fair_setting, set_suffix]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate SingleCC, KAM-CAL, SingleCC+KAM-CAL over original test data")
    parser.add_argument("--setting", type=str,
                        help="method of fairness interventions. Choose from [SingleCC, KAM-CAL, SingleCC+KAM-CAL] that represent SingleCC, KAM-CAL, and SingleCC+KAM-CAL, respectively.")

    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    parser.add_argument("--sensi", type=int, default=1,
                        help="whether to include the sensitive attribute as a feature in training ML models.")

    # parameters for running over smaller number of datasets and few number of executions
    parser.add_argument("--set_n", type=int, default=9,
                        help="number of datasets over which the script is running. Default is 9 for all the datasets.")
    parser.add_argument("--exec_n", type=int, default=5,
                        help="number of executions with different random seeds. Default is 20.")
    args = parser.parse_args()

    datasets = ['adult', 'german', 'compas', 'cardio', 'bank', 'meps16', 'lawgpa', 'credit', 'UFRGS']
    y_cols = ['Income Binary', 'credit', 'two_year_recid'] + ['Y' for i in range(6)]
    sensi_cols = ['sex', 'age', 'race'] + ['C0' for i in range(6)]

    seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 0xdeadcafe, 0xdeadbeef, 0xbeefcafe]

    on_same_thres = False
    res_path = '../intermediate/models/'

    if args.setting is None:
        raise ValueError(
            'The input "setting" is requried. Choose from [SingleCC, KAM-CAL, SingleCC+KAM-CAL] that represent SingleCC, KAM-CAL, and SingleCC+KAM-CAL, respectively.')

    if args.setting not in ['SingleCC', 'KAM-CAL', 'SingleCC+KAM-CAL']:
        raise ValueError(
            'The input "setting" is not supported. Choose from [SingleCC, KAM-CAL, SingleCC+KAM-CAL] that represent SingleCC, KAM-CAL, and SingleCC+KAM-CAL, respectively.')

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
                tasks.append([data_name, seed, y_col, sensi_col, args.setting, args.sensi, on_same_thres, res_path])

        with Pool(cpu_count()//2) as pool:
            pool.starmap(eval_single_models, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')


